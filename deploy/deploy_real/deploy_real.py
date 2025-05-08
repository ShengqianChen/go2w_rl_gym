from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch     

# region ########### Unitree通信相关导入 ###########
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
# endregion

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

class Controller:
    def __init__(self, config: Config) -> None:
        # region ########### 初始化阶段 ###########
        # 1. 配置加载
        self.config = config # 加载配置
        self.remote_controller = RemoteController() # 初始化远程控制器

        # 2. 策略网络加载
        self.policy = torch.jit.load(config.policy_path) # 加载训练好的策略网络模型
        
        # 3. 过程变量初始化
        self.qj = np.zeros(config.num_actions, dtype=np.float32) # 初始化关节位置
        self.dqj = np.zeros(config.num_actions, dtype=np.float32) # 初始化关节速度
        self.action = np.zeros(config.num_actions, dtype=np.float32) # 初始化action
        self.target_dof_pos = config.default_angles.copy() # 复制默认关节角度作为目标位置
        self.obs = np.zeros(config.num_obs, dtype=np.float32) # 初始化观测向量
        self.cmd = np.array([0.0, 0, 0]) # 初始化速度指令
        self.counter = 0 # 初始化计数器

        # 4. DDS通信初始化
        if config.msg_type == "hg":
            # H1 Gen2使用hg消息类型
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # H1 Gen1使用go消息类型
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # 5. 等待底层状态连接
        self.wait_for_low_state()

        # 6. 指令消息初始化
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        # endregion

        # 上面是人形机器人如何收发命令，机器狗需要更改底层传输的命令

    # region ########### 状态回调处理 ###########
    def LowStateHgHandler(self, msg: LowStateHG):
        """处理H1 Gen2的低层状态消息"""
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        """处理H1 Gen1的低层状态消息""" 
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    # endregion

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]): # 发送指令
        """发送指令（自动添加CRC校验）"""
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd) 

    def wait_for_low_state(self): # 等待命令
        """等待直到收到底层状态数据"""
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # region ########### 状态机流程 ###########
    def zero_torque_state(self):
        """
        零力矩状态（安全准备阶段）
        1. 发送零力矩指令
        2. 等待Start按键触发
        """
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd) # 发送零力矩命令
            time.sleep(self.config.control_dt)  # 这个控制零力矩命令就是把所有电机都设置为0不用改

    def move_to_default_pos(self):
        """
        平滑移动到默认姿态（耗时2秒）
        使用线性插值生成关节轨迹
        """
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt) # 计算需要几步
        
        # 组合腿部与手臂/腰部的关节索引
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # 记录初始位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # 执行插值运动
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        """
        默认姿态保持状态
        1. 维持默认关节位置
        2. 等待A按键触发
        """
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # 腿部关节控制
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            # 手臂/腰部关节控制（机器狗可移除）
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    # endregion

    def run(self):
        """主控制循环（每个控制周期执行）"""
        self.counter += 1
        
        # region ########### 状态采集 ###########
        # 1. 关节状态采集
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # 2. IMU数据处理
        quat = self.low_state.imu_state.quaternion  # 四元数格式: w, x, y, z
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # 人形机器人需要转换IMU坐标系（机器狗可能不需要）
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel
            )
        # endregion

        # region ########### 观测构建 ###########
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        
        # 步态相位生成（机器狗需要调整）
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # 遥控指令处理
        self.cmd[0] = self.remote_controller.ly  # 前后速度
        self.cmd[1] = self.remote_controller.lx * -1  # 横向速度
        self.cmd[2] = self.remote_controller.rx * -1  # 旋转速度

        # 观测向量构建
        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel                # 角速度
        self.obs[3:6] = gravity_orientation    # 重力方向
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd  # 速度指令
        self.obs[9:9+num_actions] = qj_obs     # 关节位置
        self.obs[9+num_actions:9+num_actions*2] = dqj_obs  # 关节速度
        self.obs[9+num_actions*2:9+num_actions*3] = self.action  # 历史动作
        self.obs[9+num_actions*3] = sin_phase  # 步态正弦相位
        self.obs[9+num_actions*3+1] = cos_phase# 步态余弦相位
        # endregion

        # region ########### 策略推理 ###########
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        # 上面的代码都在生成观测数据


        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        # 把观测数据喂到模型里得到action


        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        # 计算关节目标位置

        # endregion

        # region ########### 指令发送 ###########
        # 腿部关节指令设置
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 手臂/腰部指令设置（机器狗可移除）
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 上面就是描述如何发送指令，直接改用机器狗的逻辑就行

        self.send_cmd(self.low_cmd) # 把命令发送过去
        time.sleep(self.config.control_dt) # 等待控制周期
        # endregion

if __name__ == "__main__":
    # region ########### 程序入口 ###########
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name", default="g1.yaml")
    args = parser.parse_args()

    # 加载配置文件
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # 初始化DDS通信
    ChannelFactoryInitialize(0, args.net)

    # 创建控制器实例
    controller = Controller(config)

    # region ########### 状态机执行流程 ###########
    # 阶段1: 零力矩模式
    controller.zero_torque_state()
    
    # 阶段2: 调试模式（阻尼状态）
    controller.move_to_default_pos()
    
    # 阶段3: 默认姿态保持
    controller.default_pos_state()
    
    # 阶段4: 主控制循环
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # 阶段5: 退出时进入阻尼模式
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    # endregion
    # endregion