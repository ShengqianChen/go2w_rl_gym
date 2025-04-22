import sys
sys.path.append("/home/hu/csq/unitree_rl_gym")
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 50
    env_cfg.noise.add_noise = False # 禁用噪声
    env_cfg.domain_rand.randomize_friction = False # 摩擦系数随机化
    env_cfg.domain_rand.push_robots = False # 对机器人施加外部扰动

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations() # 获得观测的环境信息
    obs[:,6] = 0.0
    obs[:,7] = 0.0
    obs[:,8] = 1.0
        
    obs_size = obs.size()

    print("Observation tensor size:", obs_size)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    action_file_path = "/home/hu/csq/unitree_rl_gym/deploy/actions_sim.log"
    obs_file_path = "/home/hu/csq/unitree_rl_gym/deploy/obs_sim.log"

    # 将观测数据追加到文件中
    with open(obs_file_path, "a") as obs_file:
        obs_file.write(",".join(map(str, obs.cpu().detach().numpy()[0])) + "\n")

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach()) # 将张量obs从计算图中分离出来，避免梯度传播

        # with open(action_file_path, "a") as action_file:
        #     action_file.write(",".join(map(str, actions.cpu().detach().numpy()[0])) + "\n")
        
        obs, _, rews, dones, infos = env.step(actions.detach()) # 获得新的观测
        obs[:,6] = 0.0
        obs[:,7] = 0.0
        obs[:,8] = 1.0
        
        # 将观测数据追加到文件中
        with open(obs_file_path, "a") as obs_file:
            obs_file.write(",".join(map(str, obs.cpu().detach().numpy()[0])) + "\n")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
