from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from .base.legged_robot import LeggedRobot
from .go2w.go2w_robot import Go2w

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2w", Go2w, GO2WRoughCfg(), GO2WRoughCfgPPO())
