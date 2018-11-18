import os.path as osp

import gym

from monitor import Monitor
import logger


MUJOCO_ENV_VERSION = 'v2'
ATARI_ENV_VERSION = 'v4'
MUJOCO_ENV_NAMES = ['InvertedPendulum', 'InvertedDoublePendulum', 'Reacher',
                    'Hopper', 'HalfCheetah', 'Ant', 'Walker2d',
                    'Humanoid', 'HumanoidStandup']
ATARI_ENV_NAMES = ['FrostbiteNoFrameskip', 'BreakoutNoFrameskip', 'AlienNoFrameskip',
                   'MsPacmanNoFrameskip', 'MontezumaRevengeNoFrameskip', 'PongNoFrameskip',
                   'PitfallNoFrameskip', 'QbertNoFrameskip', 'SeaquestNoFrameskip',
                   'SolarisNoFrameskip', 'WizardOfWorNoFrameskip', 'SpaceInvadersNoFrameskip']


def assert_admissibility(env_id):
    """Verify that the specified env is amongst the admissible ones"""
    adm_mujoco_env = env_id in [name + '-' + MUJOCO_ENV_VERSION for name in MUJOCO_ENV_NAMES]
    adm_atari_env = env_id in [name + '-' + ATARI_ENV_VERSION for name in ATARI_ENV_NAMES]
    assert adm_mujoco_env or adm_atari_env, "non-admissible env, refer to 'env_makers.py'"
    logger.info("admissibility check passed for env '{}'".format(env_id))


def make_mujoco_env(env_id, seed, name, horizon=None, allow_early_resets=False):
    """Create a wrapped, monitored gym.Env for MuJoCo"""
    assert_admissibility(env_id)
    env = gym.make(env_id)
    if horizon is not None:
        # Override the default episode horizon
        # by hacking the private attribute of the `TimeLimit` wrapped env
        env._max_episode_steps = horizon
    # Wrap the `env` with `Monitor`
    env = Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), name),
                  allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env


def make_atari_env(env_id, seed, name, horizon=None, allow_early_resets=False):
    """Create a wrapped, monitored gym.Env for Atari"""
    assert_admissibility(env_id)
    from atari_wrappers import make_atari, wrap_deepmind
    env = make_atari(env_id)
    if horizon is not None:
        # Override the default episode horizon
        # by hacking the private attribute of the `TimeLimit` wrapped env
        env._max_episode_steps = horizon
    # Wrap the `env` with `Monitor`
    env = Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), name),
                  allow_early_resets=allow_early_resets)
    env.seed(seed)
    # Wrap (second wrapper) with DeepMind's wrapper
    env = wrap_deepmind(env)
    env.seed(seed)
    return env


def make_env(env_id, seed, name, horizon=None, allow_early_resets=None):
    """Create an environment"""
    env_id_stem = env_id.split('-')[0]
    if env_id_stem in MUJOCO_ENV_NAMES:
        _make_env = make_mujoco_env
    elif env_id_stem in ATARI_ENV_NAMES:
        _make_env = make_atari_env
    else:
        raise RuntimeError("unknown benchmark, check what's available in 'env_makers.py'")
    env = _make_env(env_id, seed, name, horizon, allow_early_resets)
    return env
