# sb3/make_env.py
from typing import Callable, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from rl.sb3.wrappers import HVACEnv

def make_env_fn(env_kwargs: Dict[str, Any], max_steps: int, seed: int) -> Callable[[], gym.Env]:
    def _thunk():
        env = HVACEnv(**env_kwargs)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env
    return _thunk
