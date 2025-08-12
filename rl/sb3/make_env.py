# sb3/make_env.py
from typing import Callable, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from rl.sb3.wrappers import HVACEnv, RandomizeInitWrapper

def make_env_fn(env_kwargs: Dict[str, Any], max_steps: int, seed: int) -> Callable[[], gym.Env]:
    def _thunk():
        env = HVACEnv(**env_kwargs)
        # reset_randomizer 설정이 있으면 혼합 초기화 래핑
        rr = env_kwargs.get("reset_randomizer", None)
        if rr:
            env = RandomizeInitWrapper(env, rr)
        env = TimeLimit(env, max_episode_steps=max_steps)
        # 재현성 확보: 환경 생성 시 1회 시드 주입
        env.reset(seed=seed)
        try:
            # 선택: space들도 시드 동일화(노이즈/샘플러 사용하는 경우 유용)
            if hasattr(env, "action_space"):      env.action_space.seed(seed)
            if hasattr(env, "observation_space"): env.observation_space.seed(seed)
        except Exception:
            pass
        return env
    return _thunk

# eval 스크립트가 import하는 헬퍼 (DummyVecEnv 한 개짜리 생성용)
def make_env(seed: int = 42, env_kwargs: Dict[str, Any] | None = None, max_steps: int = 720) -> gym.Env:
    env_kwargs = env_kwargs or {}
    env = HVACEnv(**env_kwargs)
    rr = env_kwargs.get("reset_randomizer", None)
    if rr:
        env = RandomizeInitWrapper(env, rr)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    return env