import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from simulator.environment import AdvancedSmartACSimulator

class HVACEnv(gym.Env):
    """SB3 학습용 Gymnasium 래퍼"""
    def __init__(self, **sim_kwargs):
        super().__init__()
        self.sim = AdvancedSmartACSimulator(**sim_kwargs)

        # 관측/행동 공간 정의
        obs0 = self.sim.reset()                     # env.reset()이 obs만 반환
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
        )
        # action = [-1,1] 정규화 14차원 (펠티어1 + 내부서보4 + 외부서보4 + 소형팬4 + 대형팬1)
        self.action_space = Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.sim.reset().astype(np.float32, copy=False)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.sim.step(np.asarray(action, dtype=np.float32))
        obs = obs.astype(np.float32, copy=False)
        terminated = bool(done)                     # 내부 종료 신호
        truncated = False                           # 타임리밋은 외부 TimeLimit 래퍼가 처리
        return obs, reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass
