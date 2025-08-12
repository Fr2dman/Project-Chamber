import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from simulator.environment import AdvancedSmartACSimulator

class HVACEnv(gym.Env):
    """SB3 학습용 Gymnasium 래퍼"""
    def __init__(self, **sim_kwargs):
        super().__init__()
        # 초기조건/외기조건 override는 따로 보관
        init_T = sim_kwargs.pop("init_temperatures", None)
        init_H = sim_kwargs.pop("init_humidities", None)
        amb_T  = sim_kwargs.pop("ambient_temp", None)
        amb_H  = sim_kwargs.pop("ambient_hum", None)

        self._init_override = (init_T, init_H)
        self._ambient_override = (amb_T, amb_H)

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
        # 생성 직후 1회 환경 override 적용
        amb_T, amb_H = self._ambient_override
        if amb_T is not None: self.sim.physics_sim.ambient_temp = float(amb_T)
        if amb_H is not None: self.sim.physics_sim.ambient_hum  = float(amb_H)
        init_T, init_H = self._init_override
        if init_T is not None and init_H is not None:
            self.sim.set_initial_state(list(init_T), list(init_H))
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.sim.step(np.asarray(action, dtype=np.float32))
        obs = obs.astype(np.float32, copy=False)
        terminated = bool(done)                     # 내부 종료 신호
        truncated = False                           # 타임리밋은 외부 TimeLimit 래퍼가 처리
        return obs, reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass

# --- 혼합 초기조건 랜덤화 래퍼 ---
class RandomizeInitWrapper(gym.Wrapper):
    """
    매 reset마다 초기 온도/습도를 지정한 버킷/범위에서 샘플링해 set_initial_state()로 주입.
    env_kwargs 예시:
      reset_randomizer:
        temp_C:
          buckets: [[23.0, 26.0], [26.0, 30.0]]
          probs:   [0.5, 0.5]
        rh_pct:
          range: [50.0, 80.0]
    """
    def __init__(self, env, reset_randomizer: dict):
        super().__init__(env)
        self.cfg = reset_randomizer or {}

    def _sample_val(self, spec: dict):
        if not isinstance(spec, dict):
            return None
        if "buckets" in spec:
            buckets = spec["buckets"]
            probs = spec.get("probs", None)
            idx = np.random.choice(len(buckets), p=probs if probs is not None else None)
            lo, hi = buckets[idx]
        else:
            lo, hi = spec.get("low") or spec.get("range", [None, None])
            if isinstance(lo, (list, tuple)): lo = lo[0]
            if isinstance(hi, (list, tuple)): hi = hi[1]
        return float(np.random.uniform(float(lo), float(hi)))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        try:
            # 존 개수 추정
            n = len(self.env.sim.physics_sim.T)
        except Exception:
            n = 4
        t_spec = (self.cfg.get("temp_C") or self.cfg.get("temperature"))
        h_spec = (self.cfg.get("rh_pct")  or self.cfg.get("humidity"))
        if t_spec is not None and h_spec is not None:
            t = [self._sample_val(t_spec) for _ in range(n)]
            h = [self._sample_val(h_spec) for _ in range(n)]
            try:
                self.env.sim.set_initial_state(t, h)
            except Exception:
                pass
        return obs, info