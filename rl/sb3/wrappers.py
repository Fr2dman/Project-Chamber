import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from simulator.environment import AdvancedSmartACSimulator

class HVACEnv(gym.Env):
    """SB3 학습용 Gymnasium 래퍼"""
    def __init__(self, fixed_targets: list[float] | None = None, **sim_kwargs):
        super().__init__()
        self.sim = AdvancedSmartACSimulator(**sim_kwargs)
        # 평가 등에서 에피소드마다 동일한 목표를 강제하고 싶을 때 사용
        self._fixed_targets = None
        if fixed_targets is not None:
            self._set_targets_safely(fixed_targets)

        # 관측/행동 공간 정의
        obs0 = self.sim.reset()                     # env.reset()이 obs만 반환
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
        )
        # action = [-1,1] 정규화 14차원 (펠티어1 + 내부서보4 + 외부서보4 + 소형팬4 + 대형팬1)
        self.action_space = Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

    # --------- 유틸: 평가 시 고정 목표온도 강제 ----------
    def _set_targets_safely(self, targets: list[float]):
        targets = list(map(float, targets))
        if len(targets) != self.sim.num_zones:
            raise ValueError(f"fixed_targets length must be {self.sim.num_zones}")
        # 환경이 helper를 갖고 있으면 사용, 아니면 속성 직접 설정
        if hasattr(self.sim, "set_target_temperatures"):
            self.sim.set_target_temperatures(targets)
        else:
            import numpy as np
            self.sim.T_target = np.asarray(targets, dtype=float)
        self._fixed_targets = targets

    def set_fixed_targets(self, targets: list[float] | None):
        """런타임에 평가 목표를 지정/해제합니다. None이면 해제."""
        if targets is None:
            self._fixed_targets = None
            return
        self._set_targets_safely(targets)
    # ----------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        # Gymnasium seeding 권장 방식
        if seed is not None:
            try:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            except Exception:
                np.random.seed(seed)
        obs = self.sim.reset().astype(np.float32, copy=False)
        # 고정 목표가 설정되어 있으면 reset 이후 강제로 주입 (환경 내부 랜덤화 무력화)
        if self._fixed_targets is not None:
            self._set_targets_safely(self._fixed_targets)
            # 목표가 바뀌었으니 관측을 다시 뽑아 일관성 유지(선택)
            obs = self.sim._get_state_vector().astype(np.float32, copy=False)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.sim.step(np.asarray(action, dtype=np.float32))
        obs = obs.astype(np.float32, copy=False)
        terminated = bool(done)                     # 내부 종료 신호
        truncated = False                           # 타임리밋은 외부 TimeLimit 래퍼가 처리
        return obs, reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass