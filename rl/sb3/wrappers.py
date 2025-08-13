# rl/sb3/wrappers.py
"""
Lightweight Gymnasium wrappers used with SB3 for the Smart-AC simulator.

Included:
- ActionScale: map policy output in [-1, 1] to env.action_space [low, high]
- ClipAction: hard-clip actions to env bounds (safety)
- DictFlattenObs: concat Dict observation into a single Box vector (key order fixed)
- RewardScale: multiply reward by a constant factor
- RewardClip: clip reward to [min_r, max_r]
- SafetyEarlyTerminate: if env info signals safety violation, end episode
- PrevActionObs: append previous action to the observation vector

Notes
-----
* Use VecNormalize for running mean/var normalization. These wrappers are complementary.
* If your env already expects physical-range actions (not [-1,1]), keep ActionScale.
  If your env already expects normalized actions [-1,1], DO NOT use ActionScale.

Example
-------
from rl.sb3.wrappers import (
    ActionScale, ClipAction, DictFlattenObs, RewardScale, RewardClip,
    SafetyEarlyTerminate, PrevActionObs
)

env = AdvancedSmartACSimulator(...)
env = TimeLimit(env, max_episode_steps=720)
env = Monitor(env)

# Pick what you need ↓
env = DictFlattenObs(env, keys=["temps","humidities","tsv","fans_S","fan_L","theta_int","theta_ext"])
env = ActionScale(env)          # only if policy outputs in [-1,1] and env expects physical ranges
env = ClipAction(env)
env = RewardScale(env, scale=1.0)
env = RewardClip(env, min_r=-5.0, max_r=5.0)
env = SafetyEarlyTerminate(env, info_flag="safety_violation")
env = PrevActionObs(env)        # optional: improves stability in control tasks
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Action wrappers
# -----------------------------
class ActionScale(gym.ActionWrapper):
    """
    Map actions from [-1, 1] to env.action_space.low/high.

    a_env = low + (a_policy + 1) * 0.5 * (high - low)
    """
    def __init__(self, env: gym.Env, low: Optional[np.ndarray] = None, high: Optional[np.ndarray] = None):
        super().__init__(env)
        assert isinstance(self.action_space, spaces.Box), "ActionScale requires Box action_space"
        self._low = self.action_space.low if low is None else np.asarray(low, dtype=np.float32)
        self._high = self.action_space.high if high is None else np.asarray(high, dtype=np.float32)
        assert self._low.shape == self._high.shape, "low/high shape mismatch"
        # policy sees normalized space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self._low.shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        scaled = self._low + (action + 1.0) * 0.5 * (self._high - self._low)
        return scaled


class ClipAction(gym.ActionWrapper):
    """Hard-clip actions to environment's action_space bounds."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(self.action_space, spaces.Box), "ClipAction requires Box action_space"

    def action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.env.action_space.low, self.env.action_space.high
        return np.clip(action, low, high)


# -----------------------------
# Observation wrappers
# -----------------------------
class DictFlattenObs(gym.ObservationWrapper):
    """
    Concatenate Dict observations into a single 1D Box vector.

    Parameters
    ----------
    keys : list[str] | None
        Order of keys to concatenate. If None, uses sorted(dict.keys()) to fix order.
    dtype : np.dtype
        Output dtype (default float32).
    """
    def __init__(self, env: gym.Env, keys: Optional[Sequence[str]] = None, dtype=np.float32):
        super().__init__(env)
        assert isinstance(env.observation_space, (spaces.Dict, spaces.Box)), \
            "DictFlattenObs expects Dict or Box observation_space"

        self.dtype = dtype
        if isinstance(env.observation_space, spaces.Box):
            # Nothing to do; passthrough
            self._is_passthrough = True
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=dtype
            )
            self._keys: List[str] = []
            return

        self._is_passthrough = False
        obs_space: spaces.Dict = env.observation_space
        all_keys = list(obs_space.spaces.keys())
        self._keys = list(keys) if keys is not None else sorted(all_keys)

        lows: List[np.ndarray] = []
        highs: List[np.ndarray] = []
        for k in self._keys:
            space_k = obs_space.spaces[k]
            assert isinstance(space_k, spaces.Box), f"Key '{k}' must be Box in DictFlattenObs"
            lows.append(space_k.low.flatten())
            highs.append(space_k.high.flatten())

        low = np.concatenate(lows, axis=0).astype(dtype)
        high = np.concatenate(highs, axis=0).astype(dtype)
        self.observation_space = spaces.Box(low=low, high=high, dtype=dtype)

    def observation(self, observation: Any) -> np.ndarray:
        if self._is_passthrough:
            return np.asarray(observation, dtype=self.dtype)
        parts: List[np.ndarray] = []
        for k in self._keys:
            v = np.asarray(observation[k], dtype=self.dtype).reshape(-1)
            parts.append(v)
        return np.concatenate(parts, axis=0)


class PrevActionObs(gym.Wrapper):
    """
    Append previous action to the observation vector (Box→Box).

    * Works when observation is a 1D Box (use DictFlattenObs beforehand if needed).
    * Adds zeros on the first step after reset.

    new_obs = concat([obs, prev_action])
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), "PrevActionObs requires Box observation"
        assert len(env.observation_space.shape) == 1, "PrevActionObs expects 1D Box observation"

        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        low = np.concatenate([
            np.asarray(env.observation_space.low, dtype=np.float32),
            np.full(self.action_space.shape, -np.inf, dtype=np.float32)
        ])
        high = np.concatenate([
            np.asarray(env.observation_space.high, dtype=np.float32),
            np.full(self.action_space.shape,  np.inf, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_aug = self._augment(obs)
        self._prev_action = np.asarray(action, dtype=np.float32)
        return obs_aug, reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([obs, self._prev_action.reshape(-1)], axis=0)


# -----------------------------
# Reward wrappers
# -----------------------------
class RewardScale(gym.RewardWrapper):
    """Multiply rewards by a constant."""
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return reward * self.scale


class RewardClip(gym.RewardWrapper):
    """Clip rewards to [min_r, max_r]."""
    def __init__(self, env: gym.Env, min_r: float = -10.0, max_r: float = 10.0):
        super().__init__(env)
        assert min_r < max_r
        self.min_r = float(min_r)
        self.max_r = float(max_r)

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.min_r, self.max_r))


# -----------------------------
# Safety wrapper
# -----------------------------
class SafetyEarlyTerminate(gym.Wrapper):
    """
    If the env reports a safety violation via info[info_flag] == True,
    convert it into an early termination (terminated=True). Optionally,
    apply an extra penalty.

    Parameters
    ----------
    info_flag : str
        Key in info dict that signals a safety violation (default: "safety_violation").
    penalty : float
        Extra negative reward applied once at termination (default: 0.0).
    """
    def __init__(self, env: gym.Env, info_flag: str = "safety_violation", penalty: float = 0.0):
        super().__init__(env)
        self.info_flag = info_flag
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if bool(info.get(self.info_flag, False)):
            terminated = True
            if self.penalty != 0.0:
                reward = reward - abs(self.penalty)
            info["safety_early_terminated"] = True
        return obs, reward, terminated, truncated, info


__all__ = [
    "ActionScale",
    "ClipAction",
    "DictFlattenObs",
    "RewardScale",
    "RewardClip",
    "SafetyEarlyTerminate",
    "PrevActionObs",
]
