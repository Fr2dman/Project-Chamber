# rl/sb3/callbacks.py
"""
Custom callbacks for SB3 training on the Smart-AC simulator.

Included
--------
- VecNormSyncCallback:   train/env ↔ eval/env 의 VecNormalize 통계 동기화
- RewardBreakdownLogger: info['reward_breakdown']를 텐서보드로 주기 로그
- SuccessEvalCallback:   eval_env로 성공률(success_rate) 측정 + 얼리스톱

사용 예 (train_sac_sb3.py)
--------------------------
from rl.sb3.callbacks import (
    VecNormSyncCallback, RewardBreakdownLogger, SuccessEvalCallback
)
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback

vns_cb  = VecNormSyncCallback(train_env, eval_env)
rb_cb   = RewardBreakdownLogger(log_every=env_cfg.get("logging", {}).get("log_every_steps", 1000))
eval_cb = EvalCallback(eval_env, best_model_save_path=str(best_model_dir),
                       log_path=str(eval_log_path), eval_freq=eval_freq,
                       n_eval_episodes=n_eval_episodes, deterministic=True)

# 성공률 기반 얼리스톱(옵션): patience=6 → 6번의 평가 동안 개선 없으면 중단
succ_cb = SuccessEvalCallback(
    eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
    success_keys=("success", "comfort_ok", "is_success", "episode_success"),
    monitor="success", patience=6, min_delta=0.01, deterministic=True
)

ckpt_cb = CheckpointCallback(save_freq=max(eval_freq, 10_000),
                             save_path=str(checkpoints_dir),
                             name_prefix="sac", save_replay_buffer=True, save_vecnormalize=True)

callbacks = CallbackList([vns_cb, eval_cb, succ_cb, rb_cb, ProgressBarCallback()])
model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _sync_vecnormalize_stats(src_env: VecEnv, dst_env: VecEnv) -> None:
    """Copy running stats if both envs are VecNormalize-wrapped."""
    if isinstance(src_env, VecNormalize) and isinstance(dst_env, VecNormalize):
        dst_env.obs_rms = src_env.obs_rms
        dst_env.ret_rms = src_env.ret_rms
        dst_env.clip_obs = src_env.clip_obs
        dst_env.clip_reward = src_env.clip_reward
        dst_env.gamma = src_env.gamma


def _extract_reward_terms(info: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """
    Try to pull a dict-like reward breakdown from info.
    Preferred: info['reward_breakdown'] (dict of floats)
    Fallback:  keys starting with 'r_' or 'R_' (e.g., 'r_energy', 'R_track')
    """
    if not isinstance(info, Mapping):
        return None
    if "reward_breakdown" in info and isinstance(info["reward_breakdown"], Mapping):
        # flatten & only keep numeric
        out = {}
        for k, v in info["reward_breakdown"].items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out if out else None

    # Heuristic fallback
    out = {}
    for k, v in info.items():
        if isinstance(k, str) and (k.startswith("r_") or k.startswith("R_")):
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out if out else None


def _any_success(info: Mapping[str, Any], keys: Sequence[str]) -> bool:
    """Return True if any of the provided keys exists and is truthy in info."""
    for k in keys:
        if bool(info.get(k, False)):
            return True
    return False


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
class VecNormSyncCallback(BaseCallback):
    """
    Keep eval VecNormalize stats synchronized with train VecNormalize stats.
    - 가벼운 연산이므로 매 스텝 호출해도 무방합니다.
    """
    def __init__(self, train_env: VecEnv, eval_env: VecEnv, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        try:
            _sync_vecnormalize_stats(self.train_env, self.eval_env)
        except Exception:
            pass
        return True


class RewardBreakdownLogger(BaseCallback):
    """
    Aggregate and log reward component averages to TensorBoard at fixed step intervals.

    기대 포맷:
      info['reward_breakdown'] = {'track': ..., 'energy': ..., 'gate': ..., ...}
    또는
      info['r_track'], info['r_energy'] 와 같은 키들

    Parameters
    ----------
    log_every : int
        몇 스텝마다 평균치를 기록할지(기본 1000).
    prefix : str
        텐서보드 키 prefix (기본 'train/reward_terms').
    """
    def __init__(self, log_every: int = 1000, prefix: str = "train/reward_terms", verbose: int = 0):
        super().__init__(verbose)
        self.log_every = int(log_every)
        self.prefix = prefix
        self._sum: Dict[str, float] = {}
        self._cnt: Dict[str, int] = {}
        self._last_logged_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            for info in infos:
                terms = _extract_reward_terms(info)
                if not terms:
                    continue
                for k, v in terms.items():
                    self._sum[k] = self._sum.get(k, 0.0) + float(v)
                    self._cnt[k] = self._cnt.get(k, 0) + 1

        # 주기적 로그
        if self.n_calls - self._last_logged_step >= self.log_every and self._cnt:
            for k, c in list(self._cnt.items()):
                if c <= 0:
                    continue
                mean_val = self._sum.get(k, 0.0) / float(c)
                self.logger.record(f"{self.prefix}/{k}", mean_val)
            self._sum.clear()
            self._cnt.clear()
            self._last_logged_step = self.n_calls
        return True


class SuccessEvalCallback(BaseCallback):
    """
    주기적으로 eval_env에서 에피소드 N개를 돌려 성공률 및 평균 reward를 측정.
    선택적으로 개선 정체 시 얼리스톱.

    Parameters
    ----------
    eval_env : VecEnv
        평가용 환경 (build_vec_env(..., is_eval=True) 권장. 통상 num_envs=1)
    eval_freq : int
        몇 스텝마다 평가할지.
    n_eval_episodes : int
        에피소드 수.
    success_keys : Sequence[str]
        info 딕셔너리에서 성공을 나타내는 키 후보들 (하나라도 True면 성공).
    monitor : {'success', 'reward'}
        얼리스톱에서 향상을 감시할 메트릭.
    patience : int | None
        개선 없을 때 허용하는 평가회수. None이면 얼리스톱 비활성.
    min_delta : float
        향상으로 간주할 최소 개선폭.
    deterministic : bool
        평가 시 정책 결정론적 예측 여부.
    """
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        success_keys: Sequence[str] = ("success", "is_success", "comfort_ok", "episode_success"),
        monitor: str = "success",
        patience: Optional[int] = None,
        min_delta: float = 1e-3,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.success_keys = tuple(success_keys)
        assert monitor in ("success", "reward")
        self.monitor = monitor
        self.patience = None if patience is None else int(max(1, patience))
        self.min_delta = float(min_delta)
        self.deterministic = bool(deterministic)

        self._best: Optional[float] = None
        self._wait: int = 0

    # --------- internal evaluation loop ----------
    def _evaluate(self) -> Tuple[float, float]:
        """
        Returns
        -------
        success_rate, mean_reward
        """
        # VecEnv 가정: eval_env는 DummyVecEnv(1) + (옵션) VecNormalize
        successes = 0
        ep_rewards = []

        # 동기화는 외부(VecNormSyncCallback)에서 수행했다고 가정하되,
        # 혹시 누락돼도 큰 문제는 없음.
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_rew = 0.0
            # VecEnv에서 단일 env 가정 → 인덱스 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, infos = self.eval_env.step(action)
                ep_rew += float(reward[0] if isinstance(reward, np.ndarray) else reward)
                info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                if _any_success(info0, self.success_keys):
                    # 에피소드 내 한 번이라도 성공 시 '성공'으로 간주
                    successes += 1
                done = bool(terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated) \
                    or bool(truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated)
            ep_rewards.append(ep_rew)

        success_rate = successes / float(self.n_eval_episodes)
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        # 텐서보드 로깅
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward_custom", mean_reward)

        return success_rate, mean_reward

    def _improved(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return (current - best) > self.min_delta

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        success_rate, mean_reward = self._evaluate()
        metric = success_rate if self.monitor == "success" else mean_reward

        # 얼리스톱 로직
        if self.patience is not None:
            if self._improved(metric, self._best):
                self._best = metric
                self._wait = 0
            else:
                self._wait += 1
                if self.verbose > 0:
                    print(f"[SuccessEvalCallback] No improvement on {self.monitor} "
                          f"({metric:.4f} vs best {self._best if self._best is not None else float('nan'):.4f}) "
                          f"for {self._wait}/{self.patience} evals.")
                if self._wait >= self.patience:
                    if self.verbose > 0:
                        print("[SuccessEvalCallback] Early stopping triggered.")
                    return False  # returning False stops training
        else:
            # best만 갱신
            if self._improved(metric, self._best):
                self._best = metric

        return True


__all__ = [
    "VecNormSyncCallback",
    "RewardBreakdownLogger",
    "SuccessEvalCallback",
]
