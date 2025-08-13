# rl/sb3/make_env.py
"""
Factory utilities to build SB3-ready VecEnvs for the Smart-AC simulator.

- Reads the same schema you use in configs/env.yaml and configs/sb3_sac.yaml
- Creates SubprocVecEnv (train) / DummyVecEnv (eval) with TimeLimit + Monitor
- Applies VecNormalize if declared in env.yaml's `wrappers` list:
    wrappers:
      - VecNormalize:
          norm_obs: true
          norm_reward: false
          clip_obs: 10.0
          gamma: 0.995
- Provides a safe way to keep eval's VecNormalize statistics in sync with train

Typical usage (inside your training script):
    from rl.sb3.make_env import build_vec_env, sync_vecnormalize_stats

    train_env, _ = build_vec_env(env_cfg, algo_env_cfg, logs_dir, is_eval=False)
    eval_env,  _ = build_vec_env(env_cfg, algo_env_cfg, logs_dir, is_eval=True)
    sync_vecnormalize_stats(train_env, eval_env)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import importlib
import numpy as np
import yaml

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

# ---------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _import_env_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _extract_vecnorm_cfg(env_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse env.yaml wrappers section. If a VecNormalize block exists, return its config.
    Example:
      wrappers:
        - VecNormalize:
            norm_obs: true
            norm_reward: false
            clip_obs: 10.0
            gamma: 0.995
    """
    wrappers = env_cfg.get("wrappers", [])
    for item in wrappers:
        if isinstance(item, dict) and "VecNormalize" in item:
            cfg = item["VecNormalize"] or {}
            return {
                "norm_obs": bool(cfg.get("norm_obs", True)),
                "norm_reward": bool(cfg.get("norm_reward", False)),
                "clip_obs": float(cfg.get("clip_obs", 10.0)),
                "gamma": float(cfg.get("gamma", 0.995)),
            }
    return None


def _make_single_env_ctor(
    env_module: str,
    env_class: str,
    env_kwargs: Dict[str, Any],
    max_episode_steps: int,
    monitor_dir: Path,
    seed: int,
) -> Callable[[], Any]:
    """
    Returns a picklable thunk that creates a single env, wrapped with TimeLimit & Monitor.
    Suitable for SubprocVecEnv.
    """
    def _thunk():
        EnvCls = _import_env_class(env_module, env_class)
        env = EnvCls(**(env_kwargs or {}))
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env, filename=str(monitor_dir / "monitor.csv"), allow_early_resets=True)
        env.reset(seed=seed)
        return env
    return _thunk


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_vec_env(
    env_cfg: Dict[str, Any],
    algo_env_cfg: Optional[Dict[str, Any]],
    log_root: Path,
    is_eval: bool = False,
) -> Tuple[VecEnv, Optional[Dict[str, Any]]]:
    """
    Build a VecEnv from config dicts (already-loaded YAMLs).

    Args
    ----
    env_cfg: dict
        Parsed configs/env.yaml content.
    algo_env_cfg: dict | None
        Parsed configs/sb3_sac.yaml['env'] block. Used for max_episode_steps / n_envs fallback.
    log_root: Path
        Directory for per-env Monitor logs (monitor.csv).
    is_eval: bool
        If True, build a single DummyVecEnv with training=False for VecNormalize.

    Returns
    -------
    (venv, vecnorm_cfg)
        venv: The constructed (VecNormalize-wrapped) VecEnv.
        vecnorm_cfg: The VecNormalize config dict if applied, else None.
    """
    env_module = env_cfg.get("env_module", "simulator.environment")
    env_class = env_cfg.get("env_class", "AdvancedSmartACSimulator")
    env_kwargs = dict(env_cfg.get("env_kwargs", {}))  # shallow copy

    # Optional knobs the simulator may expect
    if "reset" in env_cfg:
        env_kwargs.setdefault("reset_randomizer", env_cfg["reset"])
    if "logging" in env_cfg:
        env_kwargs.setdefault("logging_cfg", env_cfg["logging"])

    seed = int(env_cfg.get("seed", 42))
    # Prefer algo's episode length (clean separation of concerns)
    max_episode_steps = int((algo_env_cfg or {}).get("max_episode_steps", 720))
    # Train uses N envs (defaults to env.yaml's num_envs or algo env.n_envs); eval always 1
    n_envs = 1 if is_eval else int(env_cfg.get("num_envs", (algo_env_cfg or {}).get("n_envs", 1)))

    # Monitor logs
    tag = "eval_env" if is_eval else "train_env"
    monitor_dir = Path(log_root) / tag
    _ensure_dir(monitor_dir)

    # Construct the vector env
    ctor = _make_single_env_ctor(
        env_module=env_module,
        env_class=env_class,
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        monitor_dir=monitor_dir,
        seed=seed,
    )
    if n_envs > 1 and not is_eval:
        venv: VecEnv = SubprocVecEnv([ctor for _ in range(n_envs)])
    else:
        venv = DummyVecEnv([ctor])

    # Optional VecNormalize
    vecnorm_cfg = _extract_vecnorm_cfg(env_cfg)
    if vecnorm_cfg:
        venv = VecNormalize(venv, training=not is_eval, **vecnorm_cfg)

    return venv, vecnorm_cfg


def sync_vecnormalize_stats(src_env: VecEnv, dst_env: VecEnv) -> None:
    """
    Copy VecNormalize running stats from src (train) to dst (eval).
    Safe no-op if either side isn't VecNormalize.
    """
    if isinstance(src_env, VecNormalize) and isinstance(dst_env, VecNormalize):
        dst_env.obs_rms = src_env.obs_rms
        dst_env.ret_rms = src_env.ret_rms
        dst_env.clip_obs = src_env.clip_obs
        dst_env.clip_reward = src_env.clip_reward
        dst_env.gamma = src_env.gamma


def save_vecnormalize_if_any(env: VecEnv, path: Path) -> None:
    """
    Save VecNormalize stats to `path` if env is wrapped by it.
    """
    if isinstance(env, VecNormalize):
        _ensure_dir(path.parent)
        env.save(str(path))


def is_vecnormalize(env: VecEnv) -> bool:
    return isinstance(env, VecNormalize)


__all__ = [
    "load_yaml",
    "build_vec_env",
    "sync_vecnormalize_stats",
    "save_vecnormalize_if_any",
    "is_vecnormalize",
]
