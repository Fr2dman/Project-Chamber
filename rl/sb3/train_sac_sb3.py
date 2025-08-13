# rl/sb3/train_sac_sb3.py
"""
SAC training entrypoint for the Smart-AC simulator (SB3).

- Loads configs from configs/env.yaml and configs/sb3_sac.yaml (and reward.yaml is consumed by env/hvac_config internally)
- Builds vectorized training and evaluation envs
- Applies VecNormalize if configured (from env.yaml 'wrappers' list)
- Sets up Eval/Checkpoint callbacks and syncs VecNormalize stats to the eval env
- Saves best model and final artifacts into checkpoints/sb3_sac

Usage:
    python rl/sb3/train_sac_sb3.py
    python rl/sb3/train_sac_sb3.py --env-config ./configs/env.yaml --algo-config ./configs/sb3_sac.yaml --run-name exp01

Requirements:
    - stable-baselines3>=2.3
    - PyYAML
"""

from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


# -------------------------------
# Utilities
# -------------------------------
def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk upwards until a 'configs' dir is found."""
    cur = Path(start or __file__).resolve().parent
    for _ in range(6):
        if (cur / "configs").is_dir():
            return cur
        cur = cur.parent
    # Fallback: assume two levels up from this file (rl/sb3 → project root)
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def import_env_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def extract_vecnorm_cfg(env_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    env.yaml supports:
    wrappers:
      - VecNormalize:
          norm_obs: true
          norm_reward: false
          clip_obs: 10.0
          gamma: 0.995
    This function returns that dict if present, else None.
    """
    wrappers = env_cfg.get("wrappers", [])
    for item in wrappers:
        if isinstance(item, dict) and "VecNormalize" in item:
            cfg = item["VecNormalize"] or {}
            # enforce defaults if missing
            return {
                "norm_obs": bool(cfg.get("norm_obs", True)),
                "norm_reward": bool(cfg.get("norm_reward", False)),
                "clip_obs": float(cfg.get("clip_obs", 10.0)),
                "gamma": float(cfg.get("gamma", 0.995)),
            }
    return None


def make_single_env_ctor(
    env_module: str,
    env_class: str,
    env_kwargs: Dict[str, Any],
    max_episode_steps: int,
    monitor_dir: Path,
    seed: int,
) -> Callable[[], Any]:
    """
    Returns a thunk that creates a single monitored/time-limited env instance.
    """
    def _thunk():
        EnvCls = import_env_class(env_module, env_class)
        env = EnvCls(**(env_kwargs or {}))
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env, filename=str(monitor_dir / "monitor.csv"), allow_early_resets=True)
        env.reset(seed=seed)
        return env
    return _thunk


def build_vec_env(
    env_cfg: Dict[str, Any],
    algo_env_cfg: Dict[str, Any],
    log_root: Path,
    is_eval: bool = False,
) -> Tuple[VecEnv, Optional[Dict[str, Any]]]:
    """
    Build a VecEnv according to env.yaml and sb3_sac.yaml's env.* block.
    Returns (env, vecnorm_cfg_if_any)
    """
    env_module = env_cfg.get("env_module", "simulator.environment")
    env_class = env_cfg.get("env_class", "AdvancedSmartACSimulator")
    env_kwargs = dict(env_cfg.get("env_kwargs", {}))  # copy

    # Merge additional knobs that env may expect
    if "reset" in env_cfg:
        env_kwargs.setdefault("reset_randomizer", env_cfg["reset"])
    if "logging" in env_cfg:
        env_kwargs.setdefault("logging_cfg", env_cfg["logging"])

    seed = int(env_cfg.get("seed", 42))
    # Prefer algo config for max steps (clean separation)
    max_episode_steps = int(algo_env_cfg.get("max_episode_steps", 720))
    n_envs = int(env_cfg.get("num_envs", algo_env_cfg.get("n_envs", 1))) if not is_eval else 1

    monitor_dir = log_root / ("eval_env" if is_eval else "train_env")
    ensure_dir(monitor_dir)

    ctor = make_single_env_ctor(
        env_module=env_module,
        env_class=env_class,
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        monitor_dir=monitor_dir,
        seed=seed,
    )

    if n_envs > 1 and not is_eval:
        venv = SubprocVecEnv([ctor for _ in range(n_envs)])
    else:
        venv = DummyVecEnv([ctor])

    vecnorm_cfg = extract_vecnorm_cfg(env_cfg)  # may be None
    if vecnorm_cfg:
        # training=True for train env; False for eval env
        venv = VecNormalize(venv, training=not is_eval, **vecnorm_cfg)

    return venv, vecnorm_cfg


def sync_vecnormalize_stats(src_env: VecEnv, dst_env: VecEnv) -> None:
    """
    Copy running means/vars from src (train) to dst (eval) if both are VecNormalize.
    """
    if isinstance(src_env, VecNormalize) and isinstance(dst_env, VecNormalize):
        dst_env.obs_rms = src_env.obs_rms
        dst_env.ret_rms = src_env.ret_rms
        dst_env.clip_obs = src_env.clip_obs
        dst_env.clip_reward = src_env.clip_reward
        dst_env.gamma = src_env.gamma


class VecNormSyncCallback(BaseCallback):
    """
    Keeps eval VecNormalize synced with the train env before each evaluation.
    Useful when EvalCallback uses a separate eval env.
    """
    def __init__(self, train_env: VecEnv, eval_env: VecEnv, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        return True  # nothing per step

    def _on_rollout_end(self) -> None:
        # Sync periodically (after each rollout); EvalCallback will also run soon.
        sync_vecnormalize_stats(self.train_env, self.eval_env)


# -------------------------------
# Main training procedure
# -------------------------------
def main():
    project_root = find_project_root()
    default_env_cfg = project_root / "configs" / "env.yaml"
    default_algo_cfg = project_root / "configs" / "sb3_sac.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=Path, default=default_env_cfg)
    parser.add_argument("--algo-config", type=Path, default=default_algo_cfg)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    # Load configs
    env_cfg = load_yaml(args.env_config)
    sac_cfg = load_yaml(args.algo_config)

    # Resolve logging/checkpoint dirs
    checkpoints_dir = project_root / "checkpoints" / "sb3_sac"
    logs_dir = project_root / "logs" / "sb3_sac"
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    if args.run_name:
        # sub-run separation: checkpoints/sb3_sac/<run_name> ; logs/sb3_sac/<run_name>
        checkpoints_dir = checkpoints_dir / args.run_name
        logs_dir = logs_dir / args.run_name
        ensure_dir(checkpoints_dir)
        ensure_dir(logs_dir)

    # Seeds
    seed = int(env_cfg.get("seed", 42))
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Build train/eval envs
    algo_env_cfg = sac_cfg.get("env", {})
    train_env, vecnorm_cfg = build_vec_env(env_cfg, algo_env_cfg, logs_dir, is_eval=False)
    eval_env, _ = build_vec_env(env_cfg, algo_env_cfg, logs_dir, is_eval=True)

    # If VecNormalize was used for train/eval, keep them in sync
    sync_vecnormalize_stats(train_env, eval_env)

    # Algorithm hyperparams
    hp = sac_cfg.get("hyperparams", {})
    policy = sac_cfg.get("policy", "MlpPolicy")
    total_timesteps = int(sac_cfg.get("total_timesteps", 1_000_000))

    # Important flags from config (already recommended in your setup)
    hp.setdefault("handle_timeout_termination", True)
    hp.setdefault("optimize_memory_usage", True)

    # Instantiate model
    model = SAC(
        policy=policy,
        env=train_env,
        tensorboard_log=str(logs_dir),
        seed=seed,
        **hp,
    )

    # Callbacks
    eval_cfg = sac_cfg.get("eval", {})
    eval_freq = int(eval_cfg.get("eval_freq", 10_000))
    n_eval_episodes = int(eval_cfg.get("n_eval_episodes", 5))
    deterministic = bool(eval_cfg.get("deterministic", True))

    # Paths
    best_model_dir = Path(eval_cfg.get("best_model_save_path", str(checkpoints_dir)))
    eval_log_path = Path(eval_cfg.get("log_path", str(logs_dir)))
    ensure_dir(best_model_dir)
    ensure_dir(eval_log_path)

    # Checkpoint callback (periodic)
    ckpt_cb = CheckpointCallback(
        save_freq=max(eval_freq, 10_000),
        save_path=str(checkpoints_dir),
        name_prefix="sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # VecNormalize sync helper
    vns_cb = VecNormSyncCallback(train_env=train_env, eval_env=eval_env, verbose=0)

    # Eval callback (uses eval env; saves best model)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_path),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
    )

    # Optional progress bar (no-op if SB3 version lacks it)
    pb_cb = ProgressBarCallback()

    callbacks = CallbackList([vns_cb, eval_cb, ckpt_cb, pb_cb])

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # Save final artifacts
    model.save(str(checkpoints_dir / "final_model"))
    if isinstance(train_env, VecNormalize):
        # Save VecNormalize stats (compatible with SB3 load later)
        train_env.save(str(checkpoints_dir / "vecnorm.pkl"))
    # Also save replay buffer for potential offline fine-tuning
    try:
        model.save_replay_buffer(str(checkpoints_dir / "replay_buffer.pkl"))
    except Exception:
        pass

    # Clean up
    train_env.close()
    eval_env.close()

    print(f"[Done] Trained SAC for {total_timesteps} steps.")
    print(f" - Best/periodic checkpoints: {best_model_dir}")
    print(f" - Final model: {checkpoints_dir / 'final_model.zip'}")
    print(f" - VecNormalize stats: {checkpoints_dir / 'vecnorm.pkl'}")


if __name__ == "__main__":
    main()
