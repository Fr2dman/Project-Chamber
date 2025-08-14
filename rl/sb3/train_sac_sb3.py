# sb3/train_sac_sb3.py
import os
import argparse
import yaml
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from rl.sb3.make_env import make_env_fn
from rl.sb3.callbacks import build_callbacks

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_vec_env(cfg, env_kwargs, for_eval=False, obs_rms_source: VecNormalize | None = None):
    n_envs = 1 if for_eval else cfg["env"]["n_envs"]
    seeds = [cfg["env"]["seed"] + i for i in range(n_envs)]
    thunks = [make_env_fn(env_kwargs, cfg["env"]["max_episode_steps"], seeds[i]) for i in range(n_envs)]

    # SubprocVecEnv가 pickling 문제 있으면 DummyVecEnv로 바꾸세요.
    venv = SubprocVecEnv(thunks) if n_envs > 1 and not for_eval else DummyVecEnv(thunks)
    venv = VecMonitor(venv)

    if cfg["env"]["vec_norm_obs"] or cfg["env"]["vec_norm_reward"]:
        venv = VecNormalize(
            venv,
            training=not for_eval,
            norm_obs=cfg["env"]["vec_norm_obs"],
            norm_reward=cfg["env"]["vec_norm_reward"],
            clip_obs=cfg["env"]["clip_obs"]
        )
        if for_eval and obs_rms_source is not None:
            # 학습 환경의 통계로 평가환경 정규화 맞춤
            venv.obs_rms = obs_rms_source.obs_rms
            venv.ret_rms = obs_rms_source.ret_rms
    return venv

def main():
    parser = argparse.ArgumentParser()
    # --run-name 인자를 추가하여 TensorBoard 로그를 구분할 수 있도록 합니다.
    parser.add_argument("--run-name", type=str, default="sac_run", help="Name for the training run, used for logging.")
    parser.add_argument("--config", type=str, default="configs/sb3_sac.yaml", help="Path to the algorithm configuration file.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps from the config file.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    # 명령줄 인자로 total_timesteps를 덮어쓸 수 있게 합니다.
    if args.total_timesteps is not None:
        cfg["train"]["total_timesteps"] = args.total_timesteps

    set_random_seed(cfg["env"]["seed"])

    # 환경 파라미터: 필요 시 configs/hvac_config.py에서 dict로 빼와 연결하세요.
    env_kwargs = dict()  # HVACEnv가 받는 인자에 맞게 필요하면 채우세요.

    # === 벡터 환경 / 평가 환경 ===
    train_env = build_vec_env(cfg, env_kwargs, for_eval=False)
    eval_env  = build_vec_env(cfg, env_kwargs, for_eval=True, obs_rms_source=train_env if isinstance(train_env, VecNormalize) else None)


    policy_kwargs = dict(
        net_arch=cfg["policy"]["net_arch"],
        activation_fn=getattr(torch.nn, cfg["policy"].get("activation_fn", "ReLU")),
        log_std_init=cfg["policy"]["log_std_init"]
    )

    lr = float(cfg["sac"]["learning_rate"])   # ← 안전하게 캐스팅
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        buffer_size=cfg["sac"]["buffer_size"],
        batch_size=cfg["sac"]["batch_size"],
        tau=cfg["sac"]["tau"],
        gamma=cfg["sac"]["gamma"],
        train_freq=cfg["sac"]["train_freq"],
        gradient_steps=cfg["sac"]["gradient_steps"],
        learning_starts=cfg["sac"]["learning_starts"],
        ent_coef=cfg["sac"]["ent_coef"],
        target_entropy=cfg["sac"]["target_entropy"],
        # --run-name 인자를 사용하여 로그 경로를 동적으로 설정합니다.
        tensorboard_log=os.path.join(cfg["train"]["tensorboard_log"], args.run_name),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg["env"]["seed"]
    )

    callbacks = build_callbacks(eval_env, cfg)

    # === 학습 ===
    model.learn(
        total_timesteps=cfg["train"]["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )

    # === 저장 ===
    ckpt_dir = cfg["train"]["checkpoint_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "final_model.zip"))

    # VecNormalize 통계 저장
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(ckpt_dir, "vecnorm.pkl"))

    print("Training done. Best/Final models saved to:", ckpt_dir)

if __name__ == "__main__":
    main()
