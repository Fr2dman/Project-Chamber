# sb3/train_sac_sb3.py
import os
import argparse
import yaml
import numpy as np
from torch import nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from rl.sb3.make_env import make_env_fn
from rl.sb3.callbacks import build_callbacks
import configs.hvac_config as hvac_config

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
    parser.add_argument("--config", type=str, default="configs/sb3_sac.yaml")
    parser.add_argument("--total_timesteps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.total_timesteps is not None:
        cfg["train"]["total_timesteps"] = args.total_timesteps

    set_random_seed(cfg["env"]["seed"])

    # 환경 파라미터: 필요 시 configs/hvac_config.py에서 dict로 빼와 연결하세요.
    # env_kwargs = dict()  # HVACEnv가 받는 인자에 맞게 필요하면 채우세요.
    # === 환경 파라미터 로딩 (configs/env.yaml) ===
    # env.yaml은 "시뮬레이터 인자"만 넣어주세요. (예: num_zones, 초기조건 등)
    # 상위 설정(seed/n_envs/episode_steps)은 sb3_sac.yaml에서 관리합니다.
    try:
        # 기본 경로: configs/env.yaml
        env_yaml_path = "configs/env.yaml"
        if os.path.isfile(env_yaml_path):
            with open(env_yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            env_kwargs = (y.get("env") or y) if isinstance(y, dict) else {}
            # episode.max_steps가 있으면 cfg의 max_episode_steps를 덮어씀
            ep = y.get("episode") or {}
            if isinstance(ep, dict) and "max_steps" in ep:
                cfg["env"]["max_episode_steps"] = int(ep["max_steps"])
    except Exception as e:
        print(f"[WARN] env.yaml load failed: {e}. Using default env_kwargs={{}}")

    # === 벡터 환경 / 평가 환경 ===
    train_env = build_vec_env(cfg, env_kwargs, for_eval=False)
    eval_env  = build_vec_env(cfg, env_kwargs, for_eval=True, obs_rms_source=train_env if isinstance(train_env, VecNormalize) else None)


    policy_kwargs = dict(
    net_arch=cfg["policy"]["net_arch"],
    activation_fn=nn.ReLU,
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
        tensorboard_log=cfg["train"]["tensorboard_log"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg["env"]["seed"]
    )

    callbacks = build_callbacks(eval_env, cfg)

    # (옵션) act_delta 가중치 점진 인상: 시작→종료를 지정하면 선형 보간
    # 예: --config 에서 train.act_delta_schedule: {start: 0.02, end: 0.05, steps: 200000}
    sch = cfg.get("train", {}).get("act_delta_schedule")
    if sch and isinstance(sch, dict):
        start = float(sch.get("start", hvac_config.RW.get("act_delta", 0.02)))
        end   = float(sch.get("end",   start))
        steps = int(sch.get("steps",   200000))
        if end != start and steps > 0:
            from stable_baselines3.common.callbacks import CallbackList
            from rl.sb3.callbacks import AnnealRWCallback
            anneal_cb = AnnealRWCallback("act_delta", start, end, steps)
            # 기존 콜백 리스트에 추가
            if isinstance(callbacks, CallbackList):
                callbacks.callbacks.insert(0, anneal_cb)
            else:
                callbacks = CallbackList([anneal_cb, callbacks])

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
