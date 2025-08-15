# scripts/rollout_to_gif.py
# This script collects a rollout from a trained SAC model in a custom HVAC environment
# and saves it as a GIF, visualizing the temperature readings and other metrics.
'''
executed with: python scripts/rollout_to_gif.py --config configs/sb3_sac.yaml --model checkpoints/sb3_sac/250814_model.zip 
--vecnorm checkpoints/sb3_sac/vecnorm.pkl --steps 180 --out checkpoints/sb3_sac/rollout.gif --fps 8
'''
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio

# 프로젝트 루트를 sys.path에 추가하여 'rl' 모듈을 찾을 수 있도록 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

# 우리 프로젝트의 Gym 래퍼
from rl.sb3.wrappers import HVACEnv
from configs.hvac_config import P_REF, CONTROL_TERM

def build_eval_env(max_steps: int, env_kwargs: dict, vecnorm_path: str | None):
    """정규화 통계를 로드(있으면)해서 평가용 단일 환경을 만듭니다."""
    def _thunk():
        return TimeLimit(HVACEnv(**env_kwargs), max_episode_steps=max_steps)
    base_env = DummyVecEnv([_thunk])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False
        return env
    return base_env

def grid_shape(n: int) -> tuple[int, int]:
    """존 개수에 맞는 격자 크기(행,열)를 찾아줍니다. (4면 2x2 등)"""
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and n % r != 0:
        r -= 1
    return r, n // r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sb3_sac.yaml")
    ap.add_argument("--model",  default="checkpoints/sb3_sac/250814_model.zip")
    ap.add_argument("--vecnorm", default="checkpoints/sb3_sac/vecnorm.pkl")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out",   default="checkpoints/sb3_sac/rollout.gif")
    ap.add_argument("--fps",   type=int, default=8)
    args = ap.parse_args()

    # 1) 설정 로드cd
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    max_steps = int(cfg["env"].get("max_episode_steps", 720))
    env_kwargs = {}  # 필요 시 configs/env.yaml, hvac_config.py에서 꺼내 사용

    # 2) 환경/모델 로드
    env = build_eval_env(max_steps, env_kwargs, args.vecnorm)
    model = SAC.load(args.model, env=env, device="cpu")

    # 3) 롤아웃 수집
    obs = env.reset()
    frames, energy_hist, budget_hist = [], [], []    
    vmin, vmax = 18, 30 # 온도 시각화 범위를 18~30도로 고정

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]  # DummyVecEnv라 0번만 사용

        sr = info.get("sensor_readings", {})
        temps = np.array(sr.get("temperatures", []), dtype=float)

        # 텍스트용 보조 정보
        hw = info.get("hardware_states", {})
        E_total = hw.get("total_energy_Wh", 0.0)
        energy_hist.append(E_total)

        rb = info.get("reward_breakdown", {})
        T_eff = np.array(rb.get("T_eff", []), dtype=float)
        avgT = float(np.mean(temps)) if temps.size else None
        E_budget = rb.get("E_budget_Wh")
        budget_hist.append(E_budget)

        frames.append(dict(temps=temps, T_eff=T_eff, E_total=E_total, E_budget=E_budget, avgT=avgT, step=t))

        if bool(dones[0]): # VecEnv는 dones가 배열
            break

    if not frames:
        raise RuntimeError("No frames collected. Check env/model paths.")

    # 4) GIF 생성
    n_z = len(frames[0]["temps"])
    R, C = grid_shape(n_z)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # 전체 누적 에너지의 최대값과 예산의 최대값을 구해 y축 범위 고정
    max_y_val = 0
    if energy_hist:
        max_y_val = max(max_y_val, max(energy_hist))
    if any(b is not None for b in budget_hist):
        max_y_val = max(max_y_val, max(b for b in budget_hist if b is not None))
    
    max_energy_y_axis = max_y_val * 1.15 if max_y_val > 0 else 1.0

    imgs = []
    for i_fr, fr in enumerate(frames):
        temps = fr["temps"]
        t_eff = fr["T_eff"]
        data_temps = temps.reshape(R, C)
        data_eff = t_eff.reshape(R, C) if t_eff.size == temps.size else np.zeros_like(data_temps)

        # Figure와 Subplot 레이아웃 설정 (위: 그리드, 아래: 그래프)
        fig = plt.figure(figsize=(3.5 * C, 3.5 * R + 2.2), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[R, 1.2], hspace=0.5)
        ax_grid = fig.add_subplot(gs[0])
        ax_graph = fig.add_subplot(gs[1])

        # --- 상단: 온도 그리드 ---
        im = ax_grid.imshow(data_temps, vmin=vmin, vmax=vmax, cmap="coolwarm")
        for i in range(R):
            for j in range(C):
                # 배경색에 따라 글자색을 바꿔 가독성 확보
                val_norm = (data_temps[i, j] - vmin) / (vmax - vmin + 1e-6)
                text_color = "w" if (val_norm < 0.2 or val_norm > 0.8) else "k"
                ax_grid.text(j, i, f"{data_temps[i,j]:.1f}°C\n(T*:{data_eff[i,j]:.1f})",
                             ha="center", va="center", color=text_color, fontsize=11, weight='bold')

        cb = plt.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.04)
        cb.set_label("Zone Temperature (°C)")
        ax_grid.set_xticks([]); ax_grid.set_yticks([])

        title_extra = []
        if fr["avgT"] is not None: title_extra.append(f"Avg Temp: {fr['avgT']:.1f}°C")
        if fr.get("E_total") is not None: title_extra.append(f"Cum. Energy: {fr['E_total']:.1f} Wh")
        ax_grid.set_title(f"Step {fr['step']}  |  " + "  |  ".join(title_extra), fontsize=14)

        # --- 하단: 누적 에너지 사용량 그래프 ---
        ax_graph.plot(range(i_fr + 1), energy_hist[:i_fr + 1], color='darkorange', linewidth=2, label="Cumulative Energy")
        # 예산(Budget)이 있는 경우 기준선으로 표시
        current_budget = fr.get("E_budget")
        if current_budget is not None:
            ax_graph.axhline(y=current_budget, color='green', linestyle='--', linewidth=1.5, label=f'Budget ({current_budget:.1f} Wh)')
        ax_graph.set_xlim(0, args.steps)
        ax_graph.set_ylim(0, max_energy_y_axis)
        ax_graph.set_xlabel("Time Step")
        ax_graph.set_ylabel("Cumulative Energy (Wh)")
        ax_graph.grid(True, linestyle=':', alpha=0.7)
        ax_graph.legend(loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 여백 조정

        # --- 이미지로 변환 및 저장 ---
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h, w, 4)[..., :3]
        imgs.append(img)
        plt.close(fig)

    imageio.mimsave(args.out, imgs, duration=1000/args.fps, loop=0)
    print(f"[OK] Saved GIF to {args.out}  ({len(imgs)} frames, range {vmin:.1f}~{vmax:.1f}°C)")

if __name__ == "__main__":
    main()
