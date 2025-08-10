import argparse
from pathlib import Path
import numpy as np
import csv

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 우리 프로젝트 유틸: env 생성은 여기서
from rl.sb3.make_env import make_env  # 이미 있으므로 그대로 사용

def maybe_load_vecnorm(vec_path: Path, venv):
    if vec_path.exists():
        venv = VecNormalize.load(str(vec_path), venv)
        venv.training = False
        venv.norm_reward = False  # 원 스케일 보상으로 보고 싶을 때
        print(f"[VecNormalize] loaded: {vec_path}")
    else:
        print("[VecNormalize] vecnorm.pkl not found. Evaluating without normalization.")
    return venv

def rollout(model, venv, episodes=5, max_steps=720, deterministic=True, csv_path: Path | None = None):
    ep_summaries, step_rows = [], []

    for ep in range(1, episodes + 1):
        obs = venv.reset()
        ep_return, ep_len = 0.0, 0
        sum_power, sum_comfort = 0.0, 0.0
        comfort_cnt = 0
        safety_viol = humidity_viol = co2_viol = 0

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = venv.step(action)

            r = float(reward[0])
            i = info[0]
            ep_return += r
            ep_len += 1

            # 선택 정보들(환경이 제공할 때만)
            hw = i.get("hardware_states", {})
            comfort = i.get("comfort_data", {})
            sensors = i.get("sensor_readings", {})
            avg_comfort = comfort.get("average_comfort")

            if "step_power_consumption" in hw:
                sum_power += float(hw["step_power_consumption"])
            if avg_comfort is not None:
                sum_comfort += float(avg_comfort); comfort_cnt += 1

            safety_viol += int(i.get("safety_violation", 0))
            humidity_viol += int(i.get("humidity_violation", 0))
            co2_viol += int(i.get("co2_violation", 0))

            if csv_path is not None:
                row = {
                    "episode": ep, "t": t, "reward": r,
                    "avg_comfort": avg_comfort,
                    "step_power": hw.get("step_power_consumption"),
                    "T_mean": (np.mean(sensors.get("temperatures", []))
                               if "temperatures" in sensors else None),
                    "H_mean": (np.mean(sensors.get("humidities", []))
                               if "humidities" in sensors else None),
                    "CO2_mean": (np.mean(sensors.get("co2", []))
                                 if "co2" in sensors else None),
                }
                # 보상 항목별이 info에 있으면 같이 기록
                for k, v in (i.get("reward_breakdown", {}) or {}).items():
                    try: row[f"R_{k}"] = float(v)
                    except Exception: row[f"R_{k}"] = None
                step_rows.append(row)

            if done[0]:
                break

        mean_comfort = (sum_comfort / max(1, comfort_cnt))
        ep_summaries.append(dict(
            episode=ep, return_sum=ep_return, ep_len=ep_len,
            mean_comfort=mean_comfort, power_sum=sum_power,
            safety_violations=safety_viol, humidity_violations=humidity_viol, co2_violations=co2_viol,
        ))
        print(f"[Ep {ep:02d}] Return={ep_return:.2f} Steps={ep_len} "
              f"Comfort(avg)={mean_comfort:.2f} Power(sum)={sum_power:.2f} "
              f"Viol(s/h/co2)={safety_viol}/{humidity_viol}/{co2_viol}")

    # 저장
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # step 로그
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for row in step_rows for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in step_rows: w.writerow(row)
        print(f"[SAVE] steps -> {csv_path}")
        # 에피소드 요약
        ep_csv = csv_path.with_name(csv_path.stem + "_episodes.csv")
        with open(ep_csv, "w", newline="", encoding="utf-8") as f:
            fields = list(ep_summaries[0].keys())
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for m in ep_summaries: w.writerow(m)
        print(f"[SAVE] episodes -> {ep_csv}")

    # 콘솔 요약
    rets = np.array([m["return_sum"] for m in ep_summaries], float)
    print("\n=== SUMMARY ===")
    print(f"Episodes: {len(ep_summaries)}")
    print(f"Return  : mean={rets.mean():.2f} std={rets.std():.2f} "
          f"min={rets.min():.2f} max={rets.max():.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/sb3_sac/best_model.zip")
    p.add_argument("--vecnorm", default="checkpoints/sb3_sac/vecnorm.pkl")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--csv", default="logs/eval_rollout.csv")
    args = p.parse_args()

    # Env 생성 (우리 프로젝트의 make_env 사용)
    base_env = make_env(seed=42)  # rl/sb3/make_env.py 내부 정의 사용
    venv = DummyVecEnv([lambda: base_env])
    venv = maybe_load_vecnorm(Path(args.vecnorm), venv)

    print(f"[LOAD] {args.model}")
    model = SAC.load(args.model, device=args.device)

    rollout(
        model, venv,
        episodes=args.episodes, max_steps=args.max_steps,
        deterministic=args.deterministic,
        csv_path=Path(args.csv) if args.csv else None
    )

if __name__ == "__main__":
    main()
