# reward_sanity_check.py
import numpy as np
import sys, os

# 로컬 패키지 경로 세팅 (필요시 수정)
sys.path.append(os.path.dirname(__file__) or ".")

from simulator.environment import AdvancedSmartACSimulator  # 경로가 다르면 수정

def run_episode(sim, action, tsv, steps=25):
    sim.reset()
    # TSV 주입
    if hasattr(sim, "current_tsv"):
        sim.current_tsv = np.asarray(tsv, dtype=float)
    rewards, rdir, rtrack, temps_hist, power = [], [], [], [], []
    for _ in range(steps):
        obs, reward, done, info = sim.step(action)
        rb = info.get("reward_breakdown", {})
        rewards.append(float(reward))
        rdir.append(float(rb.get("R_dir", 0.0)))
        rtrack.append(float(rb.get("R_track", 0.0)))
        temps_hist.append(np.array(info["sensor_readings"]["temperatures"], dtype=float))
        power.append(float(info["hardware_states"]["total_power"]))
        if done: break
    return {
        "R": np.array(rewards),
        "R_dir": np.array(rdir),
        "R_track": np.array(rtrack),
        "T": np.array(temps_hist),          # [t, zone]
        "P": np.array(power)
    }

def sign(x): return (x>0) ^ (x<0)

def check_directionality(tsv_val, temps_hist, rdir):
    # ΔT 평균 부호와 R_dir의 부호가 일치해야 함 (TSV 부호 고려)
    dT = temps_hist[1:] - temps_hist[:-1]
    mean_dT = np.mean(dT)
    # TSV>0이면 ΔT<0가 좋아야 하므로 -ΔT가 좋아야 함 → 기대 부호 = sign(-mean_dT)
    expected = sign(-mean_dT) if tsv_val>0 else sign(mean_dT)
    got = sign(np.mean(rdir[1:]))  # 초기 1~2스텝 전이 제거
    return expected == got, dict(mean_dT=float(mean_dT), rdir_mean=float(np.mean(rdir[1:])))

def main():
    sim = AdvancedSmartACSimulator()

    # 액션
    a_idle = np.zeros(14, dtype=np.float32)
    a_cool = np.array([1.0, 1.0,1.0,1.0,1.0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0, 1.0], dtype=np.float32)

    # S1: hot_all
    tsv_hot = np.full(sim.num_zones, +2.0)
    res_idle = run_episode(sim, a_idle, tsv_hot)
    res_cool = run_episode(sim, a_cool, tsv_hot)

    ok_dir_hot, diag_dir_hot = check_directionality(+2.0, res_cool["T"], res_cool["R_dir"])
    print("[S1] TSV=+2 방향성:", "PASS" if ok_dir_hot else "FAIL", diag_dir_hot)

    # R_track 개선 여부(마지막 5스텝 평균 비교)
    tr_idle  = np.mean(res_idle["R_track"][-5:])
    tr_cool  = np.mean(res_cool["R_track"][-5:])
    print("[S1] 추적성(R_track) idle vs cool:", tr_idle, "->", tr_cool)

    # S2: cold_all
    tsv_cold = np.full(sim.num_zones, -2.0)
    res_cold = run_episode(sim, a_cool, tsv_cold)
    ok_dir_cold, diag_dir_cold = check_directionality(-2.0, res_cold["T"], res_cold["R_dir"])
    print("[S2] TSV=-2 방향성(과냉각 억제):", "PASS" if ok_dir_cold else "FAIL", diag_dir_cold)

    # S3: 에너지 일관성(동일 TSV·비슷한 온도대에서 전력↑ → 보상↓)
    #   간단 비교: 강냉각(a_cool) vs 미냉각(a_idle)에서 평균 전력과 R_energy 관련 총보상 추세 확인
    meanP_idle = float(np.mean(res_idle["P"][-5:]))
    meanP_cool = float(np.mean(res_cool["P"][-5:]))
    meanR_idle = float(np.mean(res_idle["R"][-5:]))
    meanR_cool = float(np.mean(res_cool["R"][-5:]))
    print("[S3] 에너지 일관성: P_idle=%.1f, P_cool=%.1f, R_idle=%.3f, R_cool=%.3f"
          % (meanP_idle, meanP_cool, meanR_idle, meanR_cool))
    # 기대: P_cool > P_idle. (총보상은 상태 따라 다르지만) 동일 상태면 전력↑이면 더 불리해야 함.

    # 종합 요약
    print("\n=== 요약 ===")
    print("S1 방향성:", "PASS" if ok_dir_hot else "FAIL")
    print("S1 R_track 개선:", "PASS" if tr_cool >= tr_idle else "FAIL")
    print("S2 방향성(과냉각 억제):", "PASS" if ok_dir_cold else "FAIL")

if __name__ == "__main__":
    main()
