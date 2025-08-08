# test/test_tsv_rollout.py
from simulator.environment import AdvancedSmartACSimulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 0) 설정 =====
NUM_STEPS = 30

# TSV 시나리오: 'hot_all' | 'cold_all' | 'oscillate' | 'zone1_hot' | 'random'
TSV_SCENARIO = 'hot_all'
TSV_MAG = 2.0  # TSV 절대값 (0~3)
TSV_CONFIDENCE = 1.0  # (선택) tsv_confidence가 있다면 0~1

# 액션 시나리오: 'max_cool' | 'gentle' | 'idle'
ACTION_SCENARIO = 'max_cool'

def make_action(scenario: str) -> np.ndarray:
    """
    액션 벡터 생성 ([-1, 1] 범위, 14차원 가정)
    [0] peltier, [1:5] internal servos, [5:9] external servos, [9:13] small fans, [13] large fan
    """
    if scenario == 'max_cool':
        return np.array([
            1.0,              # 펠티어: 최대 냉각
            1.0, 1.0, 1.0, 1.0,  # 내부 서보: 모두 크게 열기
            0.5, 0.5, 0.5, 0.5,  # 외부 서보: 중간 각도
            1.0, 1.0, 1.0, 1.0,  # 소형 팬: 모두 최대
            1.0               # 대형 팬: 최대
        ], dtype=np.float32)
    elif scenario == 'gentle':
        return np.array([
            0.3,
            0.2, 0.2, 0.2, 0.2,
            0.0, 0.0, 0.0, 0.0,
            0.3, 0.3, 0.3, 0.3,
            0.3
        ], dtype=np.float32)
    elif scenario == 'idle':
        return np.zeros(14, dtype=np.float32)
    else:
        raise ValueError("Unknown action scenario")

def tsv_pattern(step: int, n_zones: int, scenario: str, mag: float) -> np.ndarray:
    """
    매 스텝 주입할 TSV 리스트 생성. 범위: [-3, +3]
    """
    if scenario == 'hot_all':
        return np.full(n_zones, +mag)
    if scenario == 'cold_all':
        return np.full(n_zones, -mag)
    if scenario == 'oscillate':
        # 홀수 스텝: 덥다(+), 짝수 스텝: 춥다(-)
        sign = +1.0 if (step % 2 == 1) else -1.0
        return np.full(n_zones, sign * mag)
    if scenario == 'zone1_hot':
        tsv = np.zeros(n_zones)
        tsv[0] = +mag
        return tsv
    if scenario == 'random':
        return np.random.uniform(-mag, +mag, size=n_zones)
    raise ValueError("Unknown TSV scenario")

def maybe_update_tsv(sim: AdvancedSmartACSimulator, tsv_values: np.ndarray):
    """
    환경에 TSV를 주입. update_tsv가 있으면 사용, 없으면 current_tsv 속성에 직접 반영.
    (선택) tsv_confidence 지원 시 함께 주입.
    """
    if hasattr(sim, "update_tsv") and callable(getattr(sim, "update_tsv")):
        sim.update_tsv(tsv_values.tolist())
    else:
        # fallback: 속성 직접 세팅
        if hasattr(sim, "current_tsv"):
            sim.current_tsv = np.asarray(tsv_values, dtype=float)
        else:
            raise AttributeError("Simulator has no TSV setter nor 'current_tsv' attribute.")

    # 선택: tsv_confidence 지원 시
    if hasattr(sim, "update_tsv_confidence") and callable(getattr(sim, "update_tsv_confidence")):
        sim.update_tsv_confidence([TSV_CONFIDENCE] * len(tsv_values))
    elif hasattr(sim, "tsv_confidence"):
        sim.tsv_confidence = np.full(len(tsv_values), TSV_CONFIDENCE, dtype=float)

def print_initial(sim: AdvancedSmartACSimulator):
    st = sim._get_current_state()
    print("--- 초기 상태 ---")
    try:
        print("내부 서보 각도:",
              [f"{s.current_angle:.1f}°" for s in sim.internal_servos])
        print("외부 서보 각도:",
              [f"{s.current_angle:.1f}°" for s in sim.external_servos])
        print("소형 팬 RPM:",
              [f"{f.current_rpm:.0f}" for f in sim.small_fans])
        print("대형 팬 RPM:",
              f"{sim.large_fan.current_rpm:.0f}")
    except Exception:
        pass

    print("온도:", [f"{t:.1f}°C" for t in st.get('temperatures', [])])
    print("습도:", [f"{h:.1f}%" for h in st.get('humidities', [])])
    cs = st.get('comfort_scores', {})
    if isinstance(cs, dict):
        scores = cs.get('comfort_scores', [])
        avg = cs.get('average_comfort', float('nan'))
    else:
        scores = cs
        avg = np.mean(scores) if len(scores) else float('nan')
    print("쾌적도:", [f"{score:.2f}pt" for score in scores])
    print("쾌적도 평균:", f"{avg:.2f}")

def main():
    sim = AdvancedSmartACSimulator()
    np.random.seed(42)

    print_initial(sim)

    action = make_action(ACTION_SCENARIO)

    logs = []
    for step in range(1, NUM_STEPS + 1):
        # 1) TSV 주입
        tsv = tsv_pattern(step, sim.num_zones, TSV_SCENARIO, TSV_MAG)
        maybe_update_tsv(sim, tsv)

        # 2) step 실행
        print(f"\n--- {step}-Step ---")
        print("주입 TSV:", [f"{x:+.1f}" for x in tsv])
        print("입력 액션:", action.tolist())

        obs, reward, done, info = sim.step(action)

        # 3) 결과 꺼내기 (info 키 이름은 구현에 따라 다를 수 있어 방어적으로 처리)
        hw = info.get('hardware_states', info.get('hardware', {}))
        sr = info.get('sensor_readings', info.get('sensors', {}))
        cf = info.get('comfort_data', info.get('comfort', {}))
        rb = info.get('reward_breakdown', info.get('reward_detail', info.get('breakdown', {})))

        # 프린트
        try:
            print("펠티어 표면온도:",
                  f"{getattr(sim.peltier, 'cold_side_temp', float('nan')):.1f}°C")
        except Exception:
            pass

        try:
            print("내부 서보 각도:", [f"{a:.1f}°" for a in hw['servos']['internal']])
            print("외부 서보 각도:", [f"{a:.1f}°" for a in hw['servos']['external']])
            print("소형 팬 RPM:", [f"{fan['rpm']:.0f}" for fan in hw['fans']['small_fans']])
            print("대형 팬 RPM:", f"{hw['fans']['large_fan']['rpm']:.0f}")
            print(f"소비 전력: {hw.get('total_power', float('nan')):.2f} W")
        except Exception:
            pass

        temps = sr.get('temperatures', [])
        hums  = sr.get('humidities', [])
        print("온도:", [f"{t:.1f}°C" for t in temps])
        print("습도:", [f"{h:.1f}%" for h in hums])

        scores = cf['comfort_scores'] if isinstance(cf, dict) else cf
        avg_c  = cf.get('average_comfort', float('nan')) if isinstance(cf, dict) else (np.mean(scores) if len(scores) else float('nan'))
        print("쾌적도:", [f"{s:.2f}pt" for s in scores])
        print("쾌적도 평균:", f"{avg_c:.2f}")

        # 보상 및 브레이크다운
        r_track = rb.get('R_track', None)
        r_dir   = rb.get('R_dir', None)
        print(f"보상: {reward:.3f}"
              + (f" | R_track: {r_track:.4f}" if r_track is not None else "")
              + (f" | R_dir: {r_dir:.4f}" if r_dir is not None else ""))

        # 로깅
        logs.append({
            "step": step,
            "reward": reward,
            "R_track": r_track,
            "R_dir": r_dir,
            "avg_comfort": avg_c,
            "power": hw.get('total_power', np.nan) if isinstance(hw, dict) else np.nan,
            **{f"T{i}": t for i, t in enumerate(temps)},
            **{f"H{i}": h for i, h in enumerate(hums)},
            **{f"TSV{i}": v for i, v in enumerate(tsv)},
        })

        if done:
            print("에피소드 종료 신호 수신 (done=True).")
            break

    # ===== 결과 저장/시각화 (선택) =====
    df = pd.DataFrame(logs)
    df.to_csv("tsv_rollout_log.csv", index=False)
    print("\n로그 저장: tsv_rollout_log.csv")

    # 간단한 플롯 (환경에 따라 주석 처리 가능)
    try:
        plt.figure()
        plt.plot(df["step"], df["reward"], label="reward")
        if "R_track" in df:
            plt.plot(df["step"], df["R_track"], label="R_track")
        if "R_dir" in df:
            plt.plot(df["step"], df["R_dir"], label="R_dir")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title("Reward / R_track / R_dir")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("플롯 오류:", e)

if __name__ == "__main__":
    main()
