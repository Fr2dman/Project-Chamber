# simple_one_step_test.py
from simulator.environment import AdvancedSmartACSimulator
import numpy as np
import pandas as pd

# ===== 설정 =====
# TSV 시나리오: 한 번의 step에서 주입할 TSV (존 수와 길이 동일)
TSV_VALUES = [2.0, -2.0, 2.0, 2.0]   # "춥다" 피드백 (cold_all)

# 액션 벡터(14차원, [-1, 1]): [peltier, 4x internal, 4x external, 4x small fans, large fan]
# 아래는 'max_cool' 예시
ACTION = np.array([
    0.0,            # Peltier: 최대 냉각
    1.0, 1.0, 1.0, 1.0,     # 내부 슬롯(각도) 크게 열기
    0.5, 0.5, 0.5, 0.5,     # 외부 슬롯(각도) 중간
    0.0, 0.0, 0.0, 0.0,     # 소형 팬 최대
    0.0                     # 대형 팬 최대
], dtype=np.float32)

def maybe_set_tsv(sim, tsv_list):
    """update_tsv 있으면 사용, 없으면 current_tsv 속성에 직접 반영."""
    if hasattr(sim, "update_tsv") and callable(getattr(sim, "update_tsv")):
        sim.update_tsv(tsv_list)
    elif hasattr(sim, "current_tsv"):
        sim.current_tsv = np.asarray(tsv_list, dtype=float)
    else:
        raise AttributeError("TSV를 설정하는 메서드/속성을 찾지 못했습니다.")

def get_pre_hw(sim):
    """step 전 하드웨어 상태(각도/RPM)를 읽어옵니다."""
    try:
        internal = [s.current_angle for s in sim.internal_servos]
        external = [s.current_angle for s in sim.external_servos]
        smallrpm = [f.current_rpm for f in sim.small_fans]
        largerpm = sim.large_fan.current_rpm
    except Exception:
        internal, external, smallrpm, largerpm = None, None, None, None
    return internal, external, smallrpm, largerpm

def prettify_angles(angles):
    return [f"{a:.1f}°" for a in angles] if angles is not None else None

def prettify_rpms(rpms):
    if rpms is None: return None
    return [f"{int(r)}" for r in rpms]

def main():
    sim = AdvancedSmartACSimulator()

    # ---- step 전 상태 출력 ----
    pre_state = sim._get_current_state()
    pre_internal, pre_external, pre_smallrpm, pre_largerpm = get_pre_hw(sim)

    print("=== [PRE] Step 이전 상태 ===")
    print("온도(°C):", [f"{t:.1f}" for t in pre_state.get("temperatures", [])])
    print("습도(%RH):", [f"{h:.1f}" for h in pre_state.get("humidities", [])])
    cs = pre_state.get("comfort_scores", {})
    c_scores = cs.get("comfort_scores", [])
    c_avg = cs.get("average_comfort", float('nan'))
    print("쾌적도 점수:", [f"{s:.2f}" for s in c_scores], "| 평균:", f"{c_avg:.2f}")
    if pre_internal is not None:
        print("내부 슬롯(각도):", prettify_angles(pre_internal))
        print("외부 슬롯(각도):", prettify_angles(pre_external))
        print("소형 팬 RPM:", prettify_rpms(pre_smallrpm))
        print("대형 팬 RPM:", int(pre_largerpm))

    # ---- TSV 주입 ----
    maybe_set_tsv(sim, TSV_VALUES)
    print("\n주입 TSV:", [f"{v:+.1f}" for v in TSV_VALUES])

    # ---- 액션 적용 & 1 step ----
    print("입력 액션(14D):", ACTION.tolist())
    obs, reward, done, info = sim.step(ACTION)

    # ---- step 후 상태/하드웨어/보상 ----
    hw = info.get("hardware_states", {})
    sr = info.get("sensor_readings", {})
    cf = info.get("comfort_data", {})
    rb = info.get("reward_breakdown", info.get("reward_detail", info.get("breakdown", {})))

    print("\n=== [POST] Step 이후 상태 ===")
    temps = np.array(sr.get("temperatures", []), dtype=float)
    hums  = np.array(sr.get("humidities", []), dtype=float)
    print("온도(°C):", [f"{t:.1f}" for t in temps])
    print("습도(%RH):", [f"{h:.1f}" for h in hums])

    c_scores2 = cf.get("comfort_scores", [])
    c_avg2 = cf.get("average_comfort", float('nan'))
    print("쾌적도 점수:", [f"{s:.2f}" for s in c_scores2], "| 평균:", f"{c_avg2:.2f}")

    try:
        print("펠티어 냉면 온도(°C):", f"{getattr(sim.peltier, 'cold_side_temp', float('nan')):.1f}")
    except Exception:
        pass

    try:
        print("내부 슬롯(각도):", [f"{a:.1f}°" for a in hw["servos"]["internal"]])
        print("외부 슬롯(각도):", [f"{a:.1f}°" for a in hw["servos"]["external"]])
        print("소형 팬 RPM:", [str(int(f["rpm"])) for f in hw["fans"]["small_fans"]])
        print("대형 팬 RPM:", int(hw["fans"]["large_fan"]["rpm"]))
        print("소비 전력(W):", f"{hw.get('total_power', float('nan')):.2f}")
    except Exception:
        pass

    # ---- 존별 변화 요약 테이블 ----
    pre_T = np.array(pre_state.get("temperatures", []), dtype=float)
    pre_H = np.array(pre_state.get("humidities", []), dtype=float)
    pre_C = np.array(c_scores, dtype=float) if c_scores else np.full_like(temps, np.nan)

    df = pd.DataFrame({
        "Zone": np.arange(len(temps)),
        "T_pre": pre_T,
        "T_post": temps,
        "ΔT": temps - pre_T,
        "RH_pre": pre_H,
        "RH_post": hums,
        "ΔRH": hums - pre_H,
        "Comfort_pre": pre_C,
        "Comfort_post": np.array(c_scores2, dtype=float) if len(c_scores2) else np.full_like(temps, np.nan),
        "TSV": np.array(TSV_VALUES, dtype=float)[:len(temps)],
    })
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print("\n--- 존별 요약(1 step) ---")
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:,.2f}"))

    # ---- 보상 브레이크다운 ----
    print("\n=== 보상(Reward) ===")
    print(f"total: {reward:.3f} | done: {done}")
    if isinstance(rb, dict):
        # 주요 항목만 보기 좋게 정렬
        keys = ["R_prog","R_level","R_fair","R_energy","R_hum","R_co2","R_act_d","R_act_u","R_track","R_dir","R_safety"]
        ordered = {k: rb[k] for k in keys if k in rb}
        # 누락된 키도 함께 출력
        for k,v in rb.items():
            if k not in ordered:
                ordered[k] = v
        for k, v in ordered.items():
            try:
                print(f"  {k:>8}: {float(v): .4f}")
            except Exception:
                print(f"  {k:>8}: {v}")

    # ---- 파생 체크(방향성 보상 직관 확인) ----
    if len(temps) == len(pre_T):
        dT = temps - pre_T
        tsv = np.array(TSV_VALUES[:len(dT)], dtype=float)
        align = -np.sign(tsv) * dT  # TSV>0이면 dT<0 좋고, TSV<0이면 dT>0 좋음
        print("\nTSV-방향 정합(−sign(TSV)*ΔT):", [f"{x:+.3f}" for x in align])
        if "R_dir" in rb:
            print(f"R_dir(평균정합) ≈ {float(rb['R_dir']):+.4f}")

if __name__ == "__main__":
    main()
