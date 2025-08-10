# reward_smoke_test.py
"""
Reward smoke test for AdvancedSmartACSimulator

✅ 무엇을 확인하나요?
- R_prog(진행): 냉각하면 + 로 증가하는가
- R_track(목표추적): TSV 기반 setpoint를 더 잘 따를수록 + 인가
- R_dir(방향성): TSV>0(덥다)일 때 온도 하강(ΔT<0)이면 + 인가
- R_energy(에너지): 출력이 커질수록 더 음수(패널티)인가
- R_fair(공정): 특정 존만 냉각하면 더 큰 음수로 패널티가 생기나

⚙️ 사용법
    python reward_smoke_test.py --steps 6 --seed 42 --verbose
"""
from __future__ import annotations

import argparse
import sys, os
import time
import types
import importlib.util
import numpy as np
from typing import Dict, Tuple, List, Any
from contextlib import contextmanager
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -----------------------------
# 1) 안전한 동적 임포트 헬퍼
# -----------------------------
def _load_module_from_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _try_import_env() -> Tuple[Any, Any]:
    """
    환경 모듈을 가져옵니다.
    - 선호: from simulator.environment import AdvancedSmartACSimulator
    - 대안: 로컬 파일(environment.py, hvac_config.py 등)을 'simulator', 'configs' 별칭으로 매핑
    """
    # 1) 정상 패키지 경로 시도
    try:
        from simulator.environment import AdvancedSmartACSimulator  # type: ignore
        import configs.hvac_config as hvac_config  # type: ignore
        return AdvancedSmartACSimulator, hvac_config
    except Exception:
        # 2) 로컬 파일에서 별칭 패키지 구성
        base = Path(__file__).resolve().parent
        maybe_env = base / "environment.py"
        maybe_cfg = base / "hvac_config.py"
        maybe_sim = {
            "components": base / "components.py",
            "sensors": base / "sensors.py",
            "physics": base / "physics.py",
            "utils": base / "utils.py",
        }
        if not maybe_env.exists():
            raise ImportError(
                "environment.py를 찾을 수 없습니다. "
                "프로젝트 루트에서 실행하거나 PYTHONPATH를 설정해주세요."
            )
        if not maybe_cfg.exists():
            raise ImportError(
                "hvac_config.py를 찾을 수 없습니다. "
                "configs/hvac_config.py가 있다면 PYTHONPATH를 설정하거나 "
                "동일 디렉토리로 복사해주세요."
            )

        # 'configs', 'simulator' 가짜 패키지 생성
        if "configs" not in sys.modules:
            pkg = types.ModuleType("configs")
            pkg.__path__ = []  # namespace pkg 흉내
            sys.modules["configs"] = pkg

        _load_module_from_path("configs.hvac_config", maybe_cfg)

        if "simulator" not in sys.modules:
            pkg = types.ModuleType("simulator")
            pkg.__path__ = []
            sys.modules["simulator"] = pkg

        # 하위 모듈 로드(순서 중요: utils가 외부 의존성 가질 수 있음)
        for sub, path in maybe_sim.items():
            if path.exists():
                _load_module_from_path(f"simulator.{sub}", path)
            else:
                print(f"[경고] {path} 없음 — 일부 기능이 제한될 수 있습니다.", file=sys.stderr)

        env_mod = _load_module_from_path("simulator.environment", maybe_env)
        AdvancedSmartACSimulator = getattr(env_mod, "AdvancedSmartACSimulator")
        import configs.hvac_config as hvac_config  # type: ignore
        return AdvancedSmartACSimulator, hvac_config


# -----------------------------
# 2) 시나리오 러너
# -----------------------------
def run_steps(env, action: np.ndarray, steps: int, tsv: List[float] | None = None):
    """
    주어진 action으로 steps 만큼 step을 실행.
    tsv가 있으면 매 step마다 업데이트.
    return: (rewards, breakdowns 리스트, last_info)
    """
    rewards = []
    breakdowns = []
    last_info = None
    for _ in range(steps):
        if tsv is not None:
            env.update_tsv(tsv)
        obs, r, done, info = env.step(action)
        rewards.append(float(r))
        breakdowns.append(info.get("reward_breakdown", {}))
        last_info = info
    return rewards, breakdowns, last_info


def sum_abs_action_delta(a_prev: np.ndarray, a_next: np.ndarray) -> float:
    return float(np.mean(np.abs(a_next - a_prev)))


# -----------------------------
# 3) 스모크 테스트 시나리오
# -----------------------------
def smoke_test(env, steps: int = 6, verbose: bool = False):
    """
    A) 냉각 후 개선: R_prog > 0 기대
    B) TSV 방향성(+): TSV=+2, 냉각 → R_dir > 0 기대, R_track 개선
    C) 에너지: Off vs Full 출력 → |R_energy_full| > |R_energy_off|
    D) 공정성: 특정 존만 냉각 → R_fair(음수) 크기 증가
    """
    report = []

    # 액션 정의
    off = getattr(env, "off_action", np.full(env.action_dim, -1.0, dtype=np.float32))
    full = off.copy()
    full[0] = +1.0  # peltier full cool
    full[9:13] = +1.0  # small fans high
    full[13] = +1.0    # large fan high
    # 서보는 중간값(= 액션 0.0 → 실제 중간각)로 두거나, 소폭 기울여도 무방
    full[1:9] = 0.0

    # 리셋
    env.reset()
    # baseline 1 step (off)
    _, base_brs, base_info = run_steps(env, off, 1, tsv=[0, 0, 0, 0])
    base = base_brs[-1]
    base_cavg = float(base_info.get("comfort_data", {}).get("average_comfort", 0.0))

    # ---------- A) 냉각 후 개선 ----------
    # off → full 로 스텝 진행
    # 이전 스텝 기준 세팅
    _, _, _ = run_steps(env, off, 1, tsv=[0, 0, 0, 0])
    # 몇 스텝 냉각
    rA, brA, last_infoA = run_steps(env, full, max(2, steps), tsv=[0, 0, 0, 0])
    lastA = brA[-1]
    # 보강된 판정: (i) 마지막 스텝 R_prog>0  OR
    #            (ii) 누적 R_prog>0 AND 평균쾌적도 개선 ≥ 0.5p
    R_prog_last = float(lastA.get("R_prog", 0.0))
    R_prog_sum = float(sum(b.get("R_prog", 0.0) for b in brA))
    last_cavg = float(last_infoA.get("comfort_data", {}).get("average_comfort", base_cavg))
    comfort_gain = last_cavg - base_cavg
    pass_A = (R_prog_last > 0.0) or (R_prog_sum > 0.0 and comfort_gain >= 0.5)
    if verbose:
        print(f"    [A] R_prog_last={R_prog_last:.4f}, R_prog_sum={R_prog_sum:.4f}, "
              f"comfort_gain={comfort_gain:.3f} (last {last_cavg:.2f} - base {base_cavg:.2f})")

    report.append(("A: Cooling improves (R_prog>0)", pass_A, lastA))

    # ---------- B) TSV 방향성(+2) ----------
    env.reset()
    # 먼저 off로 한 번 (ΔT 기준선)
    run_steps(env, off, 1, tsv=[+2, +2, +2, +2])
    # 그 다음 full로 냉각
    rB, brB, _ = run_steps(env, full, max(2, steps), tsv=[+2, +2, +2, +2])
    lastB = brB[-1]
    pass_B_dir = (lastB.get("R_dir", 0.0) > 0.0)
    # R_track이 초기 대비 개선(증가)했는지 체크
    firstB = brB[0]
    pass_B_track = (lastB.get("R_track", 0.0) >= firstB.get("R_track", 0.0) - 1e-6)

    report.append(("B1: TSV>0 cooling direction (R_dir>0)", pass_B_dir, lastB))
    report.append(("B2: Tracking improves under TSV", pass_B_track, {"first_R_track": firstB.get("R_track"), "last_R_track": lastB.get("R_track")}))

    # ---------- C) 에너지 패널티 크기 ----------
    env.reset()
    _, brC_off, _ = run_steps(env, off, 1, tsv=[0, 0, 0, 0])
    _, brC_full, _ = run_steps(env, full, 1, tsv=[0, 0, 0, 0])
    R_energy_off = brC_off[-1].get("R_energy", 0.0)
    R_energy_full = brC_full[-1].get("R_energy", 0.0)
    pass_C = (abs(R_energy_full) > abs(R_energy_off) + 1e-6)
    report.append(("C: Energy penalty scales with power", pass_C, {"R_energy_off": R_energy_off, "R_energy_full": R_energy_full}))

    # ---------- D) 공정성: 특정 존만 냉각 ----------
    env.reset()
    # 균형 냉각(비교군): 펠티어/팬/슬롯 모두 균등 가동
    even = off.copy()
    even[0] = +1.0           # peltier on
    even[1:5] = +1.0         # internal servos 45°
    even[5:9] = +1.0         # external servos 80°
    even[9:13] = +1.0        # small fans high
    even[13] = +1.0          # large fan high
    # 스큐 냉각: 0,1,2 존만 공급(슬롯 open), 3번은 닫음 + 3번 팬 off
    skew = even.copy()
    skew[1:4] = +1.0         # zones 0..2 internal = 45°
    skew[4]   = -1.0         # zone 3 internal = 0°
    skew[12]  = -1.0         # zone 3 small fan off
    # 워밍업
    run_steps(env, off, 1, tsv=[0, 0, 0, 0])
    # 균형 대비 스큐의 공정성 비교
    _, brD_even, _ = run_steps(env, even, max(2, steps), tsv=[0, 0, 0, 0])
    _, brD_skew, _ = run_steps(env, skew, max(2, steps), tsv=[0, 0, 0, 0])
    R_fair_even = brD_even[-1].get("R_fair", 0.0)
    R_fair_skew = brD_skew[-1].get("R_fair", 0.0)
    pass_D = (R_fair_skew < R_fair_even - 1e-6)
    report.append(("D: Fairness penalty (skew worse than balanced)", pass_D, {"R_fair_even": R_fair_even, "R_fair_skew": R_fair_skew}))
    # 결과 출력
    ok = True
    print("\n==== Reward Smoke Test ====")
    for name, passed, detail in report:
        mark = "✅" if passed else "❌"
        ok = ok and passed
        print(f"{mark} {name}")
        if verbose:
            print(f"    details: {detail}")
    print(f"\nOverall: {'PASS' if ok else 'NEEDS ATTENTION'}")
    return ok


# -----------------------------
# 4) 엔트리포인트
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6, help="각 시나리오에서 냉각 액션 지속 스텝 수(>=2 권장)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    try:
        AdvancedSmartACSimulator, hvac_config = _try_import_env()
    except Exception as e:
        print("[임포트 오류] 환경 모듈을 불러오지 못했습니다:", e, file=sys.stderr)
        sys.exit(2)

    try:
        env = AdvancedSmartACSimulator(num_zones=getattr(hvac_config, "NUM_ZONES", 4))
    except Exception as e:
        print("[생성 오류] 시뮬레이터 인스턴스를 만들 수 없습니다:", e, file=sys.stderr)
        sys.exit(3)

    try:
        smoke_test(env, steps=max(2, args.steps), verbose=args.verbose)
    except Exception as e:
        print("[실행 오류] 스모크 테스트 중 예외 발생:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
