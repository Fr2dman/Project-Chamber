"""configs/hvac_config.py
전역 HVAC 설정
=================
코어 시뮬레이터 코드를 손대지 않고도 목표값·안전한계·물리 파라미터를
한 곳에서 관리할 수 있도록 모든 상수를 정의합니다.
"""

# ---------------- 기본 공통 ----------------
NUM_ZONES: int = 4  # 상자 존 개수 – Simulator 생성 시 동일하게 맞춰야 함
CONTROL_TERM = 15  # 제어 주기 (초) – 시뮬레이터와 일치해야 함
DT_SECONDS = CONTROL_TERM  # 외부 모듈에서 시간스텝을 명시적으로 참조하고 싶을 때 사용(옵션)

# ---------------- 존 레이아웃  --------------
# 사용자가 바꾸고 싶으면 여기만 수정.
ZONE_LAYOUT = {
    0: "front_left",
    1: "front_right",
    2: "back_left",
    3: "back_right",
}

# ---------------- 목표 조건 ----------------
TARGET_TEMP_C: float = 25.0   # °C
TARGET_RH_PCT: float = 50.0   # % RH
COMFORT_THRESHOLD: float = 85.0  # Comfort score 목표(0–100)

target_conditions = {
    "temperature": [TARGET_TEMP_C] * NUM_ZONES,
    "humidity": [TARGET_RH_PCT] * NUM_ZONES,
    "comfort_threshold": COMFORT_THRESHOLD,
}

# 학습 시 에피소드별 목표 온도 랜덤화(도메인 랜덤라이제이션)
# - enable=True 로 두면 reset()마다 목표온도를 샘플링하여 self.T_target 으로 사용
# - per_zone=True 면 존별로 서로 다른 목표를 샘플링
TARGET_RANDOMIZE = {
    "enable": True,
    "per_zone": True,
    "mode": "uniform",                 # "uniform" | "discrete"
    "uniform_range": (23.0, 27.0),     # °C
    "discrete_pool": [23.0, 24.0, 25.0, 26.0, 27.0]
}

# ------------- 안전 한계 ----------------
TEMP_LOWER, TEMP_UPPER = 20.0, 30.0  # °C
RH_LOWER,   RH_UPPER   = 25.0, 90.0  # %

safety_limits = {
    "temperature": {"min": TEMP_LOWER,"max": TEMP_UPPER},
    "humidity": {"min": RH_LOWER,"max": RH_UPPER},
}

# ---------------- 물리 파라미터 (physics.py 덮어쓰기) ---------------
C_D   : float = 0.62    # 방출 계수
K_AREA: float = 1.3e-4  # m² per degree (내부 슬롯 면적 계수)
UA    : float = 4.8     # W/°C (벽체 열손실)

# ---------------- 액추에이터 제약 ----------------
SMALL_FAN_MAX_PWM   = 90.0  # %
LARGE_FAN_MAX_PWM   = 90.0  # %
SERVO_INTERNAL_RANGE = (0, 45)
SERVO_EXTERNAL_RANGE = (0, 80)


# ------------------------------------------------------------
# TSV 하이브리드 목표온도 설정 (보상에만 적용)
#   USE_TSV_HYBRID: True면 보상의 R_track 계산에서 T*_eff 사용
#   K_TSV: TSV 1단위당 목표 이동량(°C) — 0.3~0.5 권장
#   CLAMP_T_EFF_TO_SAFETY: 안전온도 범위로 T*_eff를 클램프할지 여부
# ------------------------------------------------------------
USE_TSV_HYBRID = True          # 보상에서 T*_eff 사용 여부
K_TSV = 0.8                    # °C/TSV (0.6~0.8 권장) — TSV 반영 강도 ↑
CLAMP_T_EFF_TO_SAFETY = True   # 안전 온도 범위로 클램프

# --- TSV/트래킹 고급 설정 ---
TSV_DEADBAND = 0.5             # |TSV|가 이보다 작으면 목표 이동 0 (노이즈 억제)
T_EFF_EMA_ALPHA = 0.30         # T_eff = (1-α)·prev + α·raw  (목표 스무딩)
TRACK_BAND = 1.5               # setpoint 추적 밴드(°C) — 작을수록 강한 보상
TRACK_TSV_WEIGHT_SCALE = 0.5   # 트래킹 가중치에 (1+scale·|TSV|) 적용
DIR_DT_NORM = 0.2              # 방향성 보조항 ΔT 정규화 기준(°C/step)

# ── 합성 TSV(온도 임계 기반) ─────────────────────────────
TSV_SIM = dict(
    enable=True,          # 학습 때만 True, 실전은 False
    mode="hybrid",      # "absolute" | "hybrid" (T_eff 기준)
    hot_thr=27.0,         # 이 온도↑에서 "더워요" 발생 확률↑
    cold_thr=24.0,        # 이 온도↓에서 "추워요" 발생 확률↑
    p_base=0.20,          # 기본 발생 확률
    p_k=0.5,             # (온도초과 °C)당 확률 증가량
    slope_deg=1.2,        # tanh 스케일(몇 도에서 강한 TSV가 나오게 할지)
    sigma=0.20,           # TSV 등급 노이즈(연속값→라운드 전)
    flip_prob=0.01,       # 가끔 반대로 누르는 오표기 확률
    decay=0.90,           # 피드백 없을 때 0으로 감쇠
    bias_std=0.6          # 존/사람 성향 바이어스 표준편차
)

# Reward targets & refs
COMFORT_REF = 85.0
COMFORT_BAND_DELTA = 5.0    # 허버 완충폭(점수)
HUMIDITY_BAND = (30.0, 70.0)
CO2_REF = 1000.0            # ppm

# Energy refs
P_REF = 80.0               # 평균 전력 정규화 기준
P_CAP = 120.0               # 피크 억제 캡

LAMBDA_RAMP = 0.2
LAMBDA_PEAK = 0.4

# --- 에너지 보상 게이팅: 쾌적 확보 후에만 에너지 절약 유도 ---
USE_ENERGY_GATE = True
ENERGY_GATE = {
    "MIN_ALL": 80.0,        # 최저 쾌적 점수 ≥ 80
    "PCT_GOOD": 0.80,       # 다음 조건 중 하나라도 만족하면 게이트 ON:
    "GOOD_THRESH": 80.0     #   (i) 최저 ≥ MIN_ALL  or  (ii) ≥GOOD_THRESH 존 비율 ≥ PCT_GOOD
}
# NOTE: ENERGY_MODE="cumulative" 에서는 위 게이트는 사용되지 않음

# ------------------------------------------------------------
# 누적 에너지 기반(터미널 1회) 보상 모드 설정
#   - ENERGY_MODE="cumulative" 이면 R_energy를
#     "쾌적 재달성까지 누적 Wh" vs "ΔT_start 기반 예산 Wh"로 산정
#   - 스텝별 R_energy, 램프/피크 패널티는 계산하지 않음
# ------------------------------------------------------------
ENERGY_MODE = "cumulative"        # "cumulative" | "off"
ENERGY_STEPS_PER_DEG = 4.0        # 1°C 차이를 메우는 데 기준 스텝 수(예: dt=15s면 90초/°C)
ENERGY_MIN_BUDGET_WH = 4.0       # 예산 하한(너무 작은 예산 방지)
SUCCESS_RULE = {                   # 쾌적 성공 판정(히스테리시스)
    "AVG": 85.0,                   # 평균 쾌적 임계
    "MIN_ALL": 80.0,               # 최저 쾌적 임계
    "STREAK": 5,                   # 연속 스텝 수(성공 유지)
    "DROP": 72.0                   # 실패 판정 하한(떨어지면 추격 재개)
}
ENERGY_MAINT_STEP_COEF = 0.03   # η in R_energy_step = -η·(E_step_Wh/step_wh_ref)
ENERGY_PURSUIT_STEP_COEF = 0.015 # 추격 구간에서도 스텝 Wh에 아주 작게 패널티

# R_level 모드: 'targeted' | 'threshold' | 'maximize'
# targeted : (기존) 목표 85에서 ±이탈을 대칭 벌점(허버)
# threshold: 85 미만만 벌점(권장)
# maximize : 점수 자체를 보상(+), 에너지/습도로 억제
LEVEL_MODE = "threshold"

# Reward weights (트래킹/TSV 강조 프로파일; 에너지 게이트로 관리)
RW = {
    # ── 핵심 밀도 신호 ──
    "prog": 0.8,     # 진행(개선량)
    "level": 0.35,   # 남은 불쾌도
    "fair": 0.20,    # 최악 존 억제(너무 크지 않게)
    "track": 0.30,   # 목표 추적
    # ── 환경 제약 ──
    "hum": 0.20, "co2": 0.15,
    # ── 조작 비용 ──
    "act_delta": 0.05,  # 액션 변화량(Δ|a|)
    "act_use": 0.00, # 비활성(중복 방지)
    # ── TSV 보조 ──
    "dir": 0.03,         # 기본 OFF (노이즈 방지)
    "cool_align": 0.5,  # 또는 이걸 소량만(둘 중 하나만 쓰세요)
    # ── 터미널 에너지 ──
    "energy": 0.20       # 누적 에너지(터미널) 보상 세기
}

# ------------------------------------------------------------
# 풍속 추정 계수 (v_i ≈ V0 + A_SMALL·(rpm_s/7000) + B_LARGE·(rpm_L/3300)·f(θ_ext))
#   - f(θ_ext): 'linear' → (1 - θ/80)  (0° 직하, 80° 수평 확산)
#   - 전체 풍속은 [VMIN, VMAX]로 클립
# ------------------------------------------------------------
AIR_VEL = {
    "V0": 0.10,        # m/s, 베이스
    "A_SMALL": 0.65,   # 소형팬 가중
    "B_LARGE": 0.55,   # 대형팬 가중
    "VMIN": 0.10,
    "VMAX": 0.80,
    "ANGLE_MODE": "linear"  # 'linear' | 'cos'
}

# ------------------------------------------------------------
# (옵션) reward.yaml 로더: 존재 시 가중치/모드 덮어쓰기
#   - 기본 탐색 경로: 
#       1) 환경변수 REWARD_YAML
#       2) 이 파일과 같은 폴더의 'reward.yaml'
# ------------------------------------------------------------
try:
    import os
    try:
        import yaml  # PyYAML
    except Exception:
        yaml = None

    _candidates = []
    if os.getenv("REWARD_YAML"):
        _candidates.append(os.getenv("REWARD_YAML"))
    _candidates.append(os.path.join(os.path.dirname(__file__), "reward.yaml"))

    for _p in _candidates:
        if not _p:
            continue
        if os.path.isfile(_p) and yaml is not None:
            with open(_p, "r", encoding="utf-8") as _f:
                _y = yaml.safe_load(_f) or {}
            # RW 가중치 병합 (주의: reward.yaml에 'prog' 넣으면 그것도 덮입니다)
            if isinstance(_y.get("RW"), dict):
                RW.update(_y["RW"])
            # 선택 항목들
            if "LEVEL_MODE" in _y:
                LEVEL_MODE = _y["LEVEL_MODE"]
            if "USE_ENERGY_GATE" in _y:
                USE_ENERGY_GATE = bool(_y["USE_ENERGY_GATE"])
            if isinstance(_y.get("ENERGY_GATE"), dict):
                ENERGY_GATE.update(_y["ENERGY_GATE"])
            break
except Exception:
    # 로더 실패는 무시(기본값 사용)
    pass
