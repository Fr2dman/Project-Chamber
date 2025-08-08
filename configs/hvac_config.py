"""configs/hvac_config.py
전역 HVAC 설정
=================
코어 시뮬레이터 코드를 손대지 않고도 목표값·안전한계·물리 파라미터를
한 곳에서 관리할 수 있도록 모든 상수를 정의합니다.
"""

# ---------------- 기본 공통 ----------------
NUM_ZONES: int = 4  # 상자 존 개수 – Simulator 생성 시 동일하게 맞춰야 함
CONTROL_TERM = 30  # 제어 주기 (초) – 시뮬레이터와 일치해야 함

# ---------------- 목표 조건 ----------------
TARGET_TEMP_C: float = 25.0   # °C
TARGET_RH_PCT: float = 50.0   # % RH
COMFORT_THRESHOLD: float = 80.0  # Comfort score 목표(0–100)

target_conditions = {
    "temperature": [TARGET_TEMP_C] * NUM_ZONES,
    "humidity": [TARGET_RH_PCT] * NUM_ZONES,
    "comfort_threshold": COMFORT_THRESHOLD,
}

# ------------------------------------------------------------
# TSV 하이브리드 목표온도 설정 (보상에만 적용)
#   USE_TSV_HYBRID: True면 보상의 R_track 계산에서 T*_eff 사용
#   K_TSV: TSV 1단위당 목표 이동량(°C) — 0.3~0.5 권장
#   CLAMP_T_EFF_TO_SAFETY: 안전온도 범위로 T*_eff를 클램프할지 여부
# ------------------------------------------------------------
USE_TSV_HYBRID = True          # 보상에서 T*_eff 사용 여부
K_TSV = 0.4                    # °C/TSV 단위 (0.3~0.5 권장)
CLAMP_T_EFF_TO_SAFETY = True   # 안전 온도 범위로 클램프할지

# ---------------- 안전 한계 ----------------
TEMP_LOWER, TEMP_UPPER = 20.0, 30.0  # °C
RH_LOWER,   RH_UPPER   = 25.0, 90.0  # %

safety_limits = {
    "temperature": {"min": TEMP_LOWER,"max": TEMP_UPPER},
    "humidity": {"min": RH_LOWER,"max": RH_UPPER},
}

# ---------------- 물리 파라미터 (physics.py 덮어쓰기) ---------------
C_D   : float = 0.62    # 방출 계수
K_AREA: float = 2.4e-4  # m² per degree (내부 슬롯 면적 계수)
UA    : float = 4.8     # W/°C (벽체 열손실)

# ---------------- 액추에이터 제약 ----------------
SMALL_FAN_MAX_PWM   = 90.0  # %
LARGE_FAN_MAX_PWM   = 90.0  # %
SERVO_INTERNAL_RANGE = (0, 45)
SERVO_EXTERNAL_RANGE = (0, 80)

# ---------------- 존 레이아웃 (시계방향) --------------
# 사용자가 바꾸고 싶으면 여기만 수정.
ZONE_LAYOUT = {
    0: "front_left",
    1: "front_right",
    2: "back_left",
    3: "back_right",
}

# Reward targets & refs
COMFORT_REF = 85.0
COMFORT_BAND_DELTA = 5.0    # 허버 완충폭(점수)
HUMIDITY_BAND = (30.0, 70.0)
CO2_REF = 1000.0            # ppm

# Energy refs
P_REF = 600.0               # 평균 전력 정규화 기준
P_CAP = 800.0               # 피크 억제 캡

# R_level 모드: 'targeted' | 'threshold' | 'maximize'
# targeted : (기존) 목표 85에서 ±이탈을 대칭 벌점(허버)
# threshold: 85 미만만 벌점(권장)
# maximize : 점수 자체를 보상(+), 에너지/습도로 억제
LEVEL_MODE = "threshold"

# Reward weights (초기값)
RW = {
    "prog": 1.0, "level": 0.35, "fair": 0.4,
    "energy": 0.25, "hum": 0.20, "co2": 0.15,
    "act_delta": 0.05, "act_use": 0.02,
    "track": 0.1,     # 목표온도 추적 가중(가볍게 시작)
    "dir": 0.1        # TSV-방향성 보상(가볍게 시작, 필요시 0.0)
}
LAMBDA_RAMP = 0.2
LAMBDA_PEAK = 0.4
