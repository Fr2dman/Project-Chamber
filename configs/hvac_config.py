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
TARGET_TEMP_C: float = 24.0   # °C
TARGET_RH_PCT: float = 50.0   # % RH
COMFORT_THRESHOLD: float = 80.0  # Comfort score 목표(0–100)

target_conditions = {
    "temperature": [TARGET_TEMP_C] * NUM_ZONES,
    "humidity": [TARGET_RH_PCT] * NUM_ZONES,
    "comfort_threshold": COMFORT_THRESHOLD,
}

# ---------------- 안전 한계 ----------------
TEMP_LOWER, TEMP_UPPER = 20.0, 26.0  # °C
RH_LOWER,   RH_UPPER   = 35.0, 65.0  # %

safety_limits = {
    "temperature": (TEMP_LOWER, TEMP_UPPER),
    "humidity": (RH_LOWER, RH_UPPER),
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
