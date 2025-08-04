from typing import Literal
import numpy as np

class PeltierModel:
    """
    Peltier 장치 모델 (v3: 최대 65 W, 표면 온도 클램프)
    - 제어 입력: -1.0(OFF) ~ +1.0(최대 냉각)
    - 상태 변수: cold_side_temp (°C)
    - 출력 dict: thermal_power(W, 음수=냉각), power_consumption(W), cold_side_temp(°C)
    """

    # -------------------------- 상수 --------------------------
    COLD_TEMP_MIN = -50.0           # °C, 냉각면 물리적 한계
    MAX_HEAT_PUMPING_RATE = 65.0    # W, 데이터시트/설계치

    def __init__(self, mode: Literal["simple", "precise"] = "simple"):
        self.mode = mode

        # ▶ 파라미터 (튜닝 가능)
        self.max_heat_pumping_rate = self.MAX_HEAT_PUMPING_RATE  # W
        self.internal_resistance = 0.5     # Ω, Joule heating
        self.thermal_conductance = 0.2     # W/K, hot↔cold leakage
        self.heat_transfer_coeff = 1.5     # W/K, cold-side ↔ air
        self.thermal_mass = 20.0           # J/K, cold-side lumped C

        # ▶ 상태
        self.cold_side_temp = 25.0  # °C, 초기 실내온도와 동일 가정

    # ----------------------------------------------------------
    def update(
        self,
        control: float,
        chamber_temp: float,
        ambient_temp: float,
        dt: float = 30.0,
    ) -> dict:
        """펠티어 1 사이클(기본 30 s) 시뮬레이션
        control ∈ [-1,1] → 냉각 intensity ∈ [0,1]
        """
        # 0) 입력 클램프
        control = float(np.clip(control, -1.0, 1.0))
        cooling_intensity = (control + 1.0) / 2.0  # 0~1

        # 1) Hot-side 온도(간략) : 주변보다 최대 +5 °C 상승
        hot_side_temp = ambient_temp + 5.0 * cooling_intensity

        # 2) 열 흐름 계산
        q_pumping = self.max_heat_pumping_rate * cooling_intensity                  # 제벡
        q_joule = 0.5 * self.internal_resistance * (cooling_intensity * 10) ** 2   # 내부저항 일부 냉측
        q_leak = self.thermal_conductance * (hot_side_temp - self.cold_side_temp)   # 누설
        q_conv = self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)    # 냉측 ↔ 공기

        # 3) 에너지 잔고 → 냉측 온도 업데이트
        net_q = q_joule + q_leak + q_conv - q_pumping   # +면 가열, –면 냉각
        self.cold_side_temp += (net_q / self.thermal_mass) * dt
        #   ▶ 물리 한계 적용
        if self.cold_side_temp < self.COLD_TEMP_MIN:
            self.cold_side_temp = self.COLD_TEMP_MIN

        # 4) 출력 (챔버 기준 열량은 –q_conv)
        thermal_power = -q_conv                       # W (음수=냉각)
        power_consumption = self.max_heat_pumping_rate * cooling_intensity  # 단순 소비전력 모델

        return {
            "thermal_power": thermal_power,
            "power_consumption": power_consumption,
            "cold_side_temp": self.cold_side_temp,
        }


class FanModel:
    """
    팬 모델
    - 입력: PWM (0~100)
    - 출력: RPM, 소비 전력
    """
    def __init__(self, max_rpm: float, fan_type: str = "small", mode: Literal["simple", "precise"] = "simple"):
        self.max_rpm = max_rpm
        self.fan_type = fan_type
        self.mode = mode
        self.current_rpm = 0.0
        self.target_pwm = 0.0

    def set_pwm(self, pwm: float) -> float:
        pwm = max(0.0, min(100.0, pwm))
        self.target_pwm = pwm
        return self.max_rpm * pwm / 100.0

    def update(self, target_rpm: float, dt: float) -> dict:
        if self.mode == "simple":
            alpha = 0.1
            self.current_rpm += alpha * (target_rpm - self.current_rpm)
        elif self.mode == "precise":
            time_constant = 1.5  # 팬의 시간 상수 (s)
            self.current_rpm += (dt / time_constant) * (target_rpm - self.current_rpm)

        # 에너지 소비량
        power = (self.current_rpm / self.max_rpm) ** 2 * (10 if self.fan_type == "small" else 30)
        return {"rpm": self.current_rpm, "power": power}


class ServoModel:
    """
    서보모터 모델
    - 제어 범위 제한, 응답 속도 시뮬레이션 포함
    """
    def __init__(self, min_angle: float, max_angle: float, mode: Literal["simple", "precise"] = "simple"):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.mode = mode
        self.current_angle = (min_angle + max_angle) / 2
        self.target_angle = self.current_angle

    def set_angle(self, angle: float):
        self.target_angle = max(self.min_angle, min(self.max_angle, angle))

    def update(self, dt: float):
        if self.mode == "simple":
            self.current_angle = self.target_angle
        elif self.mode == "precise":
            angle_diff = self.target_angle - self.current_angle
            max_speed = 30.0  # deg/s
            step = max_speed * dt
            if abs(angle_diff) <= step:
                self.current_angle = self.target_angle
            else:
                self.current_angle += step if angle_diff > 0 else -step
