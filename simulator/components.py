from typing import Literal
import numpy as np
from configs.hvac_config import CONTROL_TERM

class PeltierModel:
    """
    Refactored PeltierModel (v6)
    - 단일 냉각량 계산 엔진 사용: 데이터시트 기반 `calculate_actual_cooling`
    - Seebeck, Joule, 전도 손실 통합 모델
    - 명확한 출력: `thermal_power` (W), `power_consumption` (W), temps
    """

    COLD_TEMP_MIN = -50.0  # °C
    MAX_CURRENT = 6.0      # A

    def __init__(
        self,
        mode: Literal["simple", "precise"] = "precise",
        *,
        internal_resistance: float = 2.1,
        thermal_conductance: float = 0.2,
        thermal_mass: float = 200.0,
        heatsink_thermal_resistance: float = 0.5,
        heat_transfer_coeff: float = 10.0,
        tau: float = 30.0
    ):
        self.mode = mode
        self.internal_resistance = internal_resistance
        self.thermal_conductance = thermal_conductance
        self.thermal_mass = thermal_mass
        self.heatsink_thermal_resistance = heatsink_thermal_resistance
        self.heat_transfer_coeff = heat_transfer_coeff
        self.tau = tau
        self.cold_side_temp = 25.0
        self.hot_side_temp = 25.0

    def update(
        self,
        control: float,
        chamber_temp: float,
        ambient_temp: float,
        dt: float = CONTROL_TERM
    ) -> dict:
        # 1) 입력 정규화 및 제어 → 전류 계산
        control = float(np.clip(control, -1.0, 1.0))
        cooling_intensity = (control + 1.0) / 2.0
        current = self.MAX_CURRENT * cooling_intensity

        # 2) 핫측 에너지 밸런스
        # 핫측 유입: Joule 발열 + 전도 손실
        # 핫측 방출: 히트싱크 열 저항
        q_joule = 0.5 * self.internal_resistance * current ** 2
        q_leak = self.thermal_conductance * (self.hot_side_temp - self.cold_side_temp)
        net_hot = q_joule + q_leak - (self.hot_side_temp - ambient_temp) / self.heatsink_thermal_resistance
        dT_hot = (net_hot / self.thermal_mass) * dt
        self.hot_side_temp += dT_hot * (dt / (dt + self.tau))

        # 3) 냉측 에너지 밸런스: 대류열 + Joule + 전도 - 냉각량
        # 대류열(q_conv): 공기와의 열전달
        q_conv = self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)
        # 실제 냉각량 계산(데이터시트 기반)
        q_pumping = self.calculate_actual_cooling(current, self.cold_side_temp, self.hot_side_temp)
        net_cold = q_conv + q_joule + q_leak - q_pumping
        dT_cold = (net_cold / self.thermal_mass) * dt
        self.cold_side_temp += dT_cold * (dt / (dt + self.tau))

        # 4) 물리적 클램핑
        self.cold_side_temp = np.clip(
            self.cold_side_temp,
            self.COLD_TEMP_MIN,
            self.hot_side_temp - 1.0
        )

        # 5) 결과
        thermal_power = -q_pumping  # 음수: 냉각
        power_consumption = current ** 2 * self.internal_resistance

        return {
            "thermal_power": thermal_power,
            "power_consumption": power_consumption,
            "cold_side_temp": self.cold_side_temp,
            "hot_side_temp": self.hot_side_temp,
        }

    def calculate_actual_cooling(self, current: float, Tc: float, Th: float) -> float:
        """
        데이터시트 기반 냉각량
        - base_cooling: 최대 냉각량(65W) × 전류 비율
        - ΔT 의존 성능 감쇠
        """
        current_ratio = current / self.MAX_CURRENT
        base = 65.0 * current_ratio
        dT = Th - Tc
        if dT <= 0:
            eff = 1.0
        elif dT <= 40:
            eff = 1.0 - (dT / 65.0) * 0.7
        else:
            eff = max(0.1, 1.0 - (dT / 65.0))
        return base * eff


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
        self.current_rpm = self.pwm_to_rpm(pwm)
        return self.max_rpm * pwm / 100.0

    def update(self, target_rpm: float, dt: float) -> dict:
        if self.mode == "simple":
            alpha = 0.1
            self.current_rpm += alpha * (target_rpm - self.current_rpm)
        elif self.mode == "precise":
            time_constant = 1.0  # 팬의 시간 상수 (s)
            self.current_rpm += (dt / time_constant) * (target_rpm - self.current_rpm)

        # 에너지 소비량
        power = (self.current_rpm / self.max_rpm) ** 2 * (10 if self.fan_type == "small" else 30)
        return {"rpm": self.current_rpm, "power": power}
    
    def pwm_to_rpm(self, pwm: float) -> float:
        return self.max_rpm * pwm / 100.0


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


# ----------------------------------------------------------------------------------
"""
class PeltierModel:
    
    # Peltier 장치 모델 (v3: 최대 65 W, 표면 온도 클램프)
    # - 제어 입력: -1.0(OFF) ~ +1.0(최대 냉각)
    # - 상태 변수: cold_side_temp (°C)
    # - 출력 dict: thermal_power(W, 음수=냉각), power_consumption(W), cold_side_temp(°C)
    
    # -------------------------- 상수 --------------------------
    COLD_TEMP_MIN = -20.0
    MAX_HEAT_PUMPING_RATE = 65.0        # Qmax @ ΔT=0 K  (datasheet)

    def __init__(self, mode: Literal["simple", "precise"] = "simple"):
        # self.mode = mode

        # ▶ 파라미터 (실측 기반 튜닝)
        self.max_heat_pumping_rate = self.MAX_HEAT_PUMPING_RATE      # 65 W
        self.internal_resistance = 2.1                               # Ω (datasheet 2.05~2.26)
        self.thermal_conductance = 0.2                               # W/K (leakage)
        self.heat_transfer_coeff = 10.0                              # W/K (공냉 핀+팬)
        self.thermal_mass = 200.0                                    # J/K (Al 60 g + 세라믹)
        self.tau = 60.0                                              # s 1-차 지연 상수

        # ▶ 상태
        self.cold_side_temp = 25.0

    # ----------------------------------------------------------
    def update(
        self,
        control: float,
        chamber_temp: float,
        ambient_temp: float,
        dt: float = CONTROL_TERM  # s
    ) -> dict:
        # control ∈ [-1, 1]  →  cooling_intensity ∈ [0, 1]
        # 0) 입력 클램프
        control = float(np.clip(control, -1.0, 1.0))
        cooling_intensity = (control + 1.0) / 2.0      # 0=OFF, 1=MAX
        print("cooling_intensity: ", cooling_intensity)

        # 1) Hot-side 온도 근사 : 주변대비 최대 +15 ℃(공냉)
        hot_side_temp = ambient_temp + 15.0 * cooling_intensity

        # 2) ΔT-의존 열펌핑 (선형 근사 : Q = Qmax·(1-ΔT/ΔTmax))
        dT = max(hot_side_temp - self.cold_side_temp, 0.0)   # K
        Qmax = self.max_heat_pumping_rate * cooling_intensity
        q_pumping = Qmax * (1.0 - dT / 65.0)                 # ΔTmax ≈ 65 K
        q_pumping = np.clip(q_pumping, 0.0, Qmax)

        # 3) 손실 열
        i_te = 6.0 * cooling_intensity                       # A (12 V 기준 선형 근사)
        q_joule = 0.5 * self.internal_resistance * i_te**2   # 냉측에 절반 도달
        q_leak = self.thermal_conductance * dT
        q_conv = self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)

        # 4) 냉측 에너지 밸런스 (1차 지연)
        net_q = q_joule + q_leak + q_conv - q_pumping        # +면 가열
        dT_cold = (net_q / self.thermal_mass) * dt
        alpha = dt / (dt + self.tau)                         # 지연 보정
        self.cold_side_temp += dT_cold * alpha

        #   ▶ 물리 한계 클램프
        self.cold_side_temp = np.clip(
            self.cold_side_temp,
            self.COLD_TEMP_MIN,
            hot_side_temp - 1e-3,                            # 냉측이 핫측보다 뜨거워지는 역전 방지
        )

        # 5) 출력
        thermal_power = -q_conv                              # 음수 = 실내로부터 냉각
        power_consumption = 12.0 * i_te                      # W = V·I (단순 전력)

        return {
            "thermal_power": thermal_power,
            "power_consumption": power_consumption,
            "cold_side_temp": self.cold_side_temp,
        }
"""