from typing import Literal
import numpy as np
from configs.hvac_config import CONTROL_TERM

class PeltierModel:
    """
    Peltier 장치 모델 (v5)
    - 제어 입력: -1.0(OFF) ~ +1.0(최대 냉각)
    - 상태 변수: cold_side_temp (°C), hot_side_temp (°C)
    - 출력 dict:
        * thermal_power (W, 음수 = 실내에서 제거된 열)
        * power_consumption (W)
        * cold_side_temp (°C)
        * hot_side_temp (°C)

    개선 사항 (v4 -> v5)
    -------------------
    1. 핫측 온도(hot_side_temp)를 에너지 평형식으로 계산하여 현실성 증대.
    2. 냉각량(q_pumping)을 Seebeck 효과, Joule 발열, 열전도 손실을 고려한 물리적 모델로 대체.
    3. 주요 파라미터를 __init__ 인자로 노출하여 실측 기반 튜닝 용이.
    """

    # -------------------------- 상수 --------------------------
    COLD_TEMP_MIN = -50.0
    
    # -------------------------- v5 파라미터 --------------------------
    # 실제 펠티어 소자(JK-TEC1-12706A)의 특성값 기반
    MAX_CURRENT = 6.0          # A (데이터시트)
    SEEBECK_COEFFICIENT = 0.04 # V/K (일반적인 Bi2Te3 소자값)
    
    def __init__(
        self,
        mode: Literal["simple", "precise"] = "precise", # 'precise' 모드 기본값
        *,
        internal_resistance: float = 2.1,  # Ω (데이터시트 2.05~2.26)
        thermal_conductance: float = 0.2,  # W/K (열 누설)
        thermal_mass: float = 200.0,       # J/K (Al 60g + 세라믹)
        heatsink_thermal_resistance: float = 0.5, # K/W (공냉 히트싱크 성능)
        heat_transfer_coeff: float = 10.0, # W/K (공냉 핀+팬)
        tau: float = 30.0                  # s, 1차 지연 상수
    ) -> None:
        self.mode = mode

        # ▶ 파라미터 (실측 기반 튜닝 가능)
        self.internal_resistance = internal_resistance
        self.thermal_conductance = thermal_conductance
        self.thermal_mass = thermal_mass
        self.heatsink_thermal_resistance = heatsink_thermal_resistance
        self.heat_transfer_coeff = heat_transfer_coeff
        self.tau = tau

        # ▶ 상태
        self.cold_side_temp: float = 25.0  # °C, 초기값
        self.hot_side_temp: float = 25.0   # °C, 초기값

    # ----------------------------------------------------------
    def update(
        self,
        control: float,
        chamber_temp: float,
        ambient_temp: float,
        dt: float = CONTROL_TERM  # s
    ) -> dict:
        """한 step 시뮬레이션.

        Parameters
        ----------
        control : float
            -1.0(OFF) ~ 1.0(MAX) 제어 입력.
        chamber_temp : float
            실내 공기(또는 흡입 공기) 온도 [°C].
        ambient_temp : float
            외기 또는 히트싱크 냉각 공기의 온도 [°C].
        dt : float, optional
            적분 시간 간격 [s]. 기본 CONTROL_TERM.
        """
        # 0) 입력 정규화 및 전류 계산
        control = float(np.clip(control, -1.0, 1.0))
        cooling_intensity = (control + 1.0) / 2.0  # 0: OFF, 1: MAX
        i_te = self.MAX_CURRENT * cooling_intensity # A

        # 1) 냉측 온도와 핫측 온도를 이용해 냉각량 및 줄 발열량 계산
        Tc_kelvin = self.cold_side_temp + 273.15
        Th_kelvin = self.hot_side_temp + 273.15

        # Seebeck 효과에 의한 열 흡수량
        q_pumping = self.SEEBECK_COEFFICIENT * i_te * Tc_kelvin
        
        # 줄 발열량 (전체의 절반이 냉측으로 전달)
        q_joule = 0.5 * self.internal_resistance * i_te**2

        # 열전도 손실 (핫측 -> 냉측)
        q_leak = self.thermal_conductance * (self.hot_side_temp - self.cold_side_temp)

        # 2) 핫측 에너지 밸런스 업데이트
        # 핫측으로 들어오는 열량 = 냉각량(q_pumping) + 줄 발열량(q_joule)
        # 핫측에서 나가는 열량 = 히트싱크를 통한 방열
        net_q_hot = q_pumping + q_joule - (self.hot_side_temp - ambient_temp) / self.heatsink_thermal_resistance
        dT_hot = (net_q_hot / self.thermal_mass) * dt
        self.hot_side_temp += dT_hot * (dt / (dt + self.tau)) # 1차 지연 적용

        # 3) 냉측 에너지 밸런스 업데이트
        # 냉측으로 들어오는 열량 = 챔버 공기 대류열(q_conv) + 줄 발열(q_joule) + 열전도(q_leak)
        # 냉측에서 나가는 열량 = 냉각 효과(q_pumping)
        q_conv = self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)
        net_q_cold = q_conv + q_joule + q_leak - q_pumping
        dT_cold = (net_q_cold / self.thermal_mass) * dt
        self.cold_side_temp += dT_cold * (dt / (dt + self.tau)) # 1차 지연 적용

        # 4) 물리적 한계 클램핑
        self.cold_side_temp = np.clip(
            self.cold_side_temp,
            self.COLD_TEMP_MIN,
            self.hot_side_temp - 1.0, # 냉측이 핫측보다 뜨거워지는 역전 방지
        )

        # 5) 출력
        # thermal_power는 챔버에서 제거된 열량이므로 q_conv의 음수값으로 정의
        thermal_power = -self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)
        power_consumption = self.internal_resistance * i_te**2 # 전력 소비량 = I^2 * R

        return {
            "thermal_power": thermal_power,
            "power_consumption": power_consumption,
            "cold_side_temp": self.cold_side_temp,
            "hot_side_temp": self.hot_side_temp,
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