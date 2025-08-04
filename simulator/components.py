from typing import Literal
import numpy as np


class PeltierModel:
    """
    Peltier 장치 모델 (v2: 표면 온도 기반 동적 모델)
    - 제어 입력: -1.0(냉각 정지) ~ +1.0(최대 냉각)
    - 상태 변수: cold_side_temp (냉각면 온도)
    - 출력: 열량(W), 소비 전력(W)
    """
    def __init__(self, mode: Literal["simple", "precise"] = "simple"):
        self.mode = mode
        
        # 물리 파라미터 (튜닝 가능)
        self.max_heat_pumping_rate = 30.0  # W, 최대 열 펌핑률 (제벡 효과)
        self.internal_resistance = 0.5     # 옴, 내부 저항 (줄 발열 관련)
        self.thermal_conductance = 0.2     # W/K, 핫사이드-콜드사이드 간 열전도율
        self.heat_transfer_coeff = 1.5     # W/K, 냉각면-공기 간 열전달계수
        self.thermal_mass = 20.0           # J/K, 냉각면의 열용량

        # 상태 변수
        self.cold_side_temp = 25.0

    def update(self, control: float, chamber_temp: float, ambient_temp: float, dt: float = 30) -> dict:
        control = max(-1.0, min(1.0, control))

        # 제어 신호 [-1, 1]을 냉각 강도 [0, 1]로 스케일링
        cooling_intensity = (control + 1.0) / 2.0

        # 1. 핫사이드 온도 추정 (간단히 주변 온도로 근사)
        hot_side_temp = ambient_temp + 5.0 * cooling_intensity # 작동 시 뜨거워지는 것 반영

        # 2. 펠티어 냉각면의 열 흐름 계산
        # (1) 제벡 효과 (열 펌핑): 챔버에서 열을 빼앗음
        q_pumping = self.max_heat_pumping_rate * cooling_intensity
        # (2) 줄 발열: 내부 저항으로 인해 발생하는 열 (절반이 냉각면으로)
        q_joule = 0.5 * self.internal_resistance * (cooling_intensity * 10)**2 # 전류를 강도로 근사
        # (3) 열 누설: 핫사이드에서 콜드사이드로 새는 열
        q_leak = self.thermal_conductance * (hot_side_temp - self.cold_side_temp)
        # (4) 챔버와의 열 교환: 냉각면과 공기 사이의 열 전달
        q_convection = self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp)

        # 3. 냉각면 온도 업데이트
        # dQ = (들어온 열) - (나간 열). 펌핑된 열은 냉각면에서 나간 것.
        net_heat_flow = q_joule + q_leak + q_convection - q_pumping
        self.cold_side_temp += (net_heat_flow / self.thermal_mass) * dt

        # 4. 최종 출력 계산
        thermal_power = -q_convection  # 챔버에 전달된 순수 냉각량 (음수)
        power_consumption = self.max_heat_pumping_rate * cooling_intensity # 소비전력은 단순 모델 유지

        return {"thermal_power": thermal_power, "power_consumption": power_consumption}


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
