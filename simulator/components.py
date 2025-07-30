from typing import Literal


class PeltierModel:
    """
    Peltier 장치 모델
    - 제어 입력: 0(냉각 종료) ~ +1.0(최대 냉각)
    - 출력: 온도 변화율, 소비 전력
    """
    def __init__(self, mode: Literal["simple", "precise"] = "simple"):
        self.mode = mode
        self.last_power = 0.0

    def update(self, control: float, ambient_temp: float, dt: float) -> dict:
        control = max(-1.0, min(1.0, control))

        if self.mode == "simple":
            cooling_power = 30.0 * -control  # ±30W
            temp_change = (cooling_power / 1000) * dt  # 간단한 냉각 효과
        elif self.mode == "precise":
            # 정밀 모드: 시간 상수, 열전달계수 적용
            thermal_capacity = 1000  # J/°C
            heat_transfer_coeff = 5  # W/°C
            q = 30.0 * -control
            temp_change = (q - heat_transfer_coeff * (ambient_temp - 25)) * dt / thermal_capacity

        power = abs(control) * 30.0  # 소비 전력(W)
        self.last_power = power

        return {"temp_change": temp_change, "power_consumption": power}


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
