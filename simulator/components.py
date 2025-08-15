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
    # 12706A 역산 기준(ΔT=0에서 Qmax≈65 W, I≈6 A)
    SEEBECK_COEFF = 0.0575  # V/K  (모듈 전체 α)

    def __init__(
        self,
        mode: Literal["simple", "precise"] = "precise",
        *,
        internal_resistance: float = 2.1,
        thermal_conductance: float = 1.0,   # W/K (모듈 열전도 K, 0.8~1.3 권장)
        thermal_mass: float = 200.0,
        heatsink_thermal_resistance: float = 0.35,  # K/W (히트싱크 열저항, 0.25~0.5 권장)
        heat_transfer_coeff: float = 10.0,
        tau: float = 30.0,
        supply_voltage: float = 12.0
    ):
        self.mode = mode
        self.internal_resistance = internal_resistance
        self.thermal_conductance = thermal_conductance
        self.thermal_mass = thermal_mass
        self.heatsink_thermal_resistance = heatsink_thermal_resistance
        self.heat_transfer_coeff = heat_transfer_coeff
        self.tau = tau
        self.supply_voltage = supply_voltage
        self.cold_side_temp = 25.0
        self.hot_side_temp = 25.0
        # 에너지(Wh) 누적
        self.energy_Wh_total = 0.0

    def update(
        self,
        control: float,
        chamber_temp: float,
        ambient_temp: float,
        dt: float = CONTROL_TERM,
        q_load_from_air: float | None = None
    ) -> dict:
        # 1) 입력 정규화 및 제어 → 전류 계산
        control = float(np.clip(control, -1.0, 1.0))
        cooling_intensity = (control + 1.0) / 2.0
        current = self.MAX_CURRENT * cooling_intensity

        # 공통 변수
        alpha = self.SEEBECK_COEFF
        R = self.internal_resistance
        K = self.thermal_conductance
        Tc, Th = self.cold_side_temp, self.hot_side_temp
        TcK = Tc + 273.15
        dT = Th - Tc

        # 2) TEC 냉각/가열 (공급전압 한계로 전류 캡)
        #    Qc: 냉측에서 '펌핑 가능한' 열량(능력치, ≥0), Qh: 핫측으로 버리는 열
        #    V_required = α ΔT + I R  (모듈 양단 전압)
        V_req = alpha * dT + current * R
        if V_req > self.supply_voltage:
            current = max(0.0, (self.supply_voltage - alpha * dT) / R)
            V_req = alpha * dT + current * R

        Qc = alpha * current * TcK - 0.5 * (current ** 2) * R - K * dT
        Qc = max(0.0, Qc)
        Qh = Qc + (current ** 2) * R + alpha * current * (Th - Tc)

        # 3) 핫측 에너지 밸런스 (히트싱크 방열)
        net_hot = Qh - (Th - ambient_temp) / self.heatsink_thermal_resistance
        self.hot_side_temp += (net_hot / self.thermal_mass) * dt * (dt / (dt + self.tau))
        Th = float(self.hot_side_temp)

        # 3) 냉측 에너지 밸런스:  dU = q_in_from_air − Qc
        if q_load_from_air is None:
            q_in = max(0.0, self.heat_transfer_coeff * (chamber_temp - self.cold_side_temp))
        else:
            q_in = max(0.0, q_load_from_air)  # 음수(가열)는 의미 없음
        net_cold = q_in - Qc
        self.cold_side_temp += (net_cold / self.thermal_mass) * dt * (dt / (dt + self.tau))

        # 4) 물리적 클램핑
        self.cold_side_temp = np.clip(
            self.cold_side_temp,
            self.COLD_TEMP_MIN,
            self.hot_side_temp - 1e-3  # 냉측은 항상 핫측보다 낮아야 함
        )

        # 시뮬레이터로 넘길 냉각 잠재력 = Qc (분배 단계에서 ADP/CBF로 실제 가능분만 사용)
        thermal_power = -Qc
        # 전기 소비전력 P = I * V = I * (αΔT + I R)
        power_consumption = current * V_req
        # Wh 누적
        energy_Wh_inc = power_consumption * dt / 3600.0
        self.energy_Wh_total += energy_Wh_inc

        return {
            "thermal_power": thermal_power,                    # W (음수=냉각)
            "power_W": power_consumption,                      # W
            "energy_Wh_increment": energy_Wh_inc,              # Wh
            "energy_Wh_total": self.energy_Wh_total,           # Wh
            "hot_side_temp": self.hot_side_temp,
            "cold_side_temp": self.cold_side_temp
        }
    
    def calculate_actual_cooling(self, current: float, Tc: float, Th: float) -> float:
        alpha = self.SEEBECK_COEFF
        R = self.internal_resistance
        K = self.thermal_conductance
        TcK = Tc + 273.15
        dT = Th - Tc
        return alpha*current*TcK - 0.5*(current**2)*R - K*dT


class FanModel:
    """
    팬 모델
    - 입력: PWM (0~100)
    - 출력: RPM, 소비 전력
    """
    def __init__(
        self,
        max_rpm: float,
        fan_type: str = "small",
        mode: Literal["simple", "precise"] = "simple",
        *,
        rated_power_W: float | None = None,   # 정격(최대) 소비전력
        pwm_deadband: float = 5.0,            # % (5% 이하=0 rpm)
        power_exponent: float = 3.0           # 팬 법칙 기본 n=3
    ):
        self.mode = mode # "simple" or "precise": simple 모드는 target_pwm이 바로 적용됨을 가정. precise 모드는 적용 지연을 고려, 다만 본 시스템에서는 simple로 충분
        self.max_rpm = max_rpm
        self.fan_type = fan_type
        self.current_rpm = 0.0
        self.target_pwm = 0.0

        # 정격전력 기본값: 소형(40mm)≈0.84(한 방향당 2개이므로 1.68적용)W, 대형(120mm)≈3.48W
        self.rated_power_W = rated_power_W if rated_power_W is not None else (1.68 if fan_type == "small" else 3.48)
        self.pwm_deadband = pwm_deadband
        self.power_exponent = power_exponent
        self.energy_Wh_total = 0.0

    def pwm_to_rpm(self, pwm: float) -> float:
        # 데드밴드 보정: deadband% 이하는 0 rpm, 이후 100%까지 선형
        pwm = max(0.0, min(100.0, pwm))
        if pwm <= self.pwm_deadband:
            return 0.0
        eff = (pwm - self.pwm_deadband) / (100.0 - self.pwm_deadband)
        return self.max_rpm * eff
    
    def set_pwm(self, pwm: float) -> float:
        pwm = max(0.0, min(100.0, pwm))
        self.target_pwm = pwm
        self.current_rpm = self.pwm_to_rpm(pwm)
        return self.max_rpm * pwm / 100.0

    def update(self, target_rpm: float, dt: float) -> dict:
        if self.mode == "simple":
            self.current_rpm = target_rpm
        elif self.mode == "precise":
            time_constant = 1.0  # 팬의 시간 상수 (s)
            self.current_rpm += (dt / time_constant) * (target_rpm - self.current_rpm)

        # 소비전력: 정격전력 × (rpm/max)^n  (기본 n=3)
        load_frac = 0.0 if self.max_rpm <= 0 else max(0.0, min(1.0, self.current_rpm / self.max_rpm))
        power_consumption = self.rated_power_W * (load_frac ** self.power_exponent)
        energy_Wh_inc = power_consumption * dt / 3600.0
        self.energy_Wh_total += energy_Wh_inc

        return {
            "rpm": self.current_rpm,
            "power_W": power_consumption,            # W
            "energy_Wh_increment": energy_Wh_inc,    # Wh
            "energy_Wh_total": self.energy_Wh_total, # Wh
        }
    


class ServoModel:
    """
    서보모터 모델
    - 제어 범위 제한, 응답 속도 시뮬레이션 포함
    """
    def __init__(self, min_angle: float, max_angle: float, mode: Literal["simple", "precise"] = "simple"):
        self.mode = mode
        self.min_angle = min_angle
        self.max_angle = max_angle
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