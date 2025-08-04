# physics.py – “축소‑현실성” 공기역학 + 에너지 시뮬레이터
# -----------------------------------------------------------------------------
# * 가벼운 계산량으로도 슬롯 각도·팬 PWM 변화가 온·습도 응답에 영향을 주도록 구성
# * 파라미터는 configs/airflow_params.yaml 에서 로드(없으면 디폴트 사용)
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Dict, List

# ▣ 기본 파라미터 (문헌값 / 소형 상자 경험값) – 필요 시 YAML 로드로 대체
C_D_DEFAULT = 0.8                      # 방출계수
K_AREA = 2.5e-5                         # A(θ)=K_AREA*θ  (m², θ in deg)  
UA_DEFAULT = 4.8                        # W / °C  (벽체 열손실)
JET_SPREAD = 0.30                       # rad/m  (제트 확산 계수)
AIR_DENSITY = 1.2                       # kg/m³
CP_AIR = 1005                           # J/kg °C  (정압비열)
BOX_VOLUME = 0.096                      # m³  (0.6×0.4×0.4)
BOX_MASS_AIR = BOX_VOLUME * AIR_DENSITY # kg  (≈0.115)
THERMAL_CAP = BOX_MASS_AIR * CP_AIR     # J/°C

# ---------------------------------------------------------------------------------
# Helper – 슬롯 각도→개구 면적, 팬 PWM→RPM 간단 선형 관계
# ---------------------------------------------------------------------------------

def slot_area(theta_deg: float) -> float:
    """슬롯 각도 → 개구 면적 A [m²]  (선형 근사)"""
    return K_AREA * max(0.0, theta_deg)


def pwm_to_rpm(pwm: float, rpm_max: float) -> float:
    return rpm_max * (pwm / 100.0)


# ---------------------------------------------------------------------------------
# JetModel :  내부슬롯·팬 → flow_matrix (nx n) 생성
# ---------------------------------------------------------------------------------

class JetModel:
    def __init__(self, num_zones: int = 4, c_d: float = C_D_DEFAULT):
        self.n = num_zones
        self.c_d = c_d
        # 2×2 기준 인접 리스트
        self.adj = {0: [1, 2],
                    1: [0, 3],
                    2: [0, 3],
                    3: [1, 2]}

        # 제트 모델 파라미터 (상수화하여 가독성 및 유지보수성 향상)
        self.MAX_FAN_RPM = 7000.0      # small_fan의 최대 RPM
        self.JET_PRESSURE_DIFF = 50.0  # Pa, 팬에 의한 기준 압력차
        self.JET_SELF_RATIO = 0.90     # 제트 유량의 자기 존 기여 비율
        self.NATURAL_MIX_RATE = 0.02   # s⁻¹, 자연 혼합 체적 교환율
        
    def get_flow_matrix(self, fan_rpms: List[float], theta_int: List[float]) -> np.ndarray:
        """fan_rpms, theta_int 길이는 num_zones.
        반환: n×n 유량행렬 [m³/s]  (대각=유입+자연혼합, off‑diag=zone간 전달)"""
        Q = np.zeros((self.n, self.n))
        for i in range(self.n):
            area = slot_area(theta_int[i])
            # 기준 압력차에 팬 RPM 비율을 곱하여 유효 압력차 계산
            effective_pressure = self.JET_PRESSURE_DIFF * (fan_rpms[i] / self.MAX_FAN_RPM)
            v_exit = (2 * effective_pressure / AIR_DENSITY) ** 0.5
            volumetric = self.c_d * area * v_exit   # m³/s

            # 분배: 자기 존에 대부분, 나머지는 인접 존으로 동적 계산
            Q[i, i] += volumetric * self.JET_SELF_RATIO
            adj_ratio = (1.0 - self.JET_SELF_RATIO) / len(self.adj[i])
            for j in self.adj[i]:
                Q[i, j] += volumetric * adj_ratio
                
        # 자연혼합 (창문 없는 박스 내부 난류) – 체적 비율 0.02/s
        natural = self.NATURAL_MIX_RATE * BOX_VOLUME
        Q += natural * (np.ones((self.n, self.n)) - np.eye(self.n)) / (self.n - 1)
        return Q


# ---------------------------------------------------------------------------------
# ZoneEnergyBalance : 열·수분 수지 계산
# ---------------------------------------------------------------------------------

class ZoneEnergyBalance:
    def __init__(self, ua: float = UA_DEFAULT, thermal_cap: float = THERMAL_CAP):
        self.ua = ua
        self.C = thermal_cap  # J/°C 전체 공기열용량 (존 동일 가정)

    def step(self, T: np.ndarray, H: np.ndarray, Q_flow: np.ndarray,
             Q_peltier: float, dt: float, T_amb: float, H_amb: float) -> tuple[np.ndarray, np.ndarray]:
        """T,H: (n,)  — 온도[°C], 상대습도[%]
        Q_flow: n×n (m³/s)   Q_peltier (W, )
        단순 절대습량 모델 ⇒ H_update  (정밀 모델은 나중에 교체)"""
        n = T.size
        m_dot = Q_flow * AIR_DENSITY               # kg/s
        zone_thermal_cap = self.C / n
        zone_air_mass = BOX_MASS_AIR / n

        # 온도 변화 (벡터화)
        # 1. 대류: m_dot.T @ T는 각 존으로 유입되는 온도의 질량 가중합, m_dot.sum(0) * T는 유출되는 온도의 질량 가중합
        conv_energy_rate = CP_AIR * (m_dot.T @ T - m_dot.sum(axis=0) * T)
        # 2. 벽체 열손실
        wall_loss_rate = -self.ua * (T - T_amb)
        # 3. 펠티어 열량 (모든 존에 균등 분배)
        peltier_rate = Q_peltier / n

        dT = (conv_energy_rate + wall_loss_rate + peltier_rate) / zone_thermal_cap
        T_new = T + dT * dt

        # 습도 변화 (벡터화)
        # 1. 혼합
        mix_rate = (m_dot.T @ H - m_dot.sum(axis=0) * H) / zone_air_mass
        # 2. 미소 환기
        vent_rate = 0.001 * (H_amb - H)
        dH = mix_rate + vent_rate
        H_new = np.clip(H + dH * dt, 0, 100)
        return T_new, H_new


# ---------------------------------------------------------------------------------
# PhysicsSimulator (고도화 버전)
# ---------------------------------------------------------------------------------

class PhysicsSimulator:
    """환경 상자 내 공기 물리 시뮬레이터 – RL 학습 단계에서 호출"""
    def __init__(self, num_zones: int = 4):
        self.n = num_zones
        self.T = np.random.uniform(22, 28, size=self.n)
        self.H = np.random.uniform(40, 60, size=self.n)
        self.CO2 = np.random.uniform(400, 800, size=self.n)
        self.Dust = np.random.uniform(0, 10, size=self.n)

        self.ambient_temp = 30.0
        self.ambient_hum = 70.0

        self.jet = JetModel(self.n)
        self.balance = ZoneEnergyBalance()

    # ------------------------------------------------------------------
    def reset(self):
        self.__init__(self.n)

    def get_current_state(self) -> Dict[str, np.ndarray]:
        return {
            'temperatures': self.T,
            'humidities': self.H,
            'co2_levels': self.CO2,
            'dust_levels': self.Dust,
        }

    # ------------------------------------------------------------------
    def update_physics(self, action_dict: Dict, peltier_states: Dict, fan_states: Dict,
                       dt: float = 15.0) -> Dict[str, np.ndarray]:
        """action_dict는 슬롯 각도·PWM 포함(환경에서 전달),
        peltier_states[0]['power_consumption'] W, fan_states['small_fans'] 리스트"""
        small_rpms = [fan['rpm'] for fan in fan_states['small_fans']]
        theta_int = action_dict['internal_servo_angles']

        # 1) 유량 행렬
        Q_matrix = self.jet.get_flow_matrix(small_rpms, theta_int)  # m³/s

        # 2) 펠티어 냉·열량 (음수=냉각)
        Q_pelt = peltier_states[0]['thermal_power']

        # 3) 에너지·습도 수지 계산
        self.T, self.H = self.balance.step(self.T, self.H, Q_matrix, Q_pelt,
                                           dt, self.ambient_temp, self.ambient_hum)

        # 4) 단순 CO₂, Dust – 환기 기반 지수 감소 모델
        decay = np.clip(Q_matrix.sum(axis=0) / BOX_VOLUME, 0, 1)
        self.CO2 = 350 + (self.CO2 - 350) * np.exp(-decay * dt)
        self.Dust = self.Dust * np.exp(-decay * dt) + np.random.normal(0, 0.05, size=self.n)
        self.Dust = np.clip(self.Dust, 0, 100)

        return self.get_current_state()

# sim = PhysicsSimulator()
# print(sim.T)
# dummy_action = {
#     'peltier_control': -0.5,
#     'internal_servo_angles': [40,40,40,40],
#     'external_servo_angles': [0,40,80,20],
#     'small_fan_pwm': [50,50,50,50],
#     'large_fan_pwm': 0,
# }
# fan_states = {'small_fans':[{'rpm':3500,'power':5}]*4}
# p_state = {0:{'power_consumption':15}}
# state = sim.update_physics(dummy_action, p_state, fan_states)
# print(state['temperatures'])
