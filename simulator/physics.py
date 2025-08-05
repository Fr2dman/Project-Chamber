"""physics.py – unified version (v2.1)
====================================================
* 65 W Peltier 냉각 분배 로직 적용
* 기존 환경(`AdvancedSmartACSimulator`)과 **완전 호환**되는 래퍼 유지
* JetModel 시그니처 확장 `(fan_rpms, θ_int, θ_ext=None)` – 레거시 두‑인자 호출도 OK
* 외부 슬롯 각도 영향은 아직 계수만 입력받고 미사용 (다음 단계)

Author: ChatGPT (August 2025)
"""
from __future__ import annotations
from typing import Sequence, Optional, Dict, List
import numpy as np

# -----------------------------------------------------------------------------
# 전역 상수 (문헌 + 실험값)
# -----------------------------------------------------------------------------
CP_AIR = 1005.0          # J/(kg·K)
RHO_AIR = 1.2            # kg/m³
DEFAULT_ZONE_VOL = 0.024 # m³ (가로30×세로20×높이40 cm - 4존 기준)
AMBIENT_TEMP = 30.0      # °C
AMBIENT_HUM = 70.0       # %RH
# -----------------------------------------------------------------------------
# ZoneEnergyBalance : 존별 열·수분 수지 (펠티어 배열 지원)
# -----------------------------------------------------------------------------
class ZoneEnergyBalance:
    """단순 박스 모델 – 온도(°C)·상대습도(%) 업데이트"""

    def __init__(self, zone_volumes: Sequence[float], ua_wall: float = 0.05):
        self.V = np.asarray(zone_volumes)           # m³
        self.m = RHO_AIR * self.V                   # kg
        self.C = CP_AIR * self.m                    # J/K
        self.ua_wall = ua_wall                      # W/K

    # ---------------------------------------------------------------------
    def step(
        self,
        temps: np.ndarray,             # (N,)
        humidities: np.ndarray,        # (N,)
        q_matrix: np.ndarray,          # (N,N) m³/s
        peltier_rates: np.ndarray,     # (N,)  W (음수=냉각)
        ambient_temp: float,
        dt: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """1 사이클(30 s) 적용 후 (T,H) 반환"""
        m_dot = q_matrix * RHO_AIR                     # kg/s
        m_out = m_dot.sum(axis=1)                      # kg/s
        # --- 열수지 ---
        q_conv = CP_AIR * (m_dot.T @ temps - m_out * temps)
        q_wall = -self.ua_wall * (temps - ambient_temp)
        q_total = q_conv + q_wall + peltier_rates
        temps_new = temps + (q_total / self.C) * dt

        # --- 습도 (단순 혼합 + 환기) ---
        w_in  = humidities
        w_out = (q_matrix.T @ humidities + (AMBIENT_HUM - humidities) * m_out) / (self.m / dt)
        humidities_new = humidities + (w_out - w_in) * dt / (self.m / RHO_AIR)
        humidities_new = np.clip(humidities_new, 0, 100)
        return temps_new, humidities_new

# -----------------------------------------------------------------------------
# JetModel : 팬 RPM + 슬롯 각도 → 체적유량 행렬
# -----------------------------------------------------------------------------
class JetModel:
    def __init__(self, num_zones: int = 4, c_d: float = 0.8):
        self.n = num_zones
        self.c_d = c_d
        self.adj = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        # 파라미터
        self.MAX_FAN_RPM = 7000.0
        self.DELTA_P_REF = 50.0      # Pa
        self.JET_SELF_RATIO = 0.9
        self.NATURAL_MIX_RATE = 0.02 # s⁻¹
        self.K_AREA = 5.3e-5         # m²/deg  (slot 개구 면적 계수)

    # ------------------------------------------------------------
    def _slot_area(self, theta_deg: float) -> float:
        return max(0.0, theta_deg) * self.K_AREA

    def get_flow_matrix(
        self,
        fan_rpms: Sequence[float],
        theta_int: Sequence[float],
        theta_ext: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        fan_rpms = np.asarray(fan_rpms)
        theta_int = np.asarray(theta_int)
        if theta_ext is None:
            theta_ext = np.zeros_like(theta_int)

        Q = np.zeros((self.n, self.n))
        for i in range(self.n):
            area = self._slot_area(theta_int[i])
            eff_dp = self.DELTA_P_REF * (fan_rpms[i] / self.MAX_FAN_RPM)
            v_exit = np.sqrt(max(0.0, 2 * eff_dp / RHO_AIR))
            volumetric = self.c_d * area * v_exit        # m³/s

            # 외부 슬롯 영향은 다음 단계 — self_ratio 보정 예정
            self_ratio = self.JET_SELF_RATIO
            Q[i, i] += volumetric * self_ratio
            adj_ratio = (1 - self_ratio) / len(self.adj[i])
            for j in self.adj[i]:
                Q[i, j] += volumetric * adj_ratio

        # 자연혼합 (난류)
        natural = self.NATURAL_MIX_RATE * DEFAULT_ZONE_VOL
        Q += natural * (np.ones((self.n, self.n)) - np.eye(self.n)) / (self.n - 1)
        return Q

# -----------------------------------------------------------------------------
# PhysicsSimulator – 호환 래퍼 포함
# -----------------------------------------------------------------------------
class PhysicsSimulator:
    """물리 시뮬레이터 (Legacy API compatible)"""

    def __init__(self, num_zones: int = 4, zone_volumes: Optional[Sequence[float]] = None):
        self.n = num_zones
        self.zone_volumes = np.asarray([DEFAULT_ZONE_VOL] * num_zones if zone_volumes is None else zone_volumes)
        # 상태 변수 초기화
        self.T = np.random.uniform(22, 28, size=self.n)
        self.H = np.random.uniform(40, 60, size=self.n)
        self.CO2 = np.random.uniform(400, 800, size=self.n)
        self.Dust = np.random.uniform(0, 10, size=self.n)

        self.ambient_temp = AMBIENT_TEMP
        self.ambient_hum = AMBIENT_HUM

        # 서브 모델
        self.jet = JetModel(self.n)
        self.balance = ZoneEnergyBalance(self.zone_volumes)

    # ------------------------------------------------------------------
    # Public helpers (환경에서 호출)
    # ------------------------------------------------------------------
    def reset(self):
        """상태 벡터를 랜덤 초기화"""
        self.__init__(self.n, self.zone_volumes)

    def get_current_state(self) -> Dict[str, np.ndarray]:
        return {
            'temperatures': self.T,
            'humidities': self.H,
            'co2_levels': self.CO2,
            'dust_levels': self.Dust,
        }

    # ------------------------------------------------------------------
    # 내부 유틸 – 펠티어 냉각 분배 계산
    # ------------------------------------------------------------------
    def _distribute_cooling(
        self,
        thermal_power: float,
        cold_side_temp: float,
        intake_temp: float,
        internal_angles: np.ndarray,
        temps: np.ndarray,
        dt: float,
        fan_mass_flow: float,  # kg/s (팬 유량)
    ) -> np.ndarray:
        if thermal_power >= 0:
            return np.zeros(self.n)
        e_removed = -thermal_power * dt                 # J
        denom = CP_AIR * max(intake_temp - cold_side_temp, 1e-3)
        # (b) 냉각가능 질량 = 팬 유량 × dt
        m_cool_cap = fan_mass_flow * dt
        m_cool = min(e_removed / denom, m_cool_cap, self.zone_volumes.sum()*RHO_AIR)                      # kg
        weights = np.clip(internal_angles / 45.0, 0.0, 1.0)
        if weights.sum() == 0:
            return np.zeros(self.n)
        frac = weights / weights.sum()
        q_rates = -(m_cool / dt) * CP_AIR * (temps - cold_side_temp) * frac
        return q_rates

    # ------------------------------------------------------------------
    # Core physics step (new signature)
    # ------------------------------------------------------------------
    def _update_physics_core(
        self,
        temps: np.ndarray,
        humidities: np.ndarray,
        fan_rpms: Sequence[float],
        internal_angles: Sequence[float],
        external_angles: Sequence[float],
        peltier_output: Dict,
        ambient_temp: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fan_rpms = np.asarray(fan_rpms)
        internal_angles = np.asarray(internal_angles)
        external_angles = np.asarray(external_angles)

        # 1) 공기 유량 행렬
        Q_matrix = self.jet.get_flow_matrix(fan_rpms, internal_angles, external_angles)
        # 1.5) 팬 유량 계산 (펠티어 냉각 분배용)
        fan_mass_flow = Q_matrix.sum() * RHO_AIR

        # 2) 펠티어 냉각량 분배
        pelt_rates = self._distribute_cooling(
            thermal_power=peltier_output['thermal_power'],
            cold_side_temp=peltier_output.get('cold_side_temp', temps[0] - 10),  # default fallback
            intake_temp=temps[0],
            internal_angles=internal_angles,
            temps=temps,
            dt=dt,
            fan_mass_flow=fan_mass_flow,
        )

        # 3) 에너지·수분 수지 계산
        new_T, new_H = self.balance.step(temps, humidities, Q_matrix, pelt_rates, ambient_temp, dt)
        return new_T, new_H, Q_matrix

    # ------------------------------------------------------------------
    # Legacy-compatible API (environment.py가 호출)
    # ------------------------------------------------------------------
    def update_physics(
        self,
        action_dict: Dict,
        peltier_states: Dict,
        fan_states: Dict,
        dt: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        # ---- 입력 파싱 ----
        internal_angles = np.array(action_dict['internal_servo_angles'])
        external_angles = np.array(action_dict.get('external_servo_angles', [0.0] * self.n))
        fan_rpms = np.array([f['rpm'] for f in fan_states['small_fans']])
        peltier_output = peltier_states[0]

        # ---- 핵심 업데이트 ----
        self.T, self.H, Q = self._update_physics_core(
            temps=self.T,
            humidities=self.H,
            fan_rpms=fan_rpms,
            internal_angles=internal_angles,
            external_angles=external_angles,
            peltier_output=peltier_output,
            ambient_temp=self.ambient_temp,
            dt=dt,
        )

        # ---- CO₂ & Dust 간단 환기 모델 ----
        decay = np.clip(Q.sum(axis=0) / self.zone_volumes, 0, 0.2)
        self.CO2 = 350 + (self.CO2 - 350) * np.exp(-decay * dt)
        self.Dust = np.maximum(0, self.Dust * np.exp(-decay * dt) + np.random.normal(0, 0.05, size=self.n))

        return self.get_current_state()
