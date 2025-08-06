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
# ---------------------------------------------------------------------------
# ZoneEnergyBalance : 존별 열·수분 수지 (펠티어 배열 지원, 질량·에너지 보존 강화)
# ---------------------------------------------------------------------------
class ZoneEnergyBalance:
    """단순 박스-모델 — 온도 (°C)·상대습도 (%) 업데이트
       · JetModel v2 에 맞춰 질량보존(m_in≈m_out) / 절대습량(kg/kg) 혼합을 사용
    """

    # ------------------------- 상수 / 보조 함수 ---------------------------
    P_ATM   = 101_325.0                     # Pa  (표준 대기)
    RHO_AIR = RHO_AIR                       # kg m⁻³ (전역 사용 값)
    C_P     = CP_AIR                        # J kg⁻¹ K⁻¹

    @staticmethod
    def _pv_sat(T: np.ndarray) -> np.ndarray:
        """포화 수증기압 Pa (Tetens 식, T[°C])"""
        return 610.94 * np.exp(17.625 * T / (T + 243.04))

    @classmethod
    def _Ws(cls, T: np.ndarray) -> np.ndarray:
        """포화 절대습량 kg/kg"""
        return 0.622 * cls._pv_sat(T) / (cls.P_ATM - cls._pv_sat(T))

    # ---------------------------------------------------------------------
    def __init__(self, zone_volumes: Sequence[float], ua_wall: float = 0.30):
        self.V = np.asarray(zone_volumes)           # m³
        self.m = self.RHO_AIR * self.V              # kg 건조공기
        self.C = self.C_P * self.m                  # J K⁻¹
        self.ua_wall = ua_wall                      # W K⁻¹  (실험값 ≈0.3)

    # ---------------------------------------------------------------------
    def step(
        self,
        temps: np.ndarray,             # (N,)  °C
        humidities: np.ndarray,        # (N,)  %RH
        q_matrix: np.ndarray,          # (N,N) m³ s⁻¹
        peltier_rates: np.ndarray,     # (N,) W (음수 = 냉각)
        ambient_temp: float,           # °C
        dt: float = 10.0,              # s
    ) -> tuple[np.ndarray, np.ndarray]:

        # ------ 1) 질량유량 행렬 -----------------------------------------
        m_dot   = q_matrix * self.RHO_AIR           # kg s⁻¹
        m_in    = m_dot.sum(axis=0)                 # kg s⁻¹ (← into zone)
        m_out   = m_dot.sum(axis=1)                 # kg s⁻¹ (→ out of zone)
        m_eff   = np.maximum(m_in, m_out)           # 보수적 – 질량 잔차 0

        # ------ 2) 대류 열수지 ------------------------------------------
        q_conv  = self.C_P * (m_dot.T @ temps) - self.C_P * m_eff * temps  # W

        # ------ 3) 벽 손실 ----------------------------------------------
        q_wall  = -self.ua_wall * (temps - ambient_temp)                  # W

        # ------ 4) 펠티어 ----------------------------------------------
        q_total = q_conv + q_wall + peltier_rates                         # W

        temps_new = temps + (q_total * dt) / self.C                       # °C

        # ================= 습도 처리 – 절대습량 혼합 =====================
        # 현재 RH → 절대습량 W (kg/kg)
        W_now  = humidities / 100.0 * self._Ws(temps)
        # 외기 절대습량 (고정 AMBIENT_HUM [%])
        W_ext  = AMBIENT_HUM / 100.0 * self._Ws(np.array([ambient_temp]))[0]

        # 절대습량 혼합 (질량보존)
        W_new  = (m_dot.T @ W_now + W_ext * m_in) / (m_in + 1e-9)

        # 다시 RH(%) 로 변환 (포화량은 새 온도 기준)
        RH_new = np.clip(W_new / self._Ws(temps_new) * 100.0, 0.0, 100.0)

        return temps_new, RH_new


# -----------------------------------------------------------------------------
# JetModel : 팬 RPM + 내부 슬롯 각도(theta_int) → 존별 체적유량 행렬 (m³/s)
#            - 각 존당 소형팬 2EA (흡기)
#            - 냉각 덕트를 통해 대형팬 1EA가 하부로 토출 후 슬롯 분배
#            - theta_int 만으로 분배 (theta_ext 무시, self‑ratio 없음)
# -----------------------------------------------------------------------------
class JetModel:
    """팬 & 슬롯 기반 체적유량 모델.

    Canonical 호출 방법
    -------------------
    Q = jet.get_flow_matrix(
            fan_rpms_S=[rpm_z0, rpm_z1, rpm_z2, rpm_z3],   # 소형팬 그룹(존)별 평균 RPM
            fan_rpms_L=large_rpm,                          # 대형 토출 팬 RPM
            theta_int=[θ0, θ1, θ2, θ3]                     # 내부 슬롯 각도(0~45°)
        )

    반환값 : (4×4) ndarray — Q[i,j] 는 시간당 zone j → i 로 유입되는 체적유량(m³/s)
    """

    # ------------------------------
    # 초기화
    # ------------------------------
    def __init__(self, num_zones: int = 4, c_d: float = 0.8):
        self.n = num_zones
        self.c_d = c_d

        # --- 팬 파라미터 (데이터시트 기반) ---
        self.MAX_SMALL_FAN_RPM = 7000.0              # 40×10 mm 팬 기준
        self.MAX_LARGE_FAN_RPM = 3300.0              # 120×25 mm 팬 기준
        self.DELTA_P_SMALL = 38.0                    # Pa (3.87 mmH2O) – 소형팬 정압
        self.LARGE_Q_MAX = 0.0382                    # m³/s (81 CFM) – 대형팬 자유공기 유량
        self.SMALL_FANS_PER_ZONE = 2                 # 흡기팬 2EA per zone

        # --- 슬롯 & 혼합 파라미터 ---
        self.K_AREA = 1.3e-4      # m²/deg  (CAD: 120 mm × 25 mm, 0~45°)
        self.NATURAL_MIX_RATE = 0.01  # s⁻¹  (난류에 의한 배경 혼합)

        # 내부 저장용
        self.last_Q_fan: Optional[np.ndarray] = None
        self.last_Q_nat: Optional[np.ndarray] = None

    # ------------------------------
    # 내부 슬롯 개구 면적 (deg → m²)
    # ------------------------------
    def _slot_area(self, theta_deg: float) -> float:
        return max(0.0, theta_deg) * self.K_AREA

    # ------------------------------
    # 메인: 팬 RPM + θ_int → Q 행렬
    # ------------------------------
    def get_flow_matrix(
        self,
        fan_rpms_S: Sequence[float],   # len = n (존별 평균 RPM)
        fan_rpms_L: float,             # 대형 팬 RPM (단일 값)
        theta_int: Sequence[float],    # len = n (0~45°)
        theta_ext: Optional[Sequence[float]] = None,  # **무시**
    ) -> np.ndarray:
        fan_rpms_S = np.asarray(fan_rpms_S, dtype=float)
        theta_int  = np.asarray(theta_int,  dtype=float)

        # --------------------------------------------------------
        # 1) 존별 흡기 유량 (소형팬 2EA, RPM 선형 스케일)
        # --------------------------------------------------------
        q_intake = np.zeros(self.n)
        area_arr = np.zeros(self.n)
        for i in range(self.n):
            area = self._slot_area(theta_int[i])
            area_arr[i] = area
            # 선형 스케일 (RPM / MAX)
            eff_dp = self.DELTA_P_SMALL * (fan_rpms_S[i] / self.MAX_SMALL_FAN_RPM)
            v = np.sqrt(max(0.0, 2 * eff_dp / RHO_AIR))
            q_single = self.c_d * area * v                # 한 개 팬 기준
            q_intake[i] = self.SMALL_FANS_PER_ZONE * q_single

        total_intake = q_intake.sum()

        # --------------------------------------------------------
        # 2) 대형팬 덕트 용량 제한 (자유공기 유량 선형 스케일)
        # --------------------------------------------------------
        large_cap = self.LARGE_Q_MAX * (fan_rpms_L / self.MAX_LARGE_FAN_RPM)
        if total_intake > 0:
            scale = min(1.0, large_cap / total_intake)
        else:
            scale = 0.0
        q_supply = q_intake * scale

        # 팬 공급 행렬 (대각)
        Q_fan = np.diag(q_supply)

        # --------------------------------------------------------
        # 3) 자연 혼합 행렬 (비대각, 고정 계수)
        # --------------------------------------------------------
        if self.n > 1:
            natural = self.NATURAL_MIX_RATE * DEFAULT_ZONE_VOL  # m³/s
            Q_nat = natural * (np.ones((self.n, self.n)) - np.eye(self.n)) / (self.n - 1)
        else:
            Q_nat = np.zeros((1, 1))

        # --------------------------------------------------------
        # 4) 합계 행렬 및 내부 저장
        # --------------------------------------------------------
        Q_total = Q_fan + Q_nat
        self.last_Q_fan = Q_fan
        self.last_Q_nat = Q_nat
        return Q_total


# -----------------------------------------------------------------------------
# PhysicsSimulator – 호환 래퍼 포함
# -----------------------------------------------------------------------------
class PhysicsSimulator:
    """물리 시뮬레이터 (Legacy API compatible)"""

    def __init__(self, num_zones: int = 4, zone_volumes: Optional[Sequence[float]] = None):
        self.n = num_zones
        self.zone_volumes = np.asarray([DEFAULT_ZONE_VOL] * num_zones if zone_volumes is None else zone_volumes)
        # 상태 변수 초기화
        # self.T = np.random.uniform(22, 28, size=self.n)
        # self.H = np.random.uniform(40, 60, size=self.n)
        # self.CO2 = np.random.uniform(400, 800, size=self.n)
        # self.Dust = np.random.uniform(0, 10, size=self.n)

        # 초기 상태 (예시)
        self.T = np.full(self.n, 28.0)  # 초기 온도 (°C)
        self.H = np.full(self.n, 70.0)  # 초기 습도 (%RH)
        self.CO2 = np.full(self.n, 400.0)  # 초기 CO2 농도 (ppm)
        self.Dust = np.full(self.n, 0.0)  # 초기 미세먼지 농도 (μg/m³)

        # Ambient conditions
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
        fan_rpms_S: Sequence[float],
        fan_rpms_L: float,
        internal_angles: Sequence[float],
        external_angles: Sequence[float],
        peltier_output: Dict,
        ambient_temp: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fan_rpms_S = np.asarray(fan_rpms_S)
        internal_angles = np.asarray(internal_angles)
        external_angles = np.asarray(external_angles)

        # 1) 공기 유량 행렬
        Q_matrix = self.jet.get_flow_matrix(fan_rpms_S, fan_rpms_L, internal_angles, external_angles)
        # 1.5) 팬 유량 계산 (펠티어 냉각 분배용)
        fan_mass_flow = np.diag(self.jet.last_Q_fan).sum() * RHO_AIR

        intake_temp = float((self.jet.last_Q_fan.diagonal() @ temps) / (self.jet.last_Q_fan.diagonal().sum()+1e-9))

        # 2) 펠티어 냉각량 분배
        pelt_rates = self._distribute_cooling(
            thermal_power=peltier_output['thermal_power'],
            cold_side_temp=peltier_output.get('cold_side_temp', temps[0] - 10),  # default fallback
            intake_temp=intake_temp,
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
        dt: float = 10.0,
    ) -> Dict[str, np.ndarray]:
        # ---- 입력 파싱 ----
        internal_angles = np.array(action_dict['internal_servo_angles'])
        external_angles = np.array(action_dict.get('external_servo_angles', [0.0] * self.n))
        fan_rpms_S = np.array([f['rpm'] for f in fan_states['small_fans']])
        fan_rpms_L = fan_states['large_fan']['rpm']
        peltier_output = peltier_states[0]

        # ---- 핵심 업데이트 ----
        self.T, self.H, Q = self._update_physics_core(
            temps=self.T,
            humidities=self.H, 
            fan_rpms_S=fan_rpms_S,
            fan_rpms_L=fan_rpms_L,
            internal_angles=internal_angles,
            external_angles=external_angles,
            peltier_output=peltier_output,
            ambient_temp=self.ambient_temp,
            dt=dt,
        )

        # ---- CO₂ & Dust 간단 환기 모델 ----
        # 권장 (팬 배출 유량만 반영):
        fan_out = np.diag(self.jet.last_Q_fan)              # m³/s per zone
        decay   = np.clip(fan_out / self.zone_volumes, 0, 0.2)
        self.CO2 = 350 + (self.CO2 - 350) * np.exp(-decay * dt)
        self.Dust = np.maximum(0, self.Dust * np.exp(-decay * dt) + np.random.normal(0, 0.05, size=self.n))

        return self.get_current_state()
