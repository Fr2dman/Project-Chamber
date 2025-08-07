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
from configs.hvac_config import CONTROL_TERM

# -----------------------------------------------------------------------------
# 전역 상수 (문헌 + 실험값)
# -----------------------------------------------------------------------------
CP_AIR = 1005.0          # J/(kg·K)
RHO_AIR = 1.2            # kg/m³
DEFAULT_ZONE_VOL = 0.096 # m³ (가로60×세로40×높이40 cm - 4존 기준)
AMBIENT_TEMP = 30.0      # °C
AMBIENT_HUM = 70.0       # %RH
INFIL_FRAC = 2.0e-3 / 3600    # 0.1 % · h⁻¹  →  s⁻¹ : 시연상자의 틈새 유입률
LATENT_HEAT_VAP = 2.45e6   # J/kg  (물 증발 잠열)
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
        """포화 절대습량 [kg/kg]"""
        p_vs = cls._pv_sat(T)
        return 0.622 * p_vs / (cls.P_ATM - p_vs)

    # ---------------------------------------------------------------------
    def __init__(self, zone_volumes: Sequence[float], ua_wall: float = 0.30):
        self.V = np.asarray(zone_volumes)           # m³
        self.m = self.RHO_AIR * self.V              # kg 건조공기
        self.C = self.C_P * self.m                  # J K⁻¹
        self.ua_wall = ua_wall                      # W K⁻¹  (실험값 ≈0.3)

    # ---------------------------------------------------------------------
    def step(
        self,
        temps: np.ndarray,          # (N,)  °C
        humidities: np.ndarray,     # (N,)  %RH
        q_matrix: np.ndarray,       # (N,N) m³ s⁻¹  (j → i)
        peltier_rates: np.ndarray,  # (N,)   W      (음수 = 냉각)
        ambient_temp: float,        # °C
        w_removed: np.ndarray,      # (N,)   kg/s   (펠티어 표면 응축수)
        dt: float = CONTROL_TERM,   # s
    ) -> tuple[np.ndarray, np.ndarray]:

        # ────────────────────────────────────────────────────────────
        # 1) 질량 유량  (kg/s)
        # ────────────────────────────────────────────────────────────
        m_dot = q_matrix * self.RHO_AIR        # j → i
        m_in  = m_dot.sum(axis=0)              # Σ_j ṁ_ij
        m_out = m_dot.sum(axis=1)              # Σ_j ṁ_ji

        m_infil  = self.RHO_AIR * self.V * INFIL_FRAC   # 틈새 유입 kg/s
        m_in_tot = m_in  + m_infil                      # 총 유입
        # (동일 질량 배출 가정) m_out_tot = m_out + m_infil

        # ────────────────────────────────────────────────────────────
        # 2) 열 수지 (온도 업데이트)
        #    q_conv_i = c_p · Σ_j ṁ_ij · (T_j − T_i)
        # ────────────────────────────────────────────────────────────
        q_conv = self.C_P * (m_dot @ temps - m_out * temps)
        q_wall = -self.ua_wall * (temps - ambient_temp)               # 벽체 열손실
        q_total = q_conv + q_wall + peltier_rates                     # + 펠티어
        temps_new = temps + q_total * dt / self.C                     # ΔU = m·c_p·ΔT

        # ────────────────────────────────────────────────────────────
        # 3) 수분 수지  (절대습량 [kg/kg])
        # ────────────────────────────────────────────────────────────
        # ① step-전 절대습량 (펠티어 코일 응축분 차감)
        W_now = humidities / 100.0 * self._Ws(temps)               # kg/kg
        W_now = np.clip(W_now - w_removed * dt / self.m, 0.0, None)

        # ② 순수 유입-유출·침투로 인한 변화량 (kg/kg·s⁻¹)
        W_ext  = AMBIENT_HUM / 100.0 * self._Ws(np.asarray([ambient_temp]))[0]
        dW_dt  = (m_dot @ W_now - m_out * W_now) / self.m          # (유입 − 유출)
        dW_dt += INFIL_FRAC * (W_ext - W_now)                      # 틈새 침투 보정

        # ③ 예측 절대습량 (응축 전)
        W_pred = np.clip(W_now + dW_dt * dt, 0.0, None)

        # ④ 공기 중 응축 (이슬점 → 포화값 초과분 제거)
        Ws_sat   = self._Ws(temps_new)                             # 포화 절대습량
        cond_dew = np.maximum(0.0, W_pred - Ws_sat)                # kg/kg   (공기응축)

        # ⑤ 총 제거 수분  = 공기응축 + 코일응축
        cond_total = cond_dew + w_removed * dt / self.m
        W_new = np.clip(W_pred - cond_total, 0.0, None)

        # ⑥ 상대습도 재계산
        RH_new = np.clip(W_new / Ws_sat * 100.0, 0.0, 100.0)

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

    # 슬롯 면적 계수 (m²/deg)
    K_AREA_INT = 1.3e-4
    MAX_SMALL_RPM = 7000.0
    MAX_LARGE_RPM = 3300.0
    DELTA_P_SMALL = 38.0  # Pa
    LARGE_Q_MAX = 0.0382  # m³/s
    SMALL_FANS_PER_ZONE = 2
    NATURAL_MIX_RATE = 0.01  # s⁻¹

    def __init__(self, num_zones: int = 4, max_ext_angle: float = 80.0, c_d: float = 0.8):
        self.n = num_zones
        self.c_d = c_d
        self.max_ext_angle = max_ext_angle
        self.last_Q_fan = None
        self.last_Q_nat = None

    def _slot_area_int(self, theta: float) -> float:
        return max(0.0, theta) * self.K_AREA_INT

    def get_flow_matrix(
        self,
        fan_rpms_S: Sequence[float],
        fan_rpms_L: float,
        theta_int: Sequence[float],
        theta_ext: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        fan_rpms_S = np.asarray(fan_rpms_S, dtype=float)
        theta_int = np.asarray(theta_int, dtype=float)
        if theta_ext is None:
            theta_ext = theta_int.copy()
        theta_ext = np.asarray(theta_ext, dtype=float)

        # 1) 존별 흡기 유량 (소형팬)
        q_intake = np.zeros(self.n)
        for i in range(self.n):
            area_int = self._slot_area_int(theta_int[i])
            eff_dp = self.DELTA_P_SMALL * (fan_rpms_S[i] / self.MAX_SMALL_RPM)
            v = np.sqrt(max(0.0, 2 * eff_dp / 1.2))
            q_intake[i] = self.c_d * area_int * v * self.SMALL_FANS_PER_ZONE

        # 2) 대형팬 토출 용량 제한
        total_intake = q_intake.sum()
        large_cap = self.LARGE_Q_MAX * (fan_rpms_L / self.MAX_LARGE_RPM)
        scale = min(1.0, large_cap / total_intake) if total_intake > 0 else 0.0
        q_supply = q_intake * scale

                # 3) 외부 슬롯(theta_ext)에 따른 자연 혼합 조정
        # theta_ext: 0(천장 수평)~max_ext_angle(지면 수직)
        # ext_factor: 각 존의 바람 방향에 따른 직접 전달 비율
        ext_factor = np.clip(theta_ext / self.max_ext_angle, 0.0, 1.0)

        # 4) 팬 공급 행렬 (대각)
        Q_fan = np.diag(q_supply)

        # 5) 자연 혼합 행렬 (기존 Q_nat에서 컬럼별 스케일링)
        if self.n > 1:
            vol = 1.0  # DEFAULT_ZONE_VOL 대체
            natural = self.NATURAL_MIX_RATE * vol
            base_nat = natural * (np.ones((self.n, self.n)) - np.eye(self.n)) / (self.n - 1)
            # JetModel: ext_factor 로 감쇠하되, 행·열 모두 곱해 대칭 유지
            Q_nat = base_nat * np.sqrt((1 - ext_factor)[None, :] * (1 - ext_factor)[:, None])
        else:
            Q_nat = np.zeros((1, 1))

        # 6) 최종 유량 행렬
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

    #    Returns
    #    -------
    #    q_rates : np.ndarray
    #        존별 sensible + latent 냉각량 [W]. (음수 = 냉각)
    #    w_removed : np.ndarray
    #        존별 결로로 제거된 수분 질량유량 [kg/s].
    # ------------------------------------------------------------------
    def _distribute_cooling(
        self,
        thermal_power: float,     # PeltierModel 가 리턴한 `thermal_power` (음수 = 냉각) [W].
        cold_side_temp: float,    # 냉측 면 온도 [°C]
        intake_temp: float,       # 덕트 흡입 공기 온도 [°C].  (팬 유량 가중 평균).
        internal_angles: np.ndarray, # 내부 슬롯 각도 0–45°.  각도 비례로 냉기가 분배됨.
        temps: np.ndarray,        # 존별 현재 온도 [°C]
        abs_hum: np.ndarray,      # (N,) zone 절대습량 kg/kg (Step 직전 값)
        fan_mass_flow: float,     # kg/s (팬 유량)
        dt: float,
    ) -> np.ndarray:
        n = temps.size
        if thermal_power >= 0:
            return np.zeros(n), np.zeros(n)
        
        # 1) 이번 step 동안 제거 가능한 냉각 공기 mass (kg)
        e_removed = -thermal_power * dt                                  # J
        denom = CP_AIR * max(intake_temp - cold_side_temp, 1e-3)
        m_cool_cap = fan_mass_flow * dt                                  # kg
        m_cool = min(e_removed/denom, m_cool_cap)      # kg

        # print(f"펠티어 냉각량: {thermal_power:.2f} W, 제거 가능 질량: {m_cool:.2f} kg")

        # 2) 가중치 (내부 슬롯 각도 × 팬 유량 동일 분배假)
        weights = np.clip(internal_angles/45.0, 0.0, 1.0)
        if weights.sum() == 0:
            return np.zeros(n), np.zeros(n)
        frac = weights/weights.sum()

        # 3) sensible 냉각분배 (기존)
        q_sensible = -(m_cool / dt) * CP_AIR * (temps - cold_side_temp) * frac  # W

        w_removed = np.zeros(n)  # 각 존별 결로로 제거된 수분 (kg/s)
        # 4) 결로량 계산 (latent)
        #   - 각 존별 냉각된 공기 온도 계산 후, 절대습량 계산
        #   - 냉각된 공기에서 제거된 수분량 계산
        #   - m_cool × frac[i] 만큼 냉각된 공기에서 제거됨
        for i in range(n):
            # 각 존별로 냉각된 공기 온도 계산
            cooled_air_temp = max(cold_side_temp, temps[i] - 5.0)  # 최대 5도 냉각
            Ws_cool = ZoneEnergyBalance._Ws(np.array([cooled_air_temp]))[0]
            w_removed[i] = max(0.0, abs_hum[i] - Ws_cool) * (m_cool * frac[i]) / dt

        q_latent = -w_removed * LATENT_HEAT_VAP                             # W (음수)

        q_rates = q_sensible + q_latent
        return q_rates, w_removed

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

        abs_hum = self.H/100.0 * ZoneEnergyBalance._Ws(self.T)
        # 2) 펠티어 냉각량 분배
        pelt_rates, w_removed = self._distribute_cooling(
            peltier_output['thermal_power'],
            peltier_output.get('cold_side_temp', self.T.min()-10.0),
            intake_temp, internal_angles, self.T, abs_hum, fan_mass_flow, dt)

        # 3) 에너지·수분 수지 계산
        new_T, new_H = self.balance.step(temps, humidities, Q_matrix, pelt_rates, ambient_temp, w_removed, dt)
        self.T, self.H = new_T, new_H

        return new_T, new_H, Q_matrix

    # ------------------------------------------------------------------
    # Legacy-compatible API (environment.py가 호출)
    # ------------------------------------------------------------------
    def update_physics(
        self,
        action_dict: Dict,
        peltier_states: Dict,
        fan_states: Dict,
        dt: float = CONTROL_TERM,
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
