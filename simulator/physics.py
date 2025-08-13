# simulator/physics.pys
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
INFIL_FRAC = 1.0e-5 / 3600    # 0.1 % · h⁻¹  →  s⁻¹ : 시연상자의 틈새 유입률
EXHAUST_FRAC = 0.05           # 대형팬 공급 유량 중 실외 배기 비율(보수적 가정)
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
        q_conv  = self.C_P * (m_dot @ temps - m_out * temps)
        q_wall  = -self.ua_wall * (temps - ambient_temp)              # 벽체 열손실
        q_infil = self.C_P * (self.RHO_AIR * self.V * INFIL_FRAC) * (ambient_temp - temps)  # 침투열
        q_total = q_conv + q_wall + q_infil + peltier_rates           # + 펠티어 + 침투열
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

        Ws_sat   = self._Ws(temps_new)                             # 포화 절대습량(응축 전 온도)
        cond_dew = np.maximum(0.0, W_pred - Ws_sat)                # kg/kg   (공간 내 응축)

        # ⑤ 공간 응축의 잠열을 에너지식에 반영하고 온도 재계산
        if np.any(cond_dew > 0):
            q_lat_space = -cond_dew * self.m * LATENT_HEAT_VAP / dt   # W
            temps_new = temps + (q_total + q_lat_space) * dt / self.C
            Ws_sat = self._Ws(temps_new)                              # 새 온도 기준 포화량 갱신

        # ⑥ 총 제거 수분  = 공간응축 + 코일응축
        cond_total = cond_dew + w_removed * dt / self.m
        W_new = np.clip(W_pred - cond_total, 0.0, None)

        # ⑦ 상대습도 재계산
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

    물리적 모델:
    - 소형팬: 각 영역 상부에서 흡기 (슬롯 각도와 무관)
    - 대형팬: 냉각된 공기를 토출
    - 내부 슬롯: 각 영역으로의 토출 분배량 제어
    - 외부 슬롯: 토출 방향 제어
        * 0도(수평): 천장 따라 흐름 → 단순 순환(short-circuit)
        * 80도(수직): 중심부 토출 → 전체 혼합
    """

    # 하드웨어 사양
    K_AREA_INT = 1.3e-4    # 내부 슬롯 면적 계수 (m²/deg)
    MAX_SMALL_RPM = 7000.0
    MAX_LARGE_RPM = 3300.0
    SMALL_Q_MAX = 0.01     # m³/s per fan
    LARGE_Q_MAX = 0.0382   # m³/s
    SMALL_FANS_PER_ZONE = 2
    
    # 혼합 파라미터
    NATURAL_MIX_RATE = 0.02    # s⁻¹ (기본 자연 혼합)
    SHORT_CIRCUIT_RATE = 0.8   # 단순 순환 시 재흡기 비율
    MIXING_ENHANCEMENT = 3.0    # 수직 토출 시 혼합 증대 계수

    RECIRC_MAX = 0.60          # 수평(0°) 부근 최대 재흡기 상한
    RECIRC_MIN = 0.05          # 수직(80°) 부근 최소 재흡기 하한
    MIX_SELF_MIN = 0.03       # 아무리 섞여도 자기 존 직류가 최소 30%는 남도록

    # --- 혼합 분포(각도→가중치) 파라미터 ---
    MIX_SPREAD_GAIN = 1.0       # α = sin^2(θ)*gain (0=수평, 1=수직)
    SELF_LOCAL_MIN = 0.05       # 수직 근처(전역화)에서 로컬 커널의 '자기' 최소 몫
    SELF_LOCAL_MAX = 0.85       # 수평 근처에서 로컬 커널의 '자기' 최대 몫
    SELF_LOCAL_SHARPNESS = 1.3  # (1-α)^γ 곡률 (1=선형, ↑일수록 수평에서 자기 강조)
    ALPHA_LOCAL_HARD = 0.02     # 매우 좁은 각도: 대각 완전 차단(α→0)

    def __init__(self, num_zones: int = 4, c_d: float = 0.8):
        self.n = num_zones
        self.c_d = c_d
        self.last_Q_intake = None
        self.last_Q_supply = None
        self.last_Q_matrix = None
        self.last_recirculation_ratio = None

    def _slot_area_int(self, theta: float) -> float:
        return max(0.0, theta) * self.K_AREA_INT
    
    def get_flow_matrix(
        self,
        fan_rpms_S: np.ndarray,  # 소형팬 RPM (각 존)
        fan_rpms_L: float,       # 대형팬 RPM
        theta_int: np.ndarray,   # 내부 슬롯 각도 (0-45°)
        theta_ext: np.ndarray,   # 외부 슬롯 각도 (0-80°)
        dt: float = 1.0,         # 시간 간격
    ) -> tuple[np.ndarray, dict]:
        """
        유량 행렬 계산
        
        Returns:
            Q_matrix: (N×N) 존간 유량 행렬 [m³/s]
                    Q[i,j] = j→i 유량
            info: 디버깅/분석용 추가 정보
        """
        # --- 입력 타입 강제 ---
        fan_rpms_S = np.asarray(fan_rpms_S, dtype=float)
        theta_int = np.asarray(theta_int, dtype=float)
        theta_ext = np.asarray(theta_ext, dtype=float)

        # ========================================
        # 1) 흡기 단계: 소형팬에 의한 각 존별 흡기
        # ========================================
        fan_ratio = fan_rpms_S / self.MAX_SMALL_RPM
        q_intake = self.SMALL_Q_MAX * fan_ratio * self.SMALL_FANS_PER_ZONE
        total_intake = q_intake.sum()

        # ========================================
        # 2) 토출 용량: 대형팬 제약
        # ========================================
        large_capacity = self.LARGE_Q_MAX * (fan_rpms_L / self.MAX_LARGE_RPM)
        actual_throughput = min(total_intake, large_capacity) if total_intake > 0 else 0.0

        # ========================================
        # 3) 내부 슬롯에 의한 분배
        # ========================================
        slot_areas = np.maximum(theta_int * self.K_AREA_INT, 1e-6)

        # 외부 슬롯 완전 닫힘 가정(-5° 이하) 시 해당 존 토출 0
        is_closed = theta_ext < -5
        slot_areas = np.where(is_closed, 0.0, slot_areas)

        if slot_areas.sum() > 0:
            distribution_ratio = slot_areas / slot_areas.sum()
        else:
            distribution_ratio = np.zeros(self.n)

        q_supply = actual_throughput * distribution_ratio  # 슬롯으로 각 존에 공급되는 양(열 기준 총합 = actual_throughput)

        # ========================================
        # 4) 외부 슬롯 각도에 따른 재흡기 + 혼합
        # ========================================
        # 4-1) 재흡기 비율 (0° 수평 클수록 ↑, 80° 수직일수록 ↓)
        angle_rad = np.deg2rad(theta_ext)
        recirc_raw = self.SHORT_CIRCUIT_RATE * (np.cos(angle_rad) ** 2)
        recirc_ratio = np.clip(recirc_raw, self.RECIRC_MIN, self.RECIRC_MAX)
        recirc_ratio = np.where(q_supply > 0, recirc_ratio, 0.0)

        # 재흡기 제외 후 실내로 나가는 양
        q_to_room = q_supply * (1.0 - recirc_ratio)

        # 4-2) 혼합의 '양' (얼마나 섞을지): 0°≈0, 80°≈1
        mixing_raw = (np.sin(angle_rad) ** 2) * self.MIXING_ENHANCEMENT
        mixing_factor = np.clip(mixing_raw, 0.0, 1.0 - self.MIX_SELF_MIN)

        # 자기 직류 + 혼합 몫 분리
        q_self_direct = (1.0 - mixing_factor) * q_to_room
        q_mix_out     = mixing_factor * q_to_room
        # 대각: 자기 존 직류
        Q_direct = np.diag(q_self_direct)

        # # 2×2 레이아웃 인접행렬(상하좌우)
        if self.n == 4:
            neighbors_map = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        else:
            neighbors_map = {i: [j for j in range(self.n) if j != i] for i in range(self.n)}

        # 4-3) 혼합의 '분포' (어디로 섞을지)
        # α=0(수평)→로컬(자기+인접), α=1(수직)→전역 균등(대각 포함)
        alpha = np.clip((np.sin(angle_rad) ** 2) * self.MIX_SPREAD_GAIN, 0.0, 1.0)
        alpha = np.where(alpha < self.ALPHA_LOCAL_HARD, 0.0, alpha)  # 매우 좁은 각도는 하드 로컬

        def _self_share_local(a: float) -> float:
            x = (1.0 - a) ** self.SELF_LOCAL_SHARPNESS
            return float(self.SELF_LOCAL_MIN + (self.SELF_LOCAL_MAX - self.SELF_LOCAL_MIN) * x)

        K_uniform = np.full(self.n, 1.0 / self.n)
        Q_mix = np.zeros((self.n, self.n))
        mix_weights = np.zeros((self.n, self.n))

        for j in range(self.n):      # 출발 존(열)
            mixed_amount = q_mix_out[j]
            if mixed_amount <= 0.0:
                continue

            # 로컬 커널: 자기 + 인접(대각 제외), 각도 좁을수록 '자기' 가중↑
            w_local = np.zeros(self.n)
            adj = neighbors_map.get(j, [])
            keep_self = _self_share_local(alpha[j])
            w_local[j] = keep_self
            rem = max(1.0 - keep_self, 0.0)
            share = rem / len(adj) if len(adj) else 0.0
            for a in adj:
                w_local[a] = share

            # 분포 보간: (1-α)*로컬 + α*전역 균등
            w = (1.0 - alpha[j]) * w_local + alpha[j] * K_uniform
            s = w.sum()
            w = K_uniform.copy() if s <= 0 else (w / s)  # 수치 안정화

            Q_mix[:, j] = mixed_amount * w
            mix_weights[:, j] = w

        # 재흡기 행렬(참고용; 실내 유동 합산에서는 제외)
        Q_recirc = np.diag(q_supply * recirc_ratio)

        # ========================================
        # 5) 자연 혼합 (확산) — 내부 교환
        # ========================================
        vol = DEFAULT_ZONE_VOL  # m³
        natural = self.NATURAL_MIX_RATE * vol
        if self.n > 1:
            Q_natural = natural * (np.ones((self.n, self.n)) - np.eye(self.n)) / (self.n - 1)
        else:
            Q_natural = np.zeros((1, 1))

        # ========================================
        # 6) 최종 유량 행렬
        # ========================================
        # 강제유동(팬 기인)만 먼저 합산
        Q_forced = Q_direct + Q_mix          # (실내 유동) = 자기 직류 + 혼합
        # 최종(내부 확산 추가)
        Q_total = Q_forced + Q_natural

        # 질량 보존 검증 (열 기준: 강제유동 열합 == q_to_room)
        col_err = Q_forced.sum(axis=0) - q_to_room
        mass_balance_error = float(np.max(np.abs(col_err)))
        if mass_balance_error > 1e-6:
            print(f"[warn] forced_col_max_err = {mass_balance_error:.6g}")

        # ========================================
        # 7) 상태 저장 및 반환
        # ========================================
        self.last_Q_intake = q_intake
        self.last_Q_supply = q_supply
        self.last_Q_matrix = Q_total
        self.last_recirculation_ratio = recirc_ratio

        info = {
            'q_intake': q_intake,
            'q_supply': q_supply,
            'q_to_room': q_to_room,
            'q_self_direct': q_self_direct,
            'q_mix_out': q_mix_out,
            'recirculation_ratio': recirc_ratio,
            'mixing_factor': mixing_factor,
            'Q_direct': Q_direct,
            'Q_recirc': Q_recirc,
            'Q_mix': Q_mix,
            'mix_spread_alpha': alpha,
            'mix_weights': mix_weights,
            'Q_natural': Q_natural,
            'Q_forced': Q_forced,
            'mass_balance_error': mass_balance_error,
            'actual_throughput': actual_throughput,
        }

        return Q_total, info


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
        self._last_q_removed = 0.0   # [W] 지난 step에서 존에서 제거된 총열량(감열+잠열)

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
        fan_flows_zone: Optional[np.ndarray] = None,  # (N,) m³/s – 가중치에 사용
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

        # 2) 가중치: 유량×각도 기반 (더 물리적)
        if fan_flows_zone is None:
            weights = np.clip(internal_angles/45.0, 0.0, 1.0)
        else:
            weights = np.maximum(0.0, fan_flows_zone) * np.clip(internal_angles/45.0, 0.0, 1.0)
        if weights.sum() == 0:
            return np.zeros(n), np.zeros(n)
        frac = weights/weights.sum()

        # --- ADP + CBF 코일 출구 상태 (공냉 소형: approach 2~3 K, CBF 0.7~0.9 권장) ---
        ADP = cold_side_temp + 3.0
        CBF = 0.8
        Ws_adp = ZoneEnergyBalance._Ws(np.array([ADP]))[0]
        T_out = ADP + CBF * (temps - ADP)             # 감열 출구 추정
        # 제습 조건: 입구의 이슬점 > ADP  ⇔  abs_hum > Ws(ADP)
        needs_latent = (abs_hum > Ws_adp)
        # Bypass factor model: W_out = Ws_adp + CBF * (W_in - Ws_adp), clip to [Ws_adp, W_in]
        W_out = np.where(needs_latent,
                         np.clip(Ws_adp + CBF * (abs_hum - Ws_adp), Ws_adp, abs_hum),
                         abs_hum)
        
        # --- 단위 질량당 에너지 ---
        sensible_perkg = CP_AIR * (temps - T_out)                               # J/kg
        latent_perkg   = LATENT_HEAT_VAP * np.maximum(0.0, abs_hum - W_out)     # J/kg
        e_perkg = sensible_perkg + latent_perkg                                   # J/kg
        # 냉각 불가(가중 평균 에너지 ≤ 0)이면 바로 0 반환
        E_unit = float(np.dot(frac, e_perkg))
        if E_unit <= 0.0:
            return np.zeros(n), np.zeros(n)

        # --- 질량 한도: (1) thermal_power 목표, (2) 팬 유량 ---
        E_target = -thermal_power * dt                                          # J
        m_energy = E_target / E_unit + 1e-9                                 # kg
        m_cool_cap = fan_mass_flow * dt                                         # kg
        m_cool = max(0.0, min(m_energy, m_cool_cap))
        m_i = m_cool * frac                                                     # kg/zone

        # --- 존별 부하 계산 ---
        q_sensible = -(m_i/dt) * sensible_perkg                                 # W
        # print(f"펠티어 냉각량: {thermal_power:.2f} W, 분배된 질량: {m_i}, 감열 냉각량: {q_sensible}")
        w_removed  =  (m_i/dt) * np.maximum(0.0, abs_hum - W_out)               # kg/s
        q_latent   = -w_removed * LATENT_HEAT_VAP                                # W
        q_rates    = q_sensible + q_latent

        # --- 에너지 보존 정규화 (감열+잠열 ≤ thermal_power) ---
        Q_calc = q_rates.sum() * dt
        # print(f"펠티어 냉각량: {thermal_power:.2f} W, 계산된 총 냉각량: {Q_calc:.2f} J")
        scale = 1.0 if Q_calc <= 0 else min(1.0, E_target / (abs(Q_calc) + 1e-9))
        q_rates  *= scale
        w_removed *= scale
        # print(f"펠티어 냉각량 분배: {q_rates}, 제거된 수분: {w_removed} kg/s")
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
        Q_matrix, q_info = self.jet.get_flow_matrix(fan_rpms_S, fan_rpms_L, internal_angles, external_angles)
        # 1.5) 팬 유량 (펠티어 냉각 분배용)
        fan_flows_zone = self.jet.last_Q_intake                      # (N,) m³/s
        actual_throughput = float(q_info.get('actual_throughput', fan_flows_zone.sum()))
        fan_mass_flow  = actual_throughput * RHO_AIR                  # kg/s
        intake_temp = float(np.dot(fan_flows_zone, temps) / (fan_flows_zone.sum() + 1e-9))

        abs_hum = humidities/100.0 * ZoneEnergyBalance._Ws(temps)

        # print("펠티어 출력: ", peltier_output['thermal_power'], "W")
        # 2) 펠티어 냉각량 분배
        pelt_rates, w_removed = self._distribute_cooling(
            peltier_output['thermal_power'],
            peltier_output.get('cold_side_temp', self.T.min()-10.0),
            intake_temp, internal_angles, self.T, abs_hum, fan_mass_flow, dt,
            fan_flows_zone=fan_flows_zone)

        # 3) 에너지·수분 수지 계산
        new_T, new_H = self.balance.step(temps, humidities, Q_matrix, pelt_rates, ambient_temp, w_removed, dt)
        # 이번 step에 실제로 뺀 총부하(양수 W) 저장 → 다음 step에서 Peltier에 전달
        self._last_q_removed = float(-pelt_rates.sum())

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
        internal_angles = np.array(action_dict.get('internal_servo_angles', [-1.0] * self.n))
        external_angles = np.array(action_dict.get('external_servo_angles', [-1.0] * self.n))
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
        # 실외 배기 비율(EXHAUST_FRAC) + 침투(INFIL_FRAC)만 환기에 기여 (재순환 제외)
        fan_out = EXHAUST_FRAC * self.jet.last_Q_supply
        decay_vol = np.clip(fan_out / self.zone_volumes, 0, 0.2)
        decay = decay_vol + INFIL_FRAC
        self.CO2 = 350 + (self.CO2 - 350) * np.exp(-decay * dt)
        self.Dust = np.maximum(0, self.Dust * np.exp(-decay * dt) + np.random.normal(0, 0.05, size=self.n))

        return self.get_current_state()
