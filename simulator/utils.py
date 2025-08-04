
# ------------------------------------------
# file: zone_comfort.py
# ------------------------------------------
"""존 단위 열쾌적도(PMV/PPD) 계산 + 실시간 β 보정 모듈.

주요 특징
---------
1. **pythermalcomfort** 라이브러리의 ISO PMV 모델을 사용.
2. 재실자 피드백(TSV)을 입력하면 β 계수를 온라인으로 학습하여
   PMV와 체감 온열감의 차이를 줄입니다.
3. β 값은 `BetaStore`에 저장되어 재시작 후에도 유지됩니다.

사용 예시
---------
>>> from zone_comfort import ZoneComfortCalculator
>>> zc = ZoneComfortCalculator("ZONE_A")
>>> zc.calculate_comfort(temp=25, rh=50, v=0.2, direction="direct", tsv=+1)
{"comfort_score": 82.1, "pmv": 0.38, "ppd": 17.9, "v_eff": 0.20, "beta": 0.45}


변경 사항 (v1.1)
----------------
* **극단 입력·NaN 방어** : 입력 범위 검증, PMV/PPD NaN 발생 시 graceful fallback.
* β 업데이트 시 NaN 발생을 차단(저장도 방지).

변경 사항 (v1.2)
----------------
* **Fallback Comfort** : `pythermalcomfort` 계산이 실패하거나 NaN 발생 시
  간단한 휴리스틱(온도·습도 기반)으로 *대체 쾌적도*를 산출해 항상 값이
  반환되도록 수정.
* β 학습은 *정상 PMV 계산*이 있을 때만 수행.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Optional, TypedDict

import numpy as np
from pythermalcomfort.models import pmv_ppd_iso  # pip install pythermalcomfort

from .beta_store import BetaStore
import logging


class ComfortResult(TypedDict):
    comfort_score: float  # 0–100 (높을수록 쾌적)
    pmv: float            # 보정 후 PMV (NaN 가능)
    ppd: float            # 예측 불쾌 비율 (NaN 가능)
    v_eff: float          # 유효 풍속(m/s)
    beta: float           # 현재 β 값
    fallback_used: bool   # True면 휴리스틱 대체값 사용


@dataclass
class ZoneComfortCalculator:
    zone_id: str
    beta: float = field(init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    # --------- 튜닝 파라미터 ---------
    DIR_FACTOR = {
        "direct": 1.0,
        "ceiling_down": 0.8,
        "ceiling_up": 0.4,
    }
    ETA = 0.05
    BETA_MIN = 0.1
    BETA_MAX = 0.8

    # 허용 범위 (ISO + 약간 확장)
    TEMP_RANGE = (10.0, 40.0)   # °C
    RH_RANGE = (1.0, 100.0)      # %
    V_RANGE = (0.0, 5.0)         # m/s

    def __post_init__(self) -> None:
        self.beta = float(BetaStore.get(self.zone_id))

    @staticmethod
    def _check_range(val: float, lo: float, hi: float, name: str) -> None:
        if not lo <= val <= hi:
            raise ValueError(f"{name} 값 {val} 가 허용 범위 {lo}–{hi} 를 벗어났습니다.")

    # --------- 휴리스틱 대체 쾌적도 ---------
    @staticmethod
    def _fallback_comfort(temp: float, rh: float) -> float:
        """온도·습도 기반 간단 점수 (0–100). 중심 목표 24 °C·50 % RH."""
        # 온도 편차 penalty (1 °C당 4점)
        temp_pen = 4.0 * abs(temp - 24.0)
        # RH penalty : 30–70 % 구간을 허용, 바깥은 1 %당 0.5점
        if rh < 30:
            rh_pen = 0.5 * (30 - rh)
        elif rh > 70:
            rh_pen = 0.5 * (rh - 70)
        else:
            rh_pen = 0.0
        score = max(0.0, 100.0 - temp_pen - rh_pen)
        return score

    # --------- 메인 API ---------
    def calculate_comfort(
        self,
        *,
        temp: float,
        rh: float,
        v: float,
        direction: str = "direct",
        tsv: Optional[float] = None,
        met: float = 1.2,
        clo: float = 0.6,
    ) -> ComfortResult:
        # 1) 입력 검증
        try:
            self._check_range(temp, *self.TEMP_RANGE, "온도")
            self._check_range(rh,   *self.RH_RANGE,   "상대 습도")
            self._check_range(v,    *self.V_RANGE,    "풍속")
        except ValueError as e:
            logging.error("Comfort input error: %s", e)
            raise
        if tsv is not None and not -3 <= tsv <= 3:
            raise ValueError("TSV는 -3에서 +3 사이여야 합니다.")

        # 2) 유효 풍속
        v_eff = v * self.DIR_FACTOR.get(direction, 1.0)

        # 3) PMV/PPD 계산 시도
        pmv_raw = ppd = math.nan
        use_fallback = False
        try:
            res = pmv_ppd_iso(tdb=temp, tr=temp, rh=rh, vr=v_eff, met=met, clo=clo)
            pmv_raw, ppd = res["pmv"], res["ppd"]
            if any(math.isnan(x) or not math.isfinite(x) for x in (pmv_raw, ppd)):
                raise ValueError("NaN from PMV")
        except Exception:
            # ISO 계산 실패 → 휴리스틱으로 대체
            use_fallback = True
            comfort_score = self._fallback_comfort(temp, rh)
        else:
            # 4) β 적용 및 학습 (정상 계산일 때만)
            pmv_adj = pmv_raw
            if tsv is not None:
                pmv_adj += self.beta * tsv
                delta_beta = self.ETA * (tsv - pmv_raw)
                new_beta = self.beta + delta_beta
                if math.isfinite(new_beta):
                    new_beta = float(np.clip(new_beta, self.BETA_MIN, self.BETA_MAX))
                    with self._lock:
                        self.beta = new_beta
                        BetaStore.set(self.zone_id, self.beta)
            comfort_score = 100.0 - ppd
            pmv_raw = pmv_adj

        return ComfortResult(
            comfort_score=round(comfort_score, 1),
            pmv=round(pmv_raw, 2) if math.isfinite(pmv_raw) else float('nan'),
            ppd=round(ppd, 1) if math.isfinite(ppd) else float('nan'),
            v_eff=round(v_eff, 2),
            beta=round(self.beta, 2),
            fallback_used=use_fallback,
        )
