# ------------------------------------------
# file: beta_store.py
# ------------------------------------------
"""β(베타) 계수를 영속적으로 저장·복원하기 위한 모듈.

- JSON 파일(`beta_store.json`)에 존(zone)별 β 값을 기록합니다.
- 서버 재시작 후에도 온라인 학습 결과가 유지됩니다.
- 스레드 환경에서 안전하도록 `RLock`으로 보호합니다.

다중 *프로세스* 환경(예: Gunicorn + Uvicorn)에서는 JSON 대신
Redis/SQLite/PostgreSQL 등 원자적 연산을 지원하는 저장소로 교체하세요.

변경 사항 (v1.1)
----------------
* **NaN 안전장치** : β 값이 NaN 이거나 실수 범위를 벗어날 경우 저장을 무시합니다.
"""
from __future__ import annotations

import atexit
import json
import math
import pathlib
import threading
from typing import Dict


class BetaStore:
    """존별 β 값을 로드·저장·조회하는 싱글턴 헬퍼 클래스."""

    _path = pathlib.Path("beta_store.json")  # 저장 파일 경로
    _lock = threading.RLock()                # 스레드 동기화용 락
    _data: Dict[str, float] = {}             # zone_id -> β

    # ---------- 파일 IO ----------
    @classmethod
    def _load(cls) -> None:
        """디스크에서 β 값을 불러옵니다(없으면 빈 dict)."""
        if cls._path.exists():
            try:
                with cls._path.open("r", encoding="utf-8") as f:
                    cls._data = json.load(f)
            except Exception:
                # 파싱 오류 시 새로 시작 (운영 환경에선 로그 권장)
                cls._data = {}

    @classmethod
    def _save(cls) -> None:
        with cls._lock:                   # <- 추가: atexit 경쟁 방지
            tmp = cls._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(cls._data, f, ensure_ascii=False, indent=2)
            tmp.replace(cls._path)        # POSIX 환경에서 원자적 rename

    # ---------- 퍼블릭 API ----------
    @classmethod
    def get(cls, zone_id: str, default: float = 0.4) -> float:
        """존 ID에 해당하는 β를 반환(없으면 default)."""
        val = cls._data.get(zone_id, default)
        return default if (val is None or math.isnan(val)) else val

    @classmethod
    def set(cls, zone_id: str, value: float) -> None:
        """β 값을 저장(단, NaN·비실수는 무시) 후 디스크 반영."""
        if value is None or math.isnan(value) or not math.isfinite(value):
            return  # NaN/Inf 저장 방지
        with cls._lock:
            cls._data[zone_id] = value
            cls._save()


# 모듈 import 시 자동 로드, 프로세스 종료 시 자동 저장
BetaStore._load()
atexit.register(BetaStore._save)