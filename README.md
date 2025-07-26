# Smart AC Simulator

---

### 파일구조 및 설명

```
smart_ac_simulator/
├── agents/                   # 강화학습 에이전트
│   ├── __init__.py
│   ├── base_agent.py         # 에이전트 추상 베이스 클래스
│   └── sac_agent.py          # SAC 에이전트 구현 (추천)
├── configs/
│   ├── __init__.py
│   ├── hvac_config.py          # 물리·환경 파라미터
│   └── sac_config.py           # SAC 하이퍼파라미터 (분리 권장)
├── data/                     # 학습/평가 결과, 로그 등을 저장
│   ├── logs/                   # TensorBoard, csv, json 등
│   └── plots/                  # .png, .pdf 등
├── docs/                       # 설계 문서, API 문서 (Sphinx/Markdown)
├── evaluator/
│   ├── __init__.py
│   └── evaluate.py             # 평가·벤치마크 모듈화
├── saved_models/             # 학습된 에이전트 모델 가중치를 저장
│   └── sac/                    # 버전·시드별 하위 폴더
├── simulator/                # 시뮬레이터 핵심 로직
│   ├── __init__.py
│   ├── environment.py        # AdvancedSmartACSimulator (Gym 환경 인터페이스)
│   ├── physics.py            # AdvancedPhysicsSimulator (물리 엔진)
│   ├── components.py         # Peltier, Servo, Fan 등 하드웨어 컴포넌트 모델
│   ├── sensors.py            # SensorModel
│   └── utils.py              # ComfortCalculator, 기타 유틸리티 함수
├── tests/                      # PyTest 단위·통합 테스트
├── train.py                  # 학습 실행 스크립트
├── evaluate.py               # 평가 실행 및 시각화 스크립트
├── requirements.txt          # 프로젝트 의존성 패키지 목록
└── README.md                 # 프로젝트 설명서
```

---
