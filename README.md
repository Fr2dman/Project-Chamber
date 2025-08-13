# Smart AC Simulator & SAC Agent Generator

멀티존(4존) 실내 환경을 모사하는 경량 HVAC 시뮬레이터와 이를 이용한 SAC(Soft Actor-Critic) 학습 파이프라인입니다. 팬/슬롯/펠티어를 통한 냉각·제습, 유량 분배와 혼합, 에너지·쾌적도 기반 보상 설계를 포함하며 Gymnasium 호환 환경과 SB3 학습 스크립트를 제공합니다.

## 주요기능

- 멀티존 물리엔진: 존 에너지/수분 수지, 제트 혼합, 미소 침기/누설, 이슬점·응축 계산.

- 액추에이터 모델링: 소형/대형 팬 RPM, 내부/외부 슬롯 각도, 펠티어(냉각/제습) 모델.

- 센서 모델: 온/습도 등 센서 값과 노이즈 모델링.

- 보상 설계: 목표 접근(온도/습도/TSV), 에너지 사용(Wh), 제어 스무딩, 안전 제약 패널티.

- Gym 환경: reset/step 구현, 관측/보상/종료 조건 제공.

- SB3 파이프라인: SAC 학습 스크립트, 콜백(평가/모델저장), 래퍼(정규화·게이팅 등).

- 테스트 스위트: 단일/시나리오 테스트 및 보상 불변성 점검 스크립트.

- 설정 분리: configs/의 YAML과 파이썬 설정을 통해 실험·환경·보상 파라미터 분리(예시 제공).

## 파일구조 및 설명

```
checkpoints/sb3_sac/            # SB3 학습 산출물(모델 zip, 리플레이 버퍼 pkl, 평가 로그 등)
configs/                         # 실험/환경/보상/RL 설정
  ├─ env.yaml                   # 환경(초기조건, 시간스텝, 존/팬/서보 한계 등) 예시 설정
  ├─ hvac_config.py             # 기본 HVAC 파라미터(상수/단위/제한치) 정의
  ├─ reward.yaml                # 보상 가중치·게이트 등 예시 설정
  └─ sb3_sac.yaml               # SB3(SAC) 하이퍼파라미터/런 설정 예시
rl/                              # RL 학습 파이프라인
  └─ sb3/
     ├─ callbacks.py            # 평가/체크포인트/얼리스톱 등 콜백 모음
     ├─ make_env.py             # 시뮬레이터 인스턴스/벡터라이즈/정규화 구성
     ├─ train_sac_sb3.py        # 학습 엔트리포인트(SB3 SAC)
     └─ wrappers.py             # 관측/보상/액션 정규화 및 실험용 래퍼
scripts/                         # (옵션) 실행/도구 스크립트 폴더(런처, 변환 등 배치 용도)
sim_test/                        # 시뮬레이터 단독 실행 및 유닛 수준 테스트 스크립트
  ├─ simple_test.py             # 1스텝/단순 동작 점검
  ├─ simple_test_with_reward.py # 보상 계산 포함 점검
  ├─ reward_smoke_test.py       # 보상 구성 스모크 테스트
  ├─ test_simulator_by_scenario.py  # 시나리오별 동작 확인
  ├─ test_step_by_step.py       # 물리량 단계별 추적
  ├─ unit_test_jet_model.py     # 제트/분배 로직 점검
  ├─ unit_test_phy.py           # 물리엔진 단위 테스트
  ├─ unit_test_reward_sanity_check.py # 보상 무결성/범위 점검
  ├─ unit_test_simulator.py     # 환경 래퍼 포함 통합 테스트(1)
  ├─ unit_test_simulator2.py    # 통합 테스트(2)
  └─ unit_test_simulator3.py    # 통합 테스트(3)
simulator/                       # 물리엔진 및 구성 요소 구현
  ├─ beta_store.json            # (베타) 임시 파라미터 저장용 샘플
  ├─ beta_store.py              # (베타) 파라미터 I/O 유틸
  ├─ components.py              # 팬/서보/펠티어 등 액추에이터 모델과 소비전력 산출
  ├─ environment.py             # Gym 호환 환경 래퍼(관측/보상/종료/안전 제약)
  ├─ physics.py                 # 핵심 물리: 에너지·수분 수지, 혼합/침기, 응축 등
  ├─ sensors.py                 # 센서 및 노이즈 모델
  └─ utils.py                   # 공통 유틸(PMV(온열쾌적도) 계산)
tests/                           # 파이프라인/환경 보장 테스트(예: API/보상 불변성)
  ├─ test_env_gym_api.py        # Gym API 준수 테스트
  └─ test_reward_invariants.py  # 보상 불변성/경계값 테스트
README.md                        # 프로젝트 개요(현재 문서)
requirements.txt                 # 의존성 목록
```

---

## 디렉토리 사용 안내

- checkpoints/

        학습 산출물 보관 폴더입니다. 대용량(*.zip, *.pkl, *.npz)은 Git LFS 사용을 권장합니다.

        실험별 하위 폴더를 두어 버전 관리(예: sb3_sac/run_YYYYMMDD/).

- configs/

        env.yaml: 초기 조건, 시뮬레이션 시간 해상도, 제어 한계치를 실험별로 분리합니다.

        reward.yaml: 보상 가중치/게이트/안전 페널티를 외부화합니다(예시).

        sb3_sac.yaml: SAC 하이퍼파라미터와 평가/저장 주기 등 런 설정.

        hvac_config.py: 하드코딩된 기본값(상수/경계)을 정의합니다. YAML을 로드해 덮어쓰는 방식을 권장합니다.

- rl/sb3/

        train_sac_sb3.py로 학습을 시작합니다. 내부에서 make_env.py를 통해 환경 생성 및 래퍼/정규화를 적용합니다.

        callbacks.py로 주기적 평가/베스트 모델 저장을 수행합니다.

        wrappers.py에는 관측/보상 스케일링, 에너지 게이트 등 실험용 래퍼가 있습니다.

- scripts/

        커맨드 단축 스크립트, 데이터 변환, 결과 리포팅 등 반복 작업을 배치합니다(선택).

- sim_test/

        시뮬레이터 단독 검증 및 회귀 테스트를 빠르게 수행하는 공간입니다.

        버그 리프로/실험 전 점검에 유용하며, CI에 연동하기 좋습니다.

- simulator/

        physics.py는 상태 전이의 소스 오브 트루스입니다. 에너지/질량(수분) 보존, 단위 일관성, 경계조건을 주기적으로 검증하세요.

        environment.py는 Gym 인터페이스와 보상 계산을 담당합니다. 보상 변경 시 여기를 우선 수정합니다.

        components.py는 제어입력→물리량 매핑과 소비전력 모델을 정의합니다(효율 맵/한계 검증 포함).

        sensors.py는 관측 노이즈/양자화 등의 센서 특성을 부여합니다.

- tests/

        배포 전 기본 회귀선입니다. 새로운 기능 추가 시 관련 테스트를 보강하고 PR에서 실행하세요.

## SAC 에이전트 학습 가이드

실행 명령어

- 학습:

```
python -m rl.sb3.train_sac_sb3 --config configs/sb3_sac.yaml

```

- 평가(sb3.eval_sb3가 있다면):

```
python -m rl.sb3.eval_sb3 \
  --model checkpoints/best_model.zip \
  --vecnorm checkpoints/vecnorm.pkl
결과물(정상 시 기대)
checkpoints/

best_model.zip (조기중단/가장 높은 평균 리워드)

final_model.zip (학습 종료 시점)

vecnorm.pkl (정규화 통계)

evaluations.npz (에피소드별 리워드/길이 기록, 콜백 기준)

```
