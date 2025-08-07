import numpy as np
from simulator.environment import AdvancedSmartACSimulator

"""
간단 Smoke‑Test
----------------
* TSV 피드백을 주입(update_tsv)
* 동일 액션 10 step 실행하면서 reward breakdown 출력
* _update_hardware 리팩터 이후 환경이 정상 동작하는지 확인용
"""

NUM_STEPS = 10

# 1) 환경 초기화
env = AdvancedSmartACSimulator()
print("\n=== 초기 상태 ===")
print("Temperatures:", env.physics_sim.T)
print("Humidities  :", env.physics_sim.H)
# 초기 물리 상태를 가져와서 센서/쾌적도 계산에 사용
initial_physics_state = env.physics_sim.get_current_state()
initial_sensor_readings = env._read_sensors(initial_physics_state)
print("Comfort Avg :", env._calculate_comfort(initial_sensor_readings)['average_comfort'])

# 2) TSV 입력 (모두 춥다고 피드백)
cold_tsv = [-2.3] * env.num_zones
env.update_tsv(cold_tsv)
print("주입 TSV:", cold_tsv)

# 3) 예시 액션 (모두 +1 → 최대 냉각/풍량)
action = np.ones(env.action_dim, dtype=np.float32)

# 4) 루프
for t in range(NUM_STEPS):
    # 예: 카메라 시스템으로부터 새로운 TSV 값을 받아온다고 가정
    new_tsv = [-2.3] * env.num_zones  # 여기서는 고정값, 실제론 탐지 결과
    env.update_tsv(new_tsv)

    state, reward, done, info = env.step(action)

    rb = info['reward_breakdown']
    print(f"--- Step {t+1} ---")
    print("TSV       :", new_tsv)
    print("Temps     :", [f"{x:.1f}" for x in info['sensor_readings']['temperatures']])
    print("ComfortAvg:", f"{info['comfort_data']['average_comfort']:.2f}")
    print("Reward    :", f"{reward:.3f}")
    print("  R_dir   :", f"{rb['R_dir']:.3f}")
    print("  R_c     :", f"{rb['R_comfort']:.3f}")
    print("  R_energy:", f"{rb['R_energy']:.3f}")
    print("  R_smooth:", f"{rb['R_smooth']:.3f}")
    print("  R_safe  :", rb['R_safety'])
    print("  R_total :", f"{rb['reward']:.3f}")

    if done:
        print("환경이 종료되었습니다.")
        break

print("\n테스트 완료.")
