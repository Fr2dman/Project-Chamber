from simulator.environment import AdvancedSmartACSimulator
import numpy as np
from typing import Dict, Literal
import matplotlib.pyplot as plt
import pandas as pd

# 1. 시뮬레이터 초기화
simulator = AdvancedSmartACSimulator()
print("--- 초기 상태 ---")

# 시뮬레이터 상태 출력
initial_state = simulator._get_current_state()
print("내부 서보 각도:", [f"{s.current_angle:.1f}°" for s in simulator.internal_servos])
print("외부 서보 각도:", [f"{s.current_angle:.1f}°" for s in simulator.external_servos])
print("소형 팬 RPM:", [f"{f.current_rpm:.0f}" for f in simulator.small_fans])
print("대형 팬 RPM:", f"{simulator.large_fan.current_rpm:.0f}")
print("온도:", [f"{t:.1f}°C" for t in initial_state['temperatures']])
print("습도:", [f"{h:.1f}%" for h in initial_state['humidities']])
print("쾌적도:", [f"{score:}pt" for score in initial_state['comfort_scores']['comfort_scores']])
print("쾌적도 평균:", f"{initial_state['comfort_scores']['average_comfort']:.2f}")

# 2. 제어 값 설정 및 1-step 실행
print("\n--- 제어 입력 및 1-Step 실행 ---")

# 액션 벡터 생성 ([-1, 1] 범위, 14차원)
# 예시: 최대 냉각 및 최대 풍량
action = np.array([
    1.0,  # 펠티어: 최대 냉각
    1.0, 1.0, 1.0, 1.0,  # 내부 서보: 모두 최대로 열기
    1.0, 1.0, 1.0, 1.0,  # 외부 서보: 중간 각도
    1.0, 1.0, 1.0, 1.0,  # 소형 팬: 모두 최대 속도
    1.0   # 대형 팬: 최대 속도
], dtype=np.float32)

print("입력 액션 벡터:", action)

# 시뮬레이터 1-step 실행
state_vector, reward, done, info = simulator.step(action)

# 3. 1-step 실행 후 상태 출력
print("\n--- 1-Step 실행 후 상태 ---")

final_hw_state = info['hardware_states']
final_sensor_readings = info['sensor_readings']
final_comfort_data = info['comfort_data']

print("내부 서보 각도:", [f"{angle:.1f}°" for angle in final_hw_state['servos']['internal']])
print("외부 서보 각도:", [f"{angle:.1f}°" for angle in final_hw_state['servos']['external']])
print("소형 팬 RPM:", [f"{fan['rpm']:.0f}" for fan in final_hw_state['fans']['small_fans']])
print("대형 팬 RPM:", f"{final_hw_state['fans']['large_fan']['rpm']:.0f}")
print("온도 (센서):", [f"{t:.1f}°C" for t in final_sensor_readings['temperatures']])
print("습도 (센서):", [f"{h:.1f}%" for h in final_sensor_readings['humidities']])
print("쾌적도:", [f"{score:.2f}pt" for score in final_comfort_data['comfort_scores']])
print("쾌적도 평균:", f"{final_comfort_data['average_comfort']:.2f}")
print(f"소비 전력: {final_hw_state['total_power']:.2f} W")
print(f"보상: {reward:.3f}")