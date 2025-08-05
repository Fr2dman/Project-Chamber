from simulator.environment import AdvancedSmartACSimulator
import numpy as np
from typing import Dict, Literal
import matplotlib.pyplot as plt
import pandas as pd
# 시뮬레이터 초기화
simulator = AdvancedSmartACSimulator()  

# 현재 상태를 반환하는 메소드 호출 예시
state = simulator._get_current_state()  
# print("Current State:", state)

# 시뮬레이터 상태 출력
print("Simulator Internal_servos:", [simulator.internal_servos[i].current_angle for i in range(len(simulator.internal_servos))])
print("Simulator External servos:", [simulator.external_servos[i].current_angle for i in range(len(simulator.external_servos))])
print("Simulator Small_fans:", [simulator.small_fans[i].current_rpm for i in range(len(simulator.small_fans))])
print("Simulator Large_fan:", simulator.large_fan.current_rpm)

print("Temperatures:", state['temperatures'])
print("Humidities:", state['humidities'])
print("Comfort Scores:", state['comfort_scores'])