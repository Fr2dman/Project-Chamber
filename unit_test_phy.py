# unit_test_phi.py
import numpy as np
import matplotlib.pyplot as plt
from simulator.physics import PhysicsSimulator
from simulator.components import PeltierModel # 펠티어 모델을 직접 사용하기 위해 import

def run_case(theta_ext, steps=20, dt=10.0):
    sim = PhysicsSimulator()
    peltier = PeltierModel() # 펠티어 모델 인스턴스 생성
    print(sim.T)
    traj = []
    # 고정 제어값
    action = {
        'peltier_control': 1.0,                # 최대 냉각 (+1.0)
        'internal_servo_angles': [40]*4,       # 풍량 중간
        'external_servo_angles': [theta_ext]*4,
        'small_fan_pwm': [50]*4,
        'large_fan_pwm': 50
    }
    fan_states = {'small_fans':[{'rpm':3500,'power':5}]*4}
    for _ in range(steps):
        # 동적 펠티어 모델을 위해 매 스텝마다 챔버 온도를 전달하여 업데이트
        avg_chamber_temp = np.mean(sim.T)
        pelt_state = {0: peltier.update(
            action['peltier_control'], avg_chamber_temp, sim.ambient_temp, dt)}
        state = sim.update_physics(action, pelt_state, fan_states, dt)
        traj.append(state['temperatures'].copy())     # (4,)
    
    print(state['temperatures'])
    return np.array(traj)

angles = [0, 40, 80]
results = {ang: run_case(ang) for ang in angles}

# ── 시각화 (Zone 0) ──────────────────────────────
t = np.arange(results[0].shape[0]) * 5 / 60   # min
plt.figure()
for ang in angles:
    plt.plot(t, results[ang][:,0], label=f'θ_ext={ang}°')
plt.xlabel('Time (min)'); plt.ylabel('Temp Zone 0 (°C)')
plt.title('External‑slot angle effect')
plt.legend(); plt.grid(True); plt.show()
