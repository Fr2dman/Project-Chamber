# unit_test_phi.py
import numpy as np
import matplotlib.pyplot as plt
from simulator.physics import PhysicsSimulator
from simulator.components import PeltierModel # 펠티어 모델을 직접 사용하기 위해 import

def run_case(theta_ext, theta_int=None, steps=50, dt=10.0):
    sim = PhysicsSimulator()
    peltier = PeltierModel() # 펠티어 모델 인스턴스 생성
    traj = []
    # 고정 제어값
    action = {
        'peltier_control': +1.0,                       # 최대 냉각 (+1.0)
        'internal_servo_angles': [theta_int, theta_int, 0, 0],       # 풍량 중간
        'external_servo_angles': [theta_ext, theta_ext, theta_ext, theta_ext], # 외부 서보 각도
        'small_fan_pwm': [10, 20, 30, 50],  # 소형 팬 PWM
        'large_fan_pwm': 50
    }

    # fan_states는 'small_fans'와 'large_fan' 키를 모두 포함해야 합니다.
    fan_states = {
        'small_fans': [{'rpm': 3500, 'power': 5}] * 4,
        'large_fan': {'rpm': 1650, 'power': 7.5}  # 50% PWM에 해당하는 대형 팬 상태
    }

    for _ in range(steps):
        # 동적 펠티어 모델을 위해 매 스텝마다 챔버 온도를 전달하여 업데이트
        avg_chamber_temp = np.mean(sim.T)
        pelt_state = {0: peltier.update(
            action['peltier_control'], avg_chamber_temp, sim.ambient_temp, dt)}
        state = sim.update_physics(action, pelt_state, fan_states, dt)
        traj.append(state['temperatures'].copy())     # (4,)
    
    print("Temperature at ", theta_ext, "°, ", theta_int, "° :", state['temperatures'])
    return np.array(traj)

ext_angles = [0, 40, 80]
in_angles = [0, 20, 40]
results = {ang: [] for ang in ext_angles}
for theta_ext in ext_angles:
    for theta_int in in_angles:
        print(f"Running case: θ_ext={theta_ext}°, θ_int={theta_int}°")
        traj = run_case(theta_ext, theta_int)
        results[theta_ext].append(traj)

# fig, axes = plt.subplots(len(ext_angles), len(in_angles), figsize=(15, 12), sharex=True, sharey=True)
# for i, theta_ext in enumerate(ext_angles):
#     for j, theta_int in enumerate(in_angles):
#         ax = axes[i, j]
#         # Retrieve trajectory for this combination
#         traj = results[theta_ext][j]
#         # Plot all zones in this subplot
#         for zone in range(traj.shape[1]):
#             ax.plot(traj[:, zone], label=f"Zone {zone}")
#         ax.set_title(f"θ_ext={theta_ext}°, θ_int={theta_int}°")
#         ax.set_xlabel("Time Steps")
#         ax.set_ylabel("Temperature (°C)")
#         ax.grid(True)
#         if i == 0 and j == 0:  # Show legend only in the first subplot to avoid clutter
#             ax.legend(fontsize="small")

# plt.tight_layout()
# plt.show()


# # ── 시각화 (Zone 0) ──────────────────────────────
# plt.figure(figsize=(12, 6))
# for theta_ext, trajs in results.items():
#     for theta_int, traj in zip(in_angles, trajs):
#         plt.plot(traj[:, 0], label=f"θ_ext={theta_ext}°, θ_int={theta_int}°")
# plt.title("Zone 0 Temperature Trajectory")
# plt.xlabel("Time Steps")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.grid()
# plt.show()
# ── 시각화 (Zone 1) ──────────────────────────────
plt.figure(figsize=(12, 6))
for theta_ext, trajs in results.items():
    for theta_int, traj in zip(in_angles, trajs):
        plt.plot(traj[:, 1], label=f"θ_ext={theta_ext}°, θ_int={theta_int}°")   
plt.title("Zone 1 Temperature Trajectory")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.show()
# # ── 시각화 (Zone 2) ──────────────────────────────
# plt.figure(figsize=(12, 6))
# for theta_ext, trajs in results.items():
#     for theta_int, traj in zip(in_angles, trajs):
#         plt.plot(traj[:, 2], label=f"θ_ext={theta_ext}°, θ_int={theta_int}°")
# plt.title("Zone 2 Temperature Trajectory")
# plt.xlabel("Time Steps")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.grid()
# plt.show()
# # ── 시각화 (Zone 3) ──────────────────────────────
# plt.figure(figsize=(12, 6))
# for theta_ext, trajs in results.items():
#     for theta_int, traj in zip(in_angles, trajs):
#         plt.plot(traj[:, 3], label=f"θ_ext={theta_ext}°, θ_int={theta_int}°")
# plt.title("Zone 3 Temperature Trajectory")
# plt.xlabel("Time Steps")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.grid()
# plt.show()
# 2×2 subplot 생성