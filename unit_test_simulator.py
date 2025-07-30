import numpy as np
import matplotlib.pyplot as plt
from simulator.environment import AdvancedSmartACSimulator
from simulator.utils import ZoneComfortCalculator

# ==========================
# 1. 초기 조건 설정 (수기 or 랜덤)
# ==========================
NUM_ZONES = 4

# 수기로 지정하려면 여기 편집
manual_init = {
    "use_manual": True,            # False → 랜덤 초기화
    "temperatures": [30, 30, 30, 30],   # °C
    "humidities":   [70, 65, 60, 55],   # %RH
    "tsv_values":   [+2, +1, +2, +1],   # Thermal Sensation Vote (-3~+3) Optional
}

# ==========================
# 2. 시뮬레이터 초기화 및 상태 주입
# ==========================
sim = AdvancedSmartACSimulator(num_zones=NUM_ZONES)

# 2‑1) comfort 계산기 준비 (존마다 하나)
comfort_calcs = [ZoneComfortCalculator(f"ZONE_{i}") for i in range(NUM_ZONES)]

if manual_init["use_manual"]:
    sim.physics_sim.T = np.array(manual_init["temperatures"], dtype=float)
    sim.physics_sim.H = np.array(manual_init["humidities"], dtype=float)
    # 센서 필터 상태도 맞춰줌
    for i in range(NUM_ZONES):
        sim.sensors[i].temp_filter_state = sim.physics_sim.T[i]
        sim.sensors[i].humidity_filter_state = sim.physics_sim.H[i]

# helper to compute comfort scores
def get_zone_scores(T, H):
    return [
        comfort_calcs[i].calculate_comfort(
            temp=T[i],
            rh=H[i],
            v=0.1          # 간단히 고정 풍속 or fan RPM → 풍속 변환함수 사용
        )["comfort_score"]
        for i in range(NUM_ZONES)
    ]
initial_scores = get_zone_scores(sim.physics_sim.T, sim.physics_sim.H)


# ==========================
# 3. 제어 시나리오 정의
#    - zone 0,1 강냉각·풍량↑  zone2,3 약제어
# ==========================
steps = 48   # 48 * 5s = 4 min
trajectory_T, trajectory_H, trajectory_score = [], [], []

for step in range(steps):
    # action vector [-1,1] 길이 14
    action = np.zeros(sim.action_dim)
    # peltier: 전체 냉각 강도
    action[0] = -1.0 if step < steps//2 else -0.3   # 전반 강냉각, 후반 완화
    
    # internal servo angles (0~45deg) normalized
    action[1:5] = np.array([+1, +1, -0.5, -0.5])    # zone0,1 wide open
    
    # external servo angles (0~80deg) normalized
    action[5:9] = np.array([-1, -1, +1, +1])        # zone0,1 수평, zone2,3 수직
    
    # small fan PWM (0~80%) normalized
    action[9:13] = np.array([+1, +1, -0.5, -0.5]) 
    
    # large fan
    action[13] = 0.5
    
    state, reward, done, info = sim.step(action)
    
    trajectory_T.append(info["sensor_readings"]["temperatures"])
    trajectory_H.append(info["sensor_readings"]["humidities"])
    trajectory_score.append(info["comfort_data"]["comfort_scores"])
    
    if done:
        break

trajectory_T = np.array(trajectory_T)
trajectory_H = np.array(trajectory_H)
trajectory_score = np.array(trajectory_score)

# ==========================
# 4. 시각화
# ==========================
zones = np.arange(NUM_ZONES)
time_axis = np.arange(trajectory_T.shape[0]) * sim.dt / 60  # minutes

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for z in zones:
    axes[0].plot(time_axis, trajectory_T[:, z], label=f"Zone {z}")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Zone Temperatures over Time")
axes[0].legend(); axes[0].grid(True)

for z in zones:
    axes[1].plot(time_axis, trajectory_H[:, z], label=f"Zone {z}")
axes[1].set_ylabel("Humidity (%RH)")
axes[1].set_title("Zone Humidity over Time")
axes[1].grid(True)

for z in zones:
    axes[2].plot(time_axis, trajectory_score[:, z], label=f"Zone {z}")
axes[2].set_ylabel("Comfort Score")
axes[2].set_xlabel("Time (minutes)")
axes[2].set_title("Comfort Score over Time")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ----------------- 초기 vs 최종 상태 이미징 -----------------
fig2, ax2 = plt.subplots(1, NUM_ZONES, figsize=(12, 3))
for z in zones:
    ax2[z].text(0.5, 0.7, f"T: {sim.physics_sim.T[z]:.1f}°C", ha='center', va='center', fontsize=12)
    ax2[z].text(0.5, 0.5, f"H: {sim.physics_sim.H[z]:.0f}%", ha='center', va='center', fontsize=12)
    ax2[z].text(0.5, 0.3, f"S: {trajectory_score[-1, z]:.1f}", ha='center', va='center', fontsize=12)
    ax2[z].set_title(f"Zone {z}")
    ax2[z].set_xticks([]); ax2[z].set_yticks([])
plt.suptitle("Final Zone States")
plt.show()

# ==========================
# 5. 요약 출력
# ==========================
import pandas as pd
summary = pd.DataFrame({
    "Zone": zones,
    "Initial_T": sim.physics_sim.T,
    "Initial_H": sim.physics_sim.H,
    "Initial_Score": initial_scores,
    "Final_T": trajectory_T[-1],
    "Final_H": trajectory_H[-1],
    "Final_Score": trajectory_score[-1]
})
print(summary.round(2))

