import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unittest # 기본적인 테스트 구조를 위해 unittest.TestCase 상속

# 제공해주신 시뮬레이터 코드가 'simulator' 폴더에 있다고 가정합니다.
from simulator.environment import AdvancedSmartACSimulator

def run_simulation(env: AdvancedSmartACSimulator, fixed_action: np.ndarray, duration_steps: int = 120):
    """
    주어진 환경과 고정 행동으로 시뮬레이션을 실행하고 결과를 반환합니다.
    """
    history = []
    
    # 초기 상태 기록
    history.append({
        'step': 0,
        'T_zone0': env.physics_sim.T[0], 'T_zone1': env.physics_sim.T[1],
        'T_zone2': env.physics_sim.T[2], 'T_zone3': env.physics_sim.T[3],
        'H_zone0': env.physics_sim.H[0],
        'T_ambient': env.physics_sim.ambient_temp,
        'peltier_thermal_power': 0,
        'peltier_cold_temp': env.peltier.cold_side_temp,
        'total_power_consumption': 0
    })

    # 시뮬레이션 루프
    for step in range(1, duration_steps + 1):
        _, _, _, info = env.step(fixed_action)
        
        hw_states = info['hardware_states']
        phys_state = env.physics_sim.get_current_state()
        
        record = {
            'step': step,
            'T_zone0': phys_state['temperatures'][0], 'T_zone1': phys_state['temperatures'][1],
            'T_zone2': phys_state['temperatures'][2], 'T_zone3': phys_state['temperatures'][3],
            'H_zone0': phys_state['humidities'][0],
            'T_ambient': env.physics_sim.ambient_temp,
            'peltier_thermal_power': hw_states['peltier'][0]['thermal_power'],
            'peltier_cold_temp': hw_states['peltier'][0]['cold_side_temp'],
            'total_power_consumption': hw_states['total_power']
        }
        history.append(record)
        
    return pd.DataFrame(history)

def plot_results_detailed(df: pd.DataFrame, title: str):
    """결과를 여러 서브플롯으로 나누어 상세히 시각화합니다."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(title, fontsize=16)
    time_ax = df['step'] * 0.5  # 분 단위 시간

    # --- Plot 1: Zone Temperatures ---
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].plot(time_ax, df['T_zone0'], label='Zone 0 Temp', color='b', marker='.')
    axes[0].plot(time_ax, df['T_zone1'], label='Zone 1 Temp', color='c', linestyle='--')
    axes[0].plot(time_ax, df['T_zone2'], label='Zone 2 Temp', color='g', linestyle='--')
    axes[0].plot(time_ax, df['T_zone3'], label='Zone 3 Temp', color='m', linestyle='--')
    axes[0].plot(time_ax, df['T_ambient'], label='Ambient Temp', color='k', linestyle='-')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title('Zone Temperature Dynamics')

    # --- Plot 2: Peltier States ---
    axes[1].set_ylabel('Temperature (°C) / Power (W)')
    ax2_twin = axes[1].twinx()
    axes[1].plot(time_ax, df['peltier_cold_temp'], label='Peltier Cold Plate Temp (°C)', color='r', marker='.')
    ax2_twin.plot(time_ax, df['peltier_thermal_power'], label='Peltier Thermal Power (W)', color='orange', linestyle=':')
    axes[1].set_ylabel('Peltier Temp (°C)', color='r')
    ax2_twin.set_ylabel('Thermal Power (W)', color='orange')
    axes[1].grid(True)
    axes[1].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    axes[1].set_title('Peltier Component Behavior')

    # --- Plot 3: Power Consumption ---
    axes[2].set_ylabel('Power (W)')
    axes[2].plot(time_ax, df['total_power_consumption'], label='Total Power Consumption', color='gold', marker='.')
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_title('System Power Consumption')
    axes[2].set_xlabel('Time (minutes)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


class TestPhysicsSimulator(unittest.TestCase):
    
    def setUp(self):
        """각 테스트 전에 공통적으로 시뮬레이터 환경을 설정합니다."""
        self.env = AdvancedSmartACSimulator(num_zones=4)

    def test_scenario_1_baseline_drift(self):
        """
        영역: 수동적 열 교환 (ZoneEnergyBalance)
        시나리오: 모든 제어 OFF. 내부 온도가 외부 온도로 수렴하는지 확인.
        """
        print("\n--- Running Scenario 1: Baseline Drift ---")
        # Arrange
        initial_temps = [25.0] * 4
        self.env.set_initial_state(temperatures=initial_temps, humidities=[50.0] * 4)
        action_off = np.full(self.env.action_dim, -1.0)
        
        # Act
        df = run_simulation(self.env, action_off, duration_steps=120)

        # Assert
        initial_temp = df['T_zone0'].iloc[0]
        final_temp = df['T_zone0'].iloc[-1]
        ambient_temp = df['T_ambient'].iloc[0]
        self.assertGreater(final_temp, initial_temp, "온도는 외부온도를 향해 올라가야 합니다.")
        self.assertLess(final_temp, ambient_temp, "온도가 외부온도를 초과해서는 안됩니다.")
        self.assertAlmostEqual(df['total_power_consumption'].iloc[-1], 0, delta=0.1, msg="전력소모는 거의 0이어야 합니다.")

        # Analyze
        plot_results_detailed(df, "Scenario 1: Baseline - Drifting towards Ambient")

    def test_scenario_2_max_cooling(self):
        """
        영역: 펠티어 + 팬 통합 동작 (PeltierModel, FanModel, ZoneEnergyBalance)
        시나리오: 모든 제어 MAX. 모든 존의 온도가 효과적으로 냉각되는지 확인.
        """
        print("\n--- Running Scenario 2: Max Cooling ---")
        # Arrange
        initial_temps = [28.0] * 4
        self.env.set_initial_state(temperatures=initial_temps, humidities=[50.0] * 4)
        action_max_cool = np.full(self.env.action_dim, 1.0)
        
        # Act
        df = run_simulation(self.env, action_max_cool, duration_steps=60)
        
        # Assert
        self.assertLess(df['T_zone0'].iloc[-1], df['T_zone0'].iloc[0], "Zone 0 온도는 감소해야 합니다.")
        self.assertLess(df['peltier_cold_temp'].min(), 0, "펠티어 냉각판 온도는 영하로 떨어져야 합니다.")
        self.assertLess(df['peltier_thermal_power'].iloc[-1], -10, "펠티어 열 출력은 큰 음수여야 합니다 (냉각).")
        self.assertGreater(df['total_power_consumption'].iloc[-1], 50, "최대 가동 시 상당한 전력을 소모해야 합니다.")

        # Analyze
        plot_results_detailed(df, "Scenario 2: Max Cooling - All Controls On")

    def test_scenario_3_air_mixing_with_fans(self):
        """
        영역: 공기 유동 및 혼합 (JetModel)
        시나리오: 펠티어는 OFF, 팬만 MAX. 서로 다른 온도의 존들이 평균 온도로 수렴하는지 확인.
        """
        print("\n--- Running Scenario 3: Air Mixing ---")
        # Arrange
        initial_temps = [28.0, 22.0, 30.0, 20.0]
        self.env.set_initial_state(temperatures=initial_temps, humidities=[50.0] * 4)
        action_fan_only = np.full(self.env.action_dim, -1.0)
        action_fan_only[9:14] = 1.0  # 모든 팬 ON
        
        # Act
        df = run_simulation(self.env, action_fan_only, duration_steps=60)
        
        # Assert
        initial_temps_std = np.std(initial_temps)
        final_temps = df[['T_zone0', 'T_zone1', 'T_zone2', 'T_zone3']].iloc[-1].values
        final_temps_std = np.std(final_temps)
        self.assertLess(final_temps_std, initial_temps_std * 0.5, "팬 작동 후 온도 표준편차는 크게 감소해야 합니다.")
        
        # Analyze
        plot_results_detailed(df, "Scenario 3: Ventilation Only - Mixing Temperatures")
        
    def test_scenario_4_peltier_without_fans(self):
        """
        영역: 펠티어 단독 동작 및 열 전달 조건
        시나리오: 펠티어만 MAX, 모든 팬 OFF. 냉각 에너지가 공기로 전달되지 않는지 확인.
        """
        print("\n--- Running Scenario 4: Peltier Only (No Fans) ---")
        # Arrange
        initial_temps = [25.0] * 4
        self.env.set_initial_state(temperatures=initial_temps, humidities=[50.0] * 4)
        action_peltier_only = np.full(self.env.action_dim, -1.0)
        action_peltier_only[0] = 1.0  # Peltier ON

        # Act
        df = run_simulation(self.env, action_peltier_only, duration_steps=60)

        # Assert
        # 팬이 없으므로 냉각 에너지가 거의 전달되지 않고, 자연대류로 온도는 오히려 약간 상승해야 함
        self.assertGreater(df['T_zone0'].iloc[-1], df['T_zone0'].iloc[0] - 0.5, "팬이 없으면 존 온도는 거의 변하지 않거나 약간 상승해야 합니다.")
        self.assertLess(df['peltier_cold_temp'].min(), -10, "펠티어 냉각판 자체는 매우 차가워져야 합니다.")
        self.assertAlmostEqual(df['peltier_thermal_power'].iloc[-1], 0, delta=1.0, msg="팬이 없으면 공기로 전달되는 열(thermal power)은 거의 0이어야 합니다.")

        # Analyze
        plot_results_detailed(df, "Scenario 4: Peltier Only (No Air Circulation)")


if __name__ == '__main__':
    # unittest를 사용하여 모든 테스트 실행
    unittest.main()