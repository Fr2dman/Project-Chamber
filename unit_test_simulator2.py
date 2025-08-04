# ------------------------------------------
# file: unit_test_simulator2.py
# ------------------------------------------
"""
HVAC 시뮬레이터 종합 테스트 코드
- 초기 환경 설정
- 다양한 제어 시나리오 테스트
- 실시간 시각화 및 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
from typing import Dict, List, Tuple, Callable
import time
from dataclasses import dataclass
from pathlib import Path

# Mock import (실제 환경에서는 from simulator.environment import AdvancedSmartACSimulator)
from simulator.environment import AdvancedSmartACSimulator

@dataclass
class TestScenario:
    """테스트 시나리오 정의"""
    name: str
    duration_steps: int
    initial_temps: List[float]
    initial_humidity: List[float]
    control_function: Callable[[int, Dict], np.ndarray]
    description: str

class HVACSimulatorTester:
    """HVAC 시뮬레이터 테스트 클래스"""
    
    def __init__(self, num_zones: int = 4):
        self.num_zones = num_zones
        self.simulator = AdvancedSmartACSimulator(num_zones)
        
        # 테스트 결과 저장
        self.test_results = {}
        self.current_data = {
            'step': [],
            'temperatures': [[] for _ in range(num_zones)],
            'humidities': [[] for _ in range(num_zones)],
            'comfort_scores': [[] for _ in range(num_zones)],
            'power_consumption': [],
            'actions': [],
            'rewards': []
        }
        
        # 시각화 설정
        plt.style.use('default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
    def create_control_scenarios(self) -> Dict[str, TestScenario]:
        """다양한 제어 시나리오 생성"""
        
        def aggressive_cooling(step: int, state_info: Dict) -> np.ndarray:
            """적극적 냉각 시나리오"""
            return np.array([
                1.0,  # 펠티어 최대 냉각
                *[0.8, 0.8, 0.8, 0.8],  # 내부 서보 각도
                *[0.6, 0.6, 0.6, 0.6],  # 외부 서보 각도
                *[0.9, 0.9, 0.9, 0.9],  # 소형 팬 PWM
                0.7   # 대형 팬 PWM
            ])
        
        def gentle_control(step: int, state_info: Dict) -> np.ndarray:
            """온화한 제어 시나리오"""
            return np.array([
                0.3,  # 펠티어 약한 냉각
                *[0.4, 0.4, 0.4, 0.4],  # 내부 서보 각도
                *[0.3, 0.3, 0.3, 0.3],  # 외부 서보 각도
                *[0.5, 0.5, 0.5, 0.5],  # 소형 팬 PWM
                0.3   # 대형 팬 PWM
            ])
        
        def adaptive_control(step: int, state_info: Dict) -> np.ndarray:
            """적응형 제어 시나리오 (온도에 따라 제어 강도 조절)"""
            if not state_info or 'sensor_readings' not in state_info:
                return gentle_control(step, state_info)
            
            temps = state_info['sensor_readings']['temperatures']
            avg_temp = np.mean(temps) # type: ignore
            target_temp = 24.0
            
            # 온도 차이에 따른 제어 강도 계산
            temp_error = avg_temp - target_temp
            cooling_intensity = np.clip(temp_error / 5.0, -1.0, 1.0)
            fan_intensity = np.clip(abs(temp_error) / 3.0, 0.2, 0.9)
            
            return np.array([
                cooling_intensity,
                *[0.5 + fan_intensity * 0.3] * 4,  # 내부 서보
                *[0.4 + fan_intensity * 0.4] * 4,  # 외부 서보
                *[fan_intensity] * 4,              # 소형 팬
                fan_intensity * 0.8                # 대형 팬
            ])
        
        def zone_differential_control(step: int, state_info: Dict) -> np.ndarray:
            """존별 차등 제어 시나리오"""
            if not state_info or 'sensor_readings' not in state_info:
                return gentle_control(step, state_info)
            
            temps = state_info['sensor_readings']['temperatures']
            action = np.zeros(14)
            action[0] = 0.5  # 펠티어 기본값
            
            int_servos, ext_servos, small_fans = [], [], []
            
            # 존별로 다른 제어 적용
            for temp in temps: # type: ignore
                if temp > 26.0:  # 뜨거운 존
                    int_servos.append(0.8)
                    ext_servos.append(0.7)
                    small_fans.append(0.9)
                elif temp < 22.0:  # 차가운 존
                    int_servos.append(0.2)
                    ext_servos.append(0.2)
                    small_fans.append(0.3)
                else:  # 적정 온도 존
                    int_servos.append(0.5)
                    ext_servos.append(0.4)
                    small_fans.append(0.6)
            
            action[1:5], action[5:9], action[9:13] = int_servos, ext_servos, small_fans
            action[13] = 0.5  # 대형 팬
            return action
        
        def step_response_test(step: int, state_info: Dict) -> np.ndarray:
            """스텝 응답 테스트 (제어 신호 급변)"""
            if step < 20:
                return np.zeros(14)  # 제어 없음
            elif step < 40:
                return np.ones(14) * 0.8  # 강한 제어
            else:
                return np.ones(14) * 0.3  # 약한 제어
        
        return {
            'aggressive': TestScenario(
                "Aggressive Cooling",
                60,
                [28.0, 27.5, 28.5, 27.0],
                [65.0, 60.0, 70.0, 55.0],
                aggressive_cooling,
                "최대 성능으로 빠른 냉각"
            ),
            'gentle': TestScenario(
                "Gentle Control",
                80,
                [26.0, 25.5, 26.5, 25.0],
                [55.0, 50.0, 60.0, 45.0],
                gentle_control,
                "에너지 절약형 온화한 제어"
            ),
            'adaptive': TestScenario(
                "Adaptive Control",
                100,
                [29.0, 25.0, 23.0, 27.0],
                [70.0, 40.0, 45.0, 65.0],
                adaptive_control,
                "온도에 따른 적응형 제어"
            ),
            'differential': TestScenario(
                "Zone Differential",
                90,
                [30.0, 22.0, 26.0, 24.0],
                [75.0, 35.0, 55.0, 50.0],
                zone_differential_control,
                "존별 차등 제어"
            ),
            'step_response': TestScenario(
                "Step Response",
                60,
                [25.0, 25.0, 25.0, 25.0],
                [50.0, 50.0, 50.0, 50.0],
                step_response_test,
                "스텝 응답 특성 테스트"
            )
        }
    
    def run_scenario(self, scenario: TestScenario, visualize: bool = True) -> Dict:
        """시나리오 실행"""
        print(f"\n=== 시나리오 실행: {scenario.name} ===")
        print(f"설명: {scenario.description}")
        print(f"Duration: {scenario.duration_steps} steps")
        
        # 시뮬레이터 초기화 (Mock)
        self.simulator.reset()
        self.simulator.set_initial_state(scenario.initial_temps, scenario.initial_humidity)
        
        # 데이터 초기화
        self._reset_data()
        
        # 이전 스텝의 정보를 저장할 변수
        info = {}

        for step in range(scenario.duration_steps):
            # 제어 액션 생성
            action = scenario.control_function(step, info)
            
            # 시뮬레이션 스텝 실행
            state, reward, done, info = self.simulator.step(action)
            
            # 실제 데이터 저장
            self._save_step_data(step, info, action, reward)
            
            if step % 10 == 0:
                avg_temp = np.mean(info['sensor_readings']['temperatures'])
                print(f"Step {step}: Avg Temp = {avg_temp:.1f}°C")
        
        # 결과 분석
        results = self._analyze_results(scenario.name)
        self.test_results[scenario.name] = results
        
        if visualize:
            self.visualize_results(scenario.name)
        
        return results
    
    def _reset_data(self):
        """데이터 저장소 초기화"""
        self.current_data = {
            'step': [],
            'temperatures': [[] for _ in range(self.num_zones)],
            'humidities': [[] for _ in range(self.num_zones)],
            'comfort_scores': [[] for _ in range(self.num_zones)],
            'power_consumption': [],
            'actions': [],
            'rewards': []
        }
    
    def _save_step_data(self, step: int, info: Dict, action: np.ndarray, reward: float):
        """스텝 데이터 저장"""
        self.current_data['step'].append(step)
        
        temps = info['sensor_readings']['temperatures']
        humids = info['sensor_readings']['humidities']
        comforts = info['comfort_data']['comfort_scores']
        power = info['hardware_states']['total_power']
        
        for i in range(self.num_zones):
            self.current_data['temperatures'][i].append(temps[i])
            self.current_data['humidities'][i].append(humids[i])
            self.current_data['comfort_scores'][i].append(comforts[i])
        self.current_data['power_consumption'].append(power)
        self.current_data['actions'].append(action.copy())
        self.current_data['rewards'].append(reward)
    
    def _analyze_results(self, scenario_name: str) -> Dict:
        """결과 분석"""
        data = self.current_data
        
        # 온도 통계
        temp_stats = {}
        for i in range(self.num_zones):
            temps = np.array(data['temperatures'][i])
            temp_stats[f'zone_{i}'] = {
                'mean': np.mean(temps),
                'std': np.std(temps),
                'min': np.min(temps),
                'max': np.max(temps),
                'final': temps[-1] if len(temps) > 0 else 0,
                'settling_time': self._calculate_settling_time(temps, 24.0)
            }
        
        # 전체 성능 지표
        total_power = np.sum(data['power_consumption'])
        avg_comfort = np.mean([np.mean(data['comfort_scores'][i]) for i in range(self.num_zones)])
        avg_reward = np.mean(data['rewards'])
        
        return {
            'scenario_name': scenario_name,
            'temperature_stats': temp_stats,
            'total_power_consumption': total_power,
            'average_comfort_score': avg_comfort,
            'average_reward': avg_reward,
            'simulation_steps': len(data['step'])
        }
    
    def _calculate_settling_time(self, temps: np.ndarray, target: float, 
                               tolerance: float = 0.5) -> int:
        """정착 시간 계산 (목표값 ± tolerance 범위에 도달하는 시간)"""
        if len(temps) == 0:
            return -1
        
        for i, temp in enumerate(temps):
            if abs(temp - target) <= tolerance:
                # 이후 5스텝 동안 유지되는지 확인
                if i + 5 < len(temps):
                    if all(abs(temps[j] - target) <= tolerance for j in range(i, i + 5)):
                        return i
        return -1  # 정착하지 않음
    
    def visualize_results(self, scenario_name: str):
        """결과 시각화"""
        data = self.current_data
        steps = data['step']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'HVAC Simulation Results: {scenario_name}', fontsize=16, fontweight='bold')
        
        # 1. 온도 변화
        ax1 = axes[0, 0]
        for i in range(self.num_zones):
            ax1.plot(steps, data['temperatures'][i], 
                    color=self.colors[i], label=f'Zone {i+1}', linewidth=2)
        ax1.axhline(y=24.0, color='red', linestyle='--', alpha=0.7, label='Target')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Control')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 습도 변화
        ax2 = axes[0, 1]
        for i in range(self.num_zones):
            ax2.plot(steps, data['humidities'][i], 
                    color=self.colors[i], label=f'Zone {i+1}', linewidth=2)
        ax2.axhline(y=50.0, color='red', linestyle='--', alpha=0.7, label='Target')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Humidity (%)')
        ax2.set_title('Humidity Control')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 쾌적도 점수
        ax3 = axes[1, 0]
        for i in range(self.num_zones):
            ax3.plot(steps, data['comfort_scores'][i], 
                    color=self.colors[i], label=f'Zone {i+1}', linewidth=2)
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Comfort Score')
        ax3.set_title('Comfort Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 소비 전력 및 보상
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        power_line = ax4.plot(steps, data['power_consumption'], 
                             color='orange', linewidth=2, label='Power')
        reward_line = ax4_twin.plot(steps, data['rewards'], 
                                   color='green', linewidth=2, label='Reward')
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Power Consumption (W)', color='orange')
        ax4_twin.set_ylabel('Reward', color='green')
        ax4.set_title('Power & Reward')
        
        # 범례 통합
        lines = power_line + reward_line
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_zone_heatmap(self, scenario_name: str, step_range: Tuple[int, int] = None):
        """존별 온도 히트맵 생성"""
        data = self.current_data
        
        if step_range is None:
            step_range = (0, len(data['step']))
        
        start_idx, end_idx = step_range
        steps_subset = data['step'][start_idx:end_idx]
        
        # 2x2 존 배치로 히트맵 데이터 구성
        heatmap_data = np.zeros((len(steps_subset), 2, 2))
        
        for t_idx, step in enumerate(steps_subset):
            heatmap_data[t_idx, 0, 0] = data['temperatures'][0][start_idx + t_idx]  # Zone 1
            heatmap_data[t_idx, 0, 1] = data['temperatures'][1][start_idx + t_idx]  # Zone 2
            heatmap_data[t_idx, 1, 0] = data['temperatures'][2][start_idx + t_idx]  # Zone 3
            heatmap_data[t_idx, 1, 1] = data['temperatures'][3][start_idx + t_idx]  # Zone 4
        
        # 애니메이션 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(heatmap_data[0], cmap='RdYlBu_r', vmin=20, vmax=30)
        ax.set_title(f'Zone Temperature Heatmap: {scenario_name}')
        
        # 존 라벨 추가
        for i in range(2):
            for j in range(2):
                zone_num = i * 2 + j + 1
                ax.text(j, i, f'Zone {zone_num}', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (°C)')
        
        def animate(frame):
            im.set_array(heatmap_data[frame])
            ax.set_title(f'Zone Temperature Heatmap: {scenario_name} (Step {steps_subset[frame]})')
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(steps_subset), 
                           interval=200, blit=True, repeat=True)
        
        plt.show()
        return anim
    
    def generate_report(self, scenarios: List[str] = None) -> str:
        """테스트 보고서 생성"""
        if scenarios is None:
            scenarios = list(self.test_results.keys())
        
        report = []
        report.append("HVAC Simulator Test Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of zones: {self.num_zones}")
        report.append("")
        
        for scenario_name in scenarios:
            if scenario_name not in self.test_results:
                continue
                
            results = self.test_results[scenario_name]
            report.append(f"Scenario: {scenario_name}")
            report.append("-" * 30)
            
            # 온도 통계
            report.append("Temperature Performance:")
            for zone, stats in results['temperature_stats'].items():
                report.append(f"  {zone}: Mean={stats['mean']:.1f}°C, "
                            f"Std={stats['std']:.1f}°C, "
                            f"Final={stats['final']:.1f}°C, "
                            f"Settling={stats['settling_time']}steps")
            
            # 전체 성능
            report.append(f"Total Power: {results['total_power_consumption']:.1f} W·steps")
            report.append(f"Average Comfort: {results['average_comfort_score']:.1f}")
            report.append(f"Average Reward: {results['average_reward']:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """결과를 파일로 저장"""
        if filename is None:
            filename = f"hvac_test_results_{int(time.time())}.json"
        
        import json
        
        # numpy 배열을 리스트로 변환
        save_data = {}
        for scenario, results in self.test_results.items():
            save_data[scenario] = results
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filename}")


def main():
    """메인 테스트 실행"""
    print("HVAC Simulator Test Suite")
    print("=" * 40)
    
    # 테스터 초기화
    tester = HVACSimulatorTester(num_zones=4)
    
    # 시나리오 생성
    scenarios = tester.create_control_scenarios()
    
    # 선택적 시나리오 실행
    test_scenarios = ['aggressive', 'gentle', 'adaptive', 'differential']
    animations = [] # 애니메이션 객체 저장 (가비지 컬렉션 방지)
    
    for scenario_name in test_scenarios:
        if scenario_name in scenarios:
            scenario = scenarios[scenario_name]
            tester.run_scenario(scenario, visualize=True)
            print(f"\n{scenario_name} 시나리오 완료!")
            
            # 히트맵 생성 (옵션)
            anim = tester.create_zone_heatmap(scenario_name, (0, 30))
            animations.append(anim)
    
    # 보고서 생성
    report = tester.generate_report(test_scenarios)
    print("\n" + report)
    
    # 결과 저장
    tester.save_results()


if __name__ == "__main__":
    main()