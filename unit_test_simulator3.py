# ------------------------------------------
# file: unit_test_simulator_v2.py
# ------------------------------------------
"""
HVAC 시뮬레이터 종합 테스트 코드 (v2.0)
- 시뮬레이터와 완전 호환
- 다양한 제어 시나리오 테스트
- 실시간 시각화 및 분석
- 성능 지표 및 안전성 검증
- 에너지 효율성 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 시뮬레이터 import
try:
    from simulator.environment import AdvancedSmartACSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    print("Warning: Simulator module not found. Using mock data for demonstration.")
    SIMULATOR_AVAILABLE = False

@dataclass
class TestScenario:
    """테스트 시나리오 정의"""
    name: str
    duration_steps: int
    initial_temps: List[float]
    initial_humidity: List[float]
    control_function: Callable[[int, Dict], np.ndarray]
    description: str
    target_temp: float = 24.0
    target_humidity: float = 50.0

@dataclass
class PerformanceMetrics:
    """성능 지표 정의"""
    temperature_rmse: float
    humidity_rmse: float
    comfort_score: float
    energy_efficiency: float  # comfort per watt
    settling_time: int
    overshoot_percentage: float
    steady_state_error: float
    safety_violations: int

class MockSimulator:
    """시뮬레이터가 없을 때 사용할 Mock 클래스"""
    def __init__(self, num_zones=4):
        self.num_zones = num_zones
        self.state_dim = 35
        self.action_dim = 14
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.temperatures = np.random.uniform(22, 28, self.num_zones)
        self.humidities = np.random.uniform(40, 70, self.num_zones)
        return np.random.randn(self.state_dim)
    
    def set_initial_state(self, temps, humids):
        self.temperatures = np.array(temps)
        self.humidities = np.array(humids)
    
    def step(self, action):
        self.step_count += 1
        # 간단한 물리 모델 시뮬레이션
        cooling_effect = action[0] * 0.1
        self.temperatures += np.random.normal(-cooling_effect, 0.1, self.num_zones)
        self.temperatures = np.clip(self.temperatures, 15, 35)
        
        info = {
            'sensor_readings': {
                'temperatures': self.temperatures.tolist(),
                'humidities': self.humidities.tolist(),
                'co2_levels': np.random.uniform(400, 800, self.num_zones).tolist(),
                'dust_levels': np.random.uniform(0, 10, self.num_zones).tolist(),
            },
            'comfort_data': {
                'comfort_scores': np.random.uniform(70, 95, self.num_zones).tolist(),
                'average_comfort': 85.0
            },
            'hardware_states': {
                'total_power': np.random.uniform(50, 150)
            },
            'reward_breakdown': {
                'comfort': 0.8,
                'temp_penalty': -0.1,
                'humidity_penalty': -0.05,
                'power_penalty': -0.02
            }
        }
        
        reward = sum(info['reward_breakdown'].values())
        done = self.step_count >= 100
        state = np.random.randn(self.state_dim)
        
        return state, reward, done, info

class HVACSimulatorTester:
    """HVAC 시뮬레이터 테스트 클래스"""
    
    def __init__(self, num_zones: int = 4, use_mock: bool = False):
        self.num_zones = num_zones
        
        # 시뮬레이터 초기화
        if SIMULATOR_AVAILABLE and not use_mock:
            self.simulator = AdvancedSmartACSimulator(num_zones)
            print(f"Using real simulator with {num_zones} zones")
        else:
            self.simulator = MockSimulator(num_zones)
            print(f"Using mock simulator with {num_zones} zones")
        
        # 테스트 결과 저장
        self.test_results = {}
        self.performance_metrics = {}
        self.current_data = {
            'step': [],
            'timestamps': [],
            'temperatures': [[] for _ in range(num_zones)],
            'humidities': [[] for _ in range(num_zones)],
            'comfort_scores': [[] for _ in range(num_zones)],
            'co2_levels': [[] for _ in range(num_zones)],
            'dust_levels': [[] for _ in range(num_zones)],
            'power_consumption': [],
            'actions': [],
            'rewards': [],
            'reward_breakdown': {
                'comfort': [],
                'temp_penalty': [],
                'humidity_penalty': [],
                'power_penalty': []
            }
        }
        
        # 시각화 설정
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48CAE4']
        
        # 안전 범위 설정
        self.safety_limits = {
            'temperature': (18.0, 30.0),
            'humidity': (30.0, 80.0),
            'power': 200.0  # W
        }
        
    def create_control_scenarios(self) -> Dict[str, TestScenario]:
        """다양한 제어 시나리오 생성"""
        
        def no_control(step: int, state_info: Dict) -> np.ndarray:
            """제어 없음 (베이스라인)"""
            return np.zeros(14)
        
        def aggressive_cooling(step: int, state_info: Dict) -> np.ndarray:
            """적극적 냉각 시나리오"""
            return np.array([
                1.0,  # 펠티어 최대 냉각
                *[0.8, 0.8, 0.8, 0.8],  # 내부 서보 각도 (높음)
                *[0.6, 0.6, 0.6, 0.6],  # 외부 서보 각도 (중간)
                *[0.9, 0.9, 0.9, 0.9],  # 소형 팬 PWM (높음)
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
            """적응형 제어 시나리오 (PID-like 제어)"""
            if not state_info or 'sensor_readings' not in state_info:
                return gentle_control(step, state_info)
            
            temps = np.array(state_info['sensor_readings']['temperatures'])
            target_temp = 24.0
            
            # PID 계수
            kp, ki, kd = 0.8, 0.1, 0.2
            
            # 온도 오차 계산
            temp_errors = temps - target_temp
            avg_error = np.mean(temp_errors)
            
            # 비례 제어
            proportional = kp * avg_error
            
            # 적분 제어 (간단화)
            if not hasattr(adaptive_control, 'integral'):
                adaptive_control.integral = 0
            adaptive_control.integral += avg_error * 0.5  # dt = 30s를 0.5로 정규화
            integral = ki * adaptive_control.integral
            
            # 미분 제어 (간단화)
            if not hasattr(adaptive_control, 'prev_error'):
                adaptive_control.prev_error = avg_error
            derivative = kd * (avg_error - adaptive_control.prev_error) / 0.5
            adaptive_control.prev_error = avg_error
            
            # PID 출력
            pid_output = proportional + integral + derivative
            cooling_intensity = np.clip(pid_output, -1.0, 1.0)
            
            # 팬 강도는 오차의 절댓값에 비례
            fan_intensity = np.clip(np.abs(avg_error) / 3.0, 0.2, 0.9)
            
            return np.array([
                cooling_intensity,
                *[0.4 + fan_intensity * 0.4] * 4,  # 내부 서보
                *[0.3 + fan_intensity * 0.5] * 4,  # 외부 서보
                *[fan_intensity] * 4,              # 소형 팬
                fan_intensity * 0.8                # 대형 팬
            ])
        
        def zone_differential_control(step: int, state_info: Dict) -> np.ndarray:
            """존별 차등 제어 시나리오"""
            if not state_info or 'sensor_readings' not in state_info:
                return gentle_control(step, state_info)
            
            temps = state_info['sensor_readings']['temperatures']
            action = np.zeros(14)
            
            # 전체 평균 온도에 따른 펠티어 제어
            avg_temp = np.mean(temps)
            action[0] = np.clip((avg_temp - 24.0) / 5.0, -1.0, 1.0)
            
            # 존별 차등 제어
            int_servos, ext_servos, small_fans = [], [], []
            
            for temp in temps:
                if temp > 26.0:  # 뜨거운 존 - 강한 냉각
                    int_servos.append(0.8)
                    ext_servos.append(0.7)
                    small_fans.append(0.9)
                elif temp < 22.0:  # 차가운 존 - 약한 제어
                    int_servos.append(0.2)
                    ext_servos.append(0.2)
                    small_fans.append(0.3)
                else:  # 적정 온도 존 - 중간 제어
                    int_servos.append(0.5)
                    ext_servos.append(0.4)
                    small_fans.append(0.6)
            
            action[1:5] = int_servos    # 내부 서보
            action[5:9] = ext_servos    # 외부 서보
            action[9:13] = small_fans   # 소형 팬
            action[13] = np.mean(small_fans) * 0.8  # 대형 팬
            
            return action
        
        def step_response_test(step: int, state_info: Dict) -> np.ndarray:
            """스텝 응답 테스트 (제어 신호 급변)"""
            if step < 20:
                return np.zeros(14)  # 제어 없음
            elif step < 40:
                return np.ones(14) * 0.8  # 강한 제어
            elif step < 60:
                return np.ones(14) * 0.3  # 약한 제어
            else:
                return np.ones(14) * 0.6  # 중간 제어
        
        def energy_efficient_control(step: int, state_info: Dict) -> np.ndarray:
            """에너지 효율 중심 제어"""
            if not state_info or 'sensor_readings' not in state_info:
                return gentle_control(step, state_info)
            
            temps = np.array(state_info['sensor_readings']['temperatures'])
            comforts = np.array(state_info['comfort_data']['comfort_scores'])
            
            # 쾌적도가 충분히 높으면 제어 강도 감소
            avg_comfort = np.mean(comforts)
            if avg_comfort > 80:
                intensity_factor = 0.3
            elif avg_comfort > 70:
                intensity_factor = 0.6
            else:
                intensity_factor = 0.9
            
            avg_temp = np.mean(temps)
            cooling_need = np.clip((avg_temp - 24.0) / 4.0, -1.0, 1.0)
            
            return np.array([
                cooling_need * intensity_factor,
                *[0.3 + intensity_factor * 0.3] * 4,
                *[0.2 + intensity_factor * 0.4] * 4,
                *[0.4 + intensity_factor * 0.4] * 4,
                0.2 + intensity_factor * 0.4
            ])
        
        return {
            'baseline': TestScenario(
                "No Control (Baseline)",
                60,
                [28.0, 27.5, 28.5, 27.0],
                [65.0, 60.0, 70.0, 55.0],
                no_control,
                "제어 없음 - 베이스라인 성능"
            ),
            'aggressive': TestScenario(
                "Aggressive Cooling",
                80,
                [29.0, 28.5, 29.5, 28.0],
                [70.0, 65.0, 75.0, 60.0],
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
                "Adaptive PID Control",
                100,
                [29.0, 25.0, 23.0, 27.0],
                [70.0, 40.0, 45.0, 65.0],
                adaptive_control,
                "PID 기반 적응형 제어"
            ),
            'differential': TestScenario(
                "Zone Differential Control",
                90,
                [30.0, 22.0, 26.0, 24.0],
                [75.0, 35.0, 55.0, 50.0],
                zone_differential_control,
                "존별 차등 제어"
            ),
            'step_response': TestScenario(
                "Step Response Test",
                80,
                [25.0, 25.0, 25.0, 25.0],
                [50.0, 50.0, 50.0, 50.0],
                step_response_test,
                "스텝 응답 특성 테스트"
            ),
            'energy_efficient': TestScenario(
                "Energy Efficient Control",
                100,
                [27.0, 26.5, 27.5, 26.0],
                [60.0, 55.0, 65.0, 50.0],
                energy_efficient_control,
                "에너지 효율성 중심 제어"
            )
        }
    
    def run_scenario(self, scenario: TestScenario, visualize: bool = True, 
                    save_data: bool = True) -> Dict:
        """시나리오 실행"""
        print(f"\n{'='*60}")
        print(f"시나리오 실행: {scenario.name}")
        print(f"설명: {scenario.description}")
        print(f"지속 시간: {scenario.duration_steps} steps ({scenario.duration_steps * 0.5:.1f} 분)")
        print(f"초기 온도: {scenario.initial_temps}")
        print(f"초기 습도: {scenario.initial_humidity}")
        print(f"{'='*60}")
        
        # 시뮬레이터 초기화
        self.simulator.reset()
        self.simulator.set_initial_state(scenario.initial_temps, scenario.initial_humidity)
        
        # 데이터 초기화
        self._reset_data()
        
        # 이전 스텝의 정보를 저장할 변수
        info = {}
        start_time = time.time()
        
        # 진행률 표시를 위한 설정
        update_interval = max(1, scenario.duration_steps // 10)
        
        for step in range(scenario.duration_steps):
            step_start = time.time()
            
            # 제어 액션 생성
            try:
                action = scenario.control_function(step, info)
                action = np.clip(action, -1.0, 1.0)  # 안전 범위 클리핑
            except Exception as e:
                print(f"Warning: Control function error at step {step}: {e}")
                action = np.zeros(14)  # 안전한 기본값
            
            # 시뮬레이션 스텝 실행
            try:
                state, reward, done, info = self.simulator.step(action)
            except Exception as e:
                print(f"Error during simulation step {step}: {e}")
                break
            
            # 데이터 저장
            self._save_step_data(step, info, action, reward, step_start)
            
            # 진행률 출력
            if step % update_interval == 0 or step == scenario.duration_steps - 1:
                temps = info['sensor_readings']['temperatures']
                avg_temp = np.mean(temps)
                comfort = info['comfort_data']['average_comfort']
                power = info['hardware_states']['total_power']
                
                print(f"Step {step:3d}/{scenario.duration_steps}: "
                      f"Temp={avg_temp:5.1f}°C, "
                      f"Comfort={comfort:5.1f}, "
                      f"Power={power:5.1f}W, "
                      f"Reward={reward:6.3f}")
            
            # 안전 검사
            safety_violations = self._check_safety(info)
            if safety_violations > 0:
                print(f"Warning: {safety_violations} safety violations at step {step}")
            
            if done:
                print(f"Simulation ended early at step {step}")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n시뮬레이션 완료! 소요 시간: {elapsed_time:.2f}초")
        
        # 결과 분석
        results = self._analyze_results(scenario)
        metrics = self._calculate_performance_metrics(scenario)
        
        self.test_results[scenario.name] = results
        self.performance_metrics[scenario.name] = metrics
        
        # 결과 출력
        self._print_summary(scenario.name, results, metrics)
        
        # 시각화
        if visualize:
            self.visualize_results(scenario.name)
        
        # 데이터 저장
        if save_data:
            self._save_scenario_data(scenario.name)
        
        return results
    
    def _reset_data(self):
        """데이터 저장소 초기화"""
        self.current_data = {
            'step': [],
            'timestamps': [],
            'temperatures': [[] for _ in range(self.num_zones)],
            'humidities': [[] for _ in range(self.num_zones)],
            'comfort_scores': [[] for _ in range(self.num_zones)],
            'co2_levels': [[] for _ in range(self.num_zones)],
            'dust_levels': [[] for _ in range(self.num_zones)],
            'power_consumption': [],
            'actions': [],
            'rewards': [],
            'reward_breakdown': {
                'comfort': [],
                'temp_penalty': [],
                'humidity_penalty': [],
                'power_penalty': []
            }
        }
    
    def _save_step_data(self, step: int, info: Dict, action: np.ndarray, 
                       reward: float, timestamp: float):
        """스텝 데이터 저장"""
        self.current_data['step'].append(step)
        self.current_data['timestamps'].append(timestamp)
        
        # 센서 데이터
        sensor_data = info['sensor_readings']
        for i in range(self.num_zones):
            self.current_data['temperatures'][i].append(sensor_data['temperatures'][i])
            self.current_data['humidities'][i].append(sensor_data['humidities'][i])
            self.current_data['comfort_scores'][i].append(info['comfort_data']['comfort_scores'][i])
            self.current_data['co2_levels'][i].append(sensor_data['co2_levels'][i])
            self.current_data['dust_levels'][i].append(sensor_data['dust_levels'][i])
        
        # 시스템 데이터
        self.current_data['power_consumption'].append(info['hardware_states']['total_power'])
        self.current_data['actions'].append(action.copy())
        self.current_data['rewards'].append(reward)
        
        # 보상 분해
        for key, value in info['reward_breakdown'].items():
            self.current_data['reward_breakdown'][key].append(value)
    
    def _check_safety(self, info: Dict) -> int:
        """안전 위반 사항 검사"""
        violations = 0
        
        # 온도 안전 범위
        temps = info['sensor_readings']['temperatures']
        for temp in temps:
            if temp < self.safety_limits['temperature'][0] or temp > self.safety_limits['temperature'][1]:
                violations += 1
        
        # 습도 안전 범위
        humids = info['sensor_readings']['humidities']
        for humid in humids:
            if humid < self.safety_limits['humidity'][0] or humid > self.safety_limits['humidity'][1]:
                violations += 1
        
        # 전력 소비 한계
        power = info['hardware_states']['total_power']
        if power > self.safety_limits['power']:
            violations += 1
        
        return violations
    
    def _analyze_results(self, scenario: TestScenario) -> Dict:
        """결과 분석"""
        data = self.current_data
        
        if not data['step']:
            return {'error': 'No data to analyze'}
        
        # 온도 통계
        temp_stats = {}
        for i in range(self.num_zones):
            if data['temperatures'][i]:
                temps = np.array(data['temperatures'][i])
                temp_stats[f'zone_{i}'] = {
                    'mean': float(np.mean(temps)),
                    'std': float(np.std(temps)),
                    'min': float(np.min(temps)),
                    'max': float(np.max(temps)),
                    'final': float(temps[-1]),
                    'initial': float(temps[0]),
                    'settling_time': self._calculate_settling_time(temps, scenario.target_temp),
                    'overshoot': self._calculate_overshoot(temps, scenario.target_temp)
                }
        
        # 습도 통계
        humid_stats = {}
        for i in range(self.num_zones):
            if data['humidities'][i]:
                humids = np.array(data['humidities'][i])
                humid_stats[f'zone_{i}'] = {
                    'mean': float(np.mean(humids)),
                    'std': float(np.std(humids)),
                    'min': float(np.min(humids)),
                    'max': float(np.max(humids)),
                    'final': float(humids[-1])
                }
        
        # 전체 성능 지표
        total_power = float(np.sum(data['power_consumption'])) if data['power_consumption'] else 0.0
        avg_comfort = float(np.mean([np.mean(data['comfort_scores'][i]) for i in range(self.num_zones) if data['comfort_scores'][i]]))
        avg_reward = float(np.mean(data['rewards'])) if data['rewards'] else 0.0
        
        # 안전성 분석
        safety_analysis = self._analyze_safety()
        
        return {
            'scenario_name': scenario.name,
            'temperature_stats': temp_stats,
            'humidity_stats': humid_stats,
            'total_power_consumption': total_power,
            'average_comfort_score': avg_comfort,
            'average_reward': avg_reward,
            'simulation_steps': len(data['step']),
            'safety_analysis': safety_analysis,
            'reward_breakdown': {
                key: float(np.mean(values)) if values else 0.0 
                for key, values in data['reward_breakdown'].items()
            }
        }
    
    def _calculate_performance_metrics(self, scenario: TestScenario) -> PerformanceMetrics:
        """성능 지표 계산"""
        data = self.current_data
        
        if not data['temperatures'][0]:
            return PerformanceMetrics(0, 0, 0, 0, -1, 0, 0, 0)
        
        # RMSE 계산
        all_temps = []
        all_humids = []
        for i in range(self.num_zones):
            all_temps.extend(data['temperatures'][i])
            all_humids.extend(data['humidities'][i])
        
        temp_rmse = np.sqrt(np.mean([(t - scenario.target_temp)**2 for t in all_temps]))
        humid_rmse = np.sqrt(np.mean([(h - scenario.target_humidity)**2 for h in all_humids]))
        
        # 쾌적도 점수
        comfort_score = np.mean([np.mean(data['comfort_scores'][i]) for i in range(self.num_zones)])
        
        # 에너지 효율성 (쾌적도/전력)
        avg_power = np.mean(data['power_consumption']) if data['power_consumption'] else 1.0
        energy_efficiency = comfort_score / max(avg_power, 1.0)
        
        # 정착 시간
        zone_0_temps = np.array(data['temperatures'][0])
        settling_time = self._calculate_settling_time(zone_0_temps, scenario.target_temp)
        
        # 오버슛
        overshoot_pct = self._calculate_overshoot(zone_0_temps, scenario.target_temp)
        
        # 정상상태 오차
        if len(zone_0_temps) > 10:
            steady_state_error = abs(np.mean(zone_0_temps[-10:]) - scenario.target_temp)
        else:
            steady_state_error = abs(zone_0_temps[-1] - scenario.target_temp) if len(zone_0_temps) > 0 else 0
        
        # 안전 위반 횟수
        safety_violations = sum([self._check_safety({'sensor_readings': {'temperatures': [data['temperatures'][i][j] for i in range(self.num_zones)], 'humidities': [data['humidities'][i][j] for i in range(self.num_zones)]}, 'hardware_states': {'total_power': data['power_consumption'][j]}}) for j in range(len(data['step']))])
        
        return PerformanceMetrics(
            temperature_rmse=temp_rmse,
            humidity_rmse=humid_rmse,
            comfort_score=comfort_score,
            energy_efficiency=energy_efficiency,
            settling_time=settling_time,
            overshoot_percentage=overshoot_pct,
            steady_state_error=steady_state_error,
            safety_violations=safety_violations
        )
    
    def _calculate_settling_time(self, values: np.ndarray, target: float, 
                               tolerance: float = 0.5, window: int = 5) -> int:
        """정착 시간 계산"""
        if len(values) <= window:
            return -1
        
        for i in range(len(values) - window):
            if all(abs(values[j] - target) <= tolerance for j in range(i, i + window)):
                return i
        return -1
    
    def _calculate_overshoot(self, values: np.ndarray, target: float) -> float:
        """오버슛 계산 (백분율)"""
        if len(values) < 2:
            return 0.0
        
        initial_value = values[0]
        if initial_value == target:
            return 0.0
        
        # 냉각의 경우 (초기 온도 > 목표 온도)
        if initial_value > target:
            min_value = np.min(values)
            if min_value < target:
                overshoot = abs(min_value - target) / abs(initial_value - target) * 100
                return overshoot
        # 가열의 경우 (초기 온도 < 목표 온도)
        else:
            max_value = np.max(values)
            if max_value > target:
                overshoot = abs(max_value - target) / abs(initial_value - target) * 100
                return overshoot
        
        return 0.0
    
    def _analyze_safety(self) -> Dict:
        """안전성 분석"""
        data = self.current_data
        violations = {
            'temperature_violations': 0,
            'humidity_violations': 0,
            'power_violations': 0,
            'total_violations': 0
        }
        
        for step in range(len(data['step'])):
            # 온도 위반
            for i in range(self.num_zones):
                if data['temperatures'][i]:
                    temp = data['temperatures'][i][step] if step < len(data['temperatures'][i]) else 0
                    if temp < self.safety_limits['temperature'][0] or temp > self.safety_limits['temperature'][1]:
                        violations['temperature_violations'] += 1
            
            # 습도 위반
            for i in range(self.num_zones):
                if data['humidities'][i]:
                    humid = data['humidities'][i][step] if step < len(data['humidities'][i]) else 0
                    if humid < self.safety_limits['humidity'][0] or humid > self.safety_limits['humidity'][1]:
                        violations['humidity_violations'] += 1
            
            # 전력 위반
            if step < len(data['power_consumption']):
                power = data['power_consumption'][step]
                if power > self.safety_limits['power']:
                    violations['power_violations'] += 1
        
        violations['total_violations'] = (violations['temperature_violations'] + 
                                        violations['humidity_violations'] + 
                                        violations['power_violations'])
        
        return violations
    
    def _print_summary(self, scenario_name: str, results: Dict, metrics: PerformanceMetrics):
        """결과 요약 출력"""
        print(f"\n{'='*60}")
        print(f"시나리오 결과 요약: {scenario_name}")
        print(f"{'='*60}")
        
        if 'error' in results:
            print(f"오류: {results['error']}")
            return
        
        # 온도 성능
        print(f"\n📊 온도 성능:")
        for zone, stats in results['temperature_stats'].items():
            print(f"  {zone.upper()}: 평균={stats['mean']:.1f}°C, "
                  f"표준편차={stats['std']:.2f}, 최종={stats['final']:.1f}°C")
        
        # 습도 성능
        print(f"\n💧 습도 성능:")
        for zone, stats in results['humidity_stats'].items():
            print(f"  {zone.upper()}: 평균={stats['mean']:.1f}%, "
                  f"표준편차={stats['std']:.2f}, 최종={stats['final']:.1f}%")
        
        # 전체 성능 지표
        print(f"\n🎯 전체 성능 지표:")
        print(f"  온도 RMSE: {metrics.temperature_rmse:.2f}°C")
        print(f"  습도 RMSE: {metrics.humidity_rmse:.2f}%")
        print(f"  평균 쾌적도: {metrics.comfort_score:.1f}/100")
        print(f"  에너지 효율성: {metrics.energy_efficiency:.3f} (쾌적도/W)")
        print(f"  정착 시간: {metrics.settling_time} steps ({metrics.settling_time * 0.5:.1f} 분)" if metrics.settling_time >= 0 else "  정착 시간: 달성 안됨")
        print(f"  오버슛: {metrics.overshoot_percentage:.1f}%")
        print(f"  정상상태 오차: {metrics.steady_state_error:.2f}°C")
        
        # 에너지 소비
        print(f"\n⚡ 에너지 소비:")
        print(f"  총 전력 소비: {results['total_power_consumption']:.1f} W·steps")
        print(f"  평균 전력: {results['total_power_consumption']/max(results['simulation_steps'], 1):.1f} W")
        
        # 안전성
        print(f"\n🛡️ 안전성:")
        safety = results['safety_analysis']
        print(f"  총 위반 횟수: {safety['total_violations']}")
        print(f"  온도 위반: {safety['temperature_violations']}")
        print(f"  습도 위반: {safety['humidity_violations']}")
        print(f"  전력 위반: {safety['power_violations']}")
        
        # 보상 분해
        print(f"\n🏆 보상 분해:")
        for component, value in results['reward_breakdown'].items():
            print(f"  {component}: {value:.3f}")
        print(f"  총 평균 보상: {results['average_reward']:.3f}")
    
    def visualize_results(self, scenario_name: str, save_plots: bool = True):
        """결과 시각화"""
        data = self.current_data
        
        if not data['step']:
            print("시각화할 데이터가 없습니다.")
            return
        
        # 시간 축 생성 (분 단위)
        time_minutes = np.array(data['step']) * 0.5
        
        # 메인 플롯 생성 (2x3 서브플롯)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'HVAC 시뮬레이션 결과: {scenario_name}', fontsize=16, fontweight='bold')
        
        # 1. 온도 변화
        ax1 = axes[0, 0]
        for i in range(self.num_zones):
            if data['temperatures'][i]:
                ax1.plot(time_minutes, data['temperatures'][i], 
                        label=f'Zone {i+1}', color=self.colors[i], linewidth=2)
        ax1.axhline(y=24, color='red', linestyle='--', alpha=0.7, label='목표 온도')
        ax1.fill_between(time_minutes, 22, 26, alpha=0.2, color='green', label='쾌적 범위')
        ax1.set_xlabel('시간 (분)')
        ax1.set_ylabel('온도 (°C)')
        ax1.set_title('존별 온도 변화')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 습도 변화
        ax2 = axes[0, 1]
        for i in range(self.num_zones):
            if data['humidities'][i]:
                ax2.plot(time_minutes, data['humidities'][i], 
                        label=f'Zone {i+1}', color=self.colors[i], linewidth=2)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='목표 습도')
        ax2.fill_between(time_minutes, 40, 60, alpha=0.2, color='blue', label='쾌적 범위')
        ax2.set_xlabel('시간 (분)')
        ax2.set_ylabel('상대습도 (%)')
        ax2.set_title('존별 습도 변화')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 쾌적도 점수
        ax3 = axes[0, 2]
        for i in range(self.num_zones):
            if data['comfort_scores'][i]:
                ax3.plot(time_minutes, data['comfort_scores'][i], 
                        label=f'Zone {i+1}', color=self.colors[i], linewidth=2)
        ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='우수 기준')
        ax3.fill_between(time_minutes, 70, 100, alpha=0.2, color='orange', label='양호 범위')
        ax3.set_xlabel('시간 (분)')
        ax3.set_ylabel('쾌적도 점수')
        ax3.set_title('존별 쾌적도 변화')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 전력 소비
        ax4 = axes[1, 0]
        if data['power_consumption']:
            ax4.plot(time_minutes, data['power_consumption'], 
                    color='red', linewidth=2, label='총 전력')
            ax4.axhline(y=self.safety_limits['power'], color='red', 
                       linestyle='--', alpha=0.7, label='안전 한계')
        ax4.set_xlabel('시간 (분)')
        ax4.set_ylabel('전력 소비 (W)')
        ax4.set_title('전력 소비 패턴')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 제어 액션 (주요 액션만)
        ax5 = axes[1, 1]
        if data['actions']:
            actions_array = np.array(data['actions'])
            ax5.plot(time_minutes, actions_array[:, 0], label='Peltier', linewidth=2)
            ax5.plot(time_minutes, np.mean(actions_array[:, 1:5], axis=1), 
                    label='평균 내부 서보', linewidth=2)
            ax5.plot(time_minutes, np.mean(actions_array[:, 9:13], axis=1), 
                    label='평균 소형 팬', linewidth=2)
        ax5.set_xlabel('시간 (분)')
        ax5.set_ylabel('제어 신호')
        ax5.set_title('주요 제어 액션')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-1.1, 1.1)
        
        # 6. 보상 구성 요소
        ax6 = axes[1, 2]
        if data['reward_breakdown']['comfort']:
            for component, values in data['reward_breakdown'].items():
                if values:
                    ax6.plot(time_minutes, values, label=component.replace('_', ' ').title(), linewidth=2)
        ax6.plot(time_minutes, data['rewards'], label='총 보상', 
                color='black', linewidth=3, alpha=0.8)
        ax6.set_xlabel('시간 (분)')
        ax6.set_ylabel('보상 값')
        ax6.set_title('보상 구성 요소')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hvac_test_{scenario_name.replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"플롯 저장됨: {filename}")
        
        plt.show()
        
        # 추가 상세 분석 플롯
        self._create_detailed_analysis_plots(scenario_name, save_plots)
    
    def _create_detailed_analysis_plots(self, scenario_name: str, save_plots: bool = True):
        """상세 분석 플롯 생성"""
        data = self.current_data
        time_minutes = np.array(data['step']) * 0.5
        
        # 상세 분석 플롯 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'상세 분석: {scenario_name}', fontsize=14, fontweight='bold')
        
        # 1. 온도 분포 (박스 플롯)
        ax1 = axes[0, 0]
        temp_data = [data['temperatures'][i] for i in range(self.num_zones) if data['temperatures'][i]]
        if temp_data:
            ax1.boxplot(temp_data, labels=[f'Zone {i+1}' for i in range(len(temp_data))])
            ax1.axhline(y=24, color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('온도 (°C)')
        ax1.set_title('존별 온도 분포')
        ax1.grid(True, alpha=0.3)
        
        # 2. 에너지 효율성 시계열
        ax2 = axes[0, 1]
        if data['power_consumption'] and any(data['comfort_scores'][i] for i in range(self.num_zones)):
            avg_comfort = [np.mean([data['comfort_scores'][i][j] if j < len(data['comfort_scores'][i]) else 0 
                                  for i in range(self.num_zones)]) for j in range(len(time_minutes))]
            efficiency = [c/max(p, 1) for c, p in zip(avg_comfort, data['power_consumption'])]
            ax2.plot(time_minutes, efficiency, color='green', linewidth=2)
        ax2.set_xlabel('시간 (분)')
        ax2.set_ylabel('효율성 (쾌적도/W)')
        ax2.set_title('에너지 효율성 변화')
        ax2.grid(True, alpha=0.3)
        
        # 3. CO2 농도
        ax3 = axes[1, 0]
        for i in range(self.num_zones):
            if data['co2_levels'][i]:
                ax3.plot(time_minutes, data['co2_levels'][i], 
                        label=f'Zone {i+1}', color=self.colors[i], linewidth=2)
        ax3.axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='권장 한계')
        ax3.set_xlabel('시간 (분)')
        ax3.set_ylabel('CO2 농도 (ppm)')
        ax3.set_title('CO2 농도 변화')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 미세먼지 농도
        ax4 = axes[1, 1]
        for i in range(self.num_zones):
            if data['dust_levels'][i]:
                ax4.plot(time_minutes, data['dust_levels'][i], 
                        label=f'Zone {i+1}', color=self.colors[i], linewidth=2)
        ax4.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='나쁨 기준')
        ax4.set_xlabel('시간 (분)')
        ax4.set_ylabel('미세먼지 (μg/m³)')
        ax4.set_title('미세먼지 농도 변화')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hvac_detailed_{scenario_name.replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"상세 플롯 저장됨: {filename}")
        
        plt.show()
    
    def _save_scenario_data(self, scenario_name: str):
        """시나리오 데이터를 파일로 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON 형태로 저장 (작은 데이터)
        json_data = {
            'scenario_name': scenario_name,
            'timestamp': timestamp,
            'summary': self.test_results.get(scenario_name, {}),
            'metrics': asdict(self.performance_metrics.get(scenario_name, PerformanceMetrics(0,0,0,0,0,0,0,0)))
        }
        
        json_filename = f"hvac_summary_{scenario_name.replace(' ', '_')}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # CSV 형태로 상세 데이터 저장
        csv_data = []
        for i, step in enumerate(self.current_data['step']):
            row = {
                'step': step,
                'time_minutes': step * 0.5,
            }
            
            # 온도 데이터
            for j in range(self.num_zones):
                if i < len(self.current_data['temperatures'][j]):
                    row[f'temp_zone_{j+1}'] = self.current_data['temperatures'][j][i]
                    row[f'humidity_zone_{j+1}'] = self.current_data['humidities'][j][i]
                    row[f'comfort_zone_{j+1}'] = self.current_data['comfort_scores'][j][i]
                    row[f'co2_zone_{j+1}'] = self.current_data['co2_levels'][j][i]
                    row[f'dust_zone_{j+1}'] = self.current_data['dust_levels'][j][i]
            
            # 시스템 데이터
            if i < len(self.current_data['power_consumption']):
                row['power_consumption'] = self.current_data['power_consumption'][i]
            if i < len(self.current_data['rewards']):
                row['reward'] = self.current_data['rewards'][i]
            
            # 액션 데이터
            if i < len(self.current_data['actions']):
                action = self.current_data['actions'][i]
                action_names = ['peltier', 'int_servo_1', 'int_servo_2', 'int_servo_3', 'int_servo_4',
                               'ext_servo_1', 'ext_servo_2', 'ext_servo_3', 'ext_servo_4',
                               'fan_1', 'fan_2', 'fan_3', 'fan_4', 'large_fan']
                for j, name in enumerate(action_names):
                    row[f'action_{name}'] = action[j]
            
            csv_data.append(row)
        
        csv_filename = f"hvac_data_{scenario_name.replace(' ', '_')}_{timestamp}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        
        print(f"데이터 저장 완료:")
        print(f"  요약: {json_filename}")
        print(f"  상세: {csv_filename}")
    
    def run_all_scenarios(self, visualize: bool = True, save_data: bool = True):
        """모든 시나리오 실행"""
        scenarios = self.create_control_scenarios()
        
        print(f"\n{'='*80}")
        print(f"HVAC 시뮬레이터 종합 테스트 시작")
        print(f"총 {len(scenarios)} 개 시나리오 실행")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        for i, (key, scenario) in enumerate(scenarios.items(), 1):
            print(f"\n[{i}/{len(scenarios)}] 시나리오 실행 중...")
            try:
                self.run_scenario(scenario, visualize=visualize, save_data=save_data)
            except Exception as e:
                print(f"시나리오 '{scenario.name}' 실행 중 오류 발생: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"전체 테스트 완료! 총 소요 시간: {total_time:.1f}초")
        print(f"{'='*80}")
        
        # 종합 비교 분석
        self.create_comparison_analysis()
    
    def create_comparison_analysis(self):
        """시나리오 간 비교 분석"""
        if len(self.performance_metrics) < 2:
            print("비교할 시나리오가 부족합니다.")
            return
        
        print(f"\n{'='*60}")
        print("시나리오 비교 분석")
        print(f"{'='*60}")
        
        # 성능 메트릭 비교 테이블
        metrics_df = pd.DataFrame({
            name: asdict(metrics) for name, metrics in self.performance_metrics.items()
        }).T
        
        print("\n📊 성능 메트릭 비교:")
        print(metrics_df.round(3))
        
        # 최고 성능 시나리오 식별
        print(f"\n🏆 최고 성능 시나리오:")
        print(f"  최저 온도 RMSE: {metrics_df['temperature_rmse'].idxmin()}")
        print(f"  최고 쾌적도: {metrics_df['comfort_score'].idxmax()}")
        print(f"  최고 에너지 효율: {metrics_df['energy_efficiency'].idxmax()}")
        print(f"  최소 안전 위반: {metrics_df['safety_violations'].idxmin()}")
        
        # 시각화
        self._create_comparison_plots()
        
        # CSV로 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_filename = f"hvac_comparison_{timestamp}.csv"
        metrics_df.to_csv(comparison_filename)
        print(f"\n비교 결과 저장: {comparison_filename}")
    
    def _create_comparison_plots(self):
        """비교 시각화 생성"""
        if len(self.performance_metrics) < 2:
            return
        
        # 레이더 차트로 성능 비교
        metrics_names = ['Temperature RMSE', 'Comfort Score', 'Energy Efficiency', 
                        'Settling Time', 'Safety Score']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 성능 메트릭 막대 그래프
        scenario_names = list(self.performance_metrics.keys())
        x_pos = np.arange(len(scenario_names))
        
        comfort_scores = [m.comfort_score for m in self.performance_metrics.values()]
        energy_effs = [m.energy_efficiency for m in self.performance_metrics.values()]
        
        ax1_twin = ax1.twinx()
        bars1 = ax1.bar(x_pos - 0.2, comfort_scores, 0.4, label='쾌적도', alpha=0.8)
        bars2 = ax1_twin.bar(x_pos + 0.2, energy_effs, 0.4, label='에너지 효율', 
                            color='orange', alpha=0.8)
        
        ax1.set_xlabel('시나리오')
        ax1.set_ylabel('쾌적도 점수', color='blue')
        ax1_twin.set_ylabel('에너지 효율', color='orange')
        ax1.set_title('시나리오별 쾌적도 vs 에너지 효율')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.split()[0] for name in scenario_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 안전성 비교
        safety_scores = [max(0, 100 - m.safety_violations) for m in self.performance_metrics.values()]
        temp_rmses = [m.temperature_rmse for m in self.performance_metrics.values()]
        
        scatter = ax2.scatter(temp_rmses, safety_scores, s=100, alpha=0.7, 
                             c=comfort_scores, cmap='viridis')
        
        for i, name in enumerate(scenario_names):
            ax2.annotate(name.split()[0], (temp_rmses[i], safety_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('온도 RMSE')
        ax2.set_ylabel('안전성 점수')
        ax2.set_title('정확도 vs 안전성 (색상: 쾌적도)')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax2, label='쾌적도')
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"hvac_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """메인 실행 함수"""
    print("HVAC 시뮬레이터 테스트 프로그램")
    print("=" * 50)
    
    # 테스터 초기화
    tester = HVACSimulatorTester(num_zones=4, use_mock=not SIMULATOR_AVAILABLE)
    
    # 사용자 선택
    print("\n실행 옵션을 선택하세요:")
    print("1. 모든 시나리오 실행")
    print("2. 특정 시나리오 실행") 
    print("3. 빠른 테스트 (시각화 제외)")
    
    try:
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == '1':
            tester.run_all_scenarios(visualize=True, save_data=True)
        
        elif choice == '2':
            scenarios = tester.create_control_scenarios()
            print("\n사용 가능한 시나리오:")
            for i, (key, scenario) in enumerate(scenarios.items(), 1):
                print(f"{i}. {scenario.name} - {scenario.description}")
            
            scenario_choice = int(input("\n시나리오 번호 선택: ")) - 1
            scenario_key = list(scenarios.keys())[scenario_choice]
            tester.run_scenario(scenarios[scenario_key], visualize=True, save_data=True)
        
        elif choice == '3':
            # 빠른 테스트용 짧은 시나리오
            quick_scenarios = {
                'adaptive': tester.create_control_scenarios()['adaptive'],
                'gentle': tester.create_control_scenarios()['gentle']
            }
            
            for scenario in quick_scenarios.values():
                scenario.duration_steps = 30  # 짧게 설정
            
            for scenario in quick_scenarios.values():
                tester.run_scenario(scenario, visualize=False, save_data=False)
            
            tester.performance_metrics = {k: v for k, v in tester.performance_metrics.items() 
                                        if k in ['Adaptive PID Control', 'Gentle Control']}
            tester.create_comparison_analysis()
        
        else:
            print("잘못된 선택입니다.")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()