"""
HVAC 시뮬레이터 단계별 테스트 및 분석 도구
==============================================

이 코드는 시뮬레이터의 동작을 단계별로 관찰하고 분석하는 도구입니다.
"""

from simulator.environment import AdvancedSmartACSimulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """테스트 결과를 저장하는 데이터 클래스"""
    step: int
    action: np.ndarray
    state_vector: np.ndarray
    sensor_readings: Dict
    comfort_data: Dict
    hardware_states: Dict
    reward: float
    reward_breakdown: Dict
    physics_state: Dict

class HVACStepTester:
    """HVAC 시뮬레이터 단계별 테스터"""
    
    def __init__(self, num_zones: int = 4):
        self.simulator = AdvancedSmartACSimulator(num_zones)
        self.num_zones = num_zones
        self.test_history: List[TestResult] = []
        self.current_step = 0
        
        # 액션 구성 가이드
        self.action_guide = {
            'peltier_control': {'index': 0, 'range': (-1, 1), 'description': '펠티어 제어 (-1:OFF, +1:최대냉각)'},
            'internal_servo': {'index': slice(1, 5), 'range': (-1, 1), 'description': '내부 서보 각도 (각 존별)'},
            'external_servo': {'index': slice(5, 9), 'range': (-1, 1), 'description': '외부 서보 각도 (각 존별)'},
            'small_fan_pwm': {'index': slice(9, 13), 'range': (-1, 1), 'description': '소형 팬 PWM (각 존별)'},
            'large_fan_pwm': {'index': 13, 'range': (-1, 1), 'description': '대형 팬 PWM'}
        }
        
        print("🔧 HVAC 시뮬레이터 단계별 테스터 초기화 완료")
        print(f"   • 존 개수: {num_zones}")
        print(f"   • 액션 차원: {self.simulator.action_dim}")
        print(f"   • 상태 차원: {self.simulator.state_dim}")
    
    def reset_simulator(self, initial_temps: Optional[List[float]] = None, 
                       initial_humidity: Optional[List[float]] = None):
        """시뮬레이터 초기화"""
        print("\n🔄 시뮬레이터 리셋")
        
        self.simulator.reset()
        
        if initial_temps and initial_humidity:
            self.simulator.set_initial_state(initial_temps, initial_humidity)
            print(f"   • 초기 온도: {initial_temps}")
            print(f"   • 초기 습도: {initial_humidity}")
        
        self.test_history.clear()
        self.current_step = 0
        
        # 초기 상태 출력
        self._print_current_state()
    
    def create_action(self, peltier: float = 0.0, 
                     internal_servos: List[float] = None,
                     external_servos: List[float] = None,
                     small_fans: List[float] = None,
                     large_fan: float = 0.0) -> np.ndarray:
        """사용자 친화적 액션 생성"""
        
        if internal_servos is None:
            internal_servos = [0.0] * self.num_zones
        if external_servos is None:
            external_servos = [0.0] * self.num_zones
        if small_fans is None:
            small_fans = [0.0] * self.num_zones
            
        action = np.zeros(14)
        action[0] = np.clip(peltier, -1, 1)
        action[1:5] = np.clip(internal_servos, -1, 1)
        action[5:9] = np.clip(external_servos, -1, 1)
        action[9:13] = np.clip(small_fans, -1, 1)
        action[13] = np.clip(large_fan, -1, 1)
        
        return action
    
    def step_once(self, action: np.ndarray, verbose: bool = True) -> TestResult:
        """단일 스텝 실행"""
        if verbose:
            print(f"\n⚡ Step {self.current_step} 실행")
            self._print_action_info(action)
        
        try:
            state_vector, reward, done, info = self.simulator.step(action)
            
            # 결과 저장
            result = TestResult(
                step=self.current_step,
                action=action.copy(),
                state_vector=state_vector.copy(),
                sensor_readings=info['sensor_readings'].copy(),
                comfort_data=info['comfort_data'].copy(),
                hardware_states=info['hardware_states'].copy(),
                reward=reward,
                reward_breakdown=info['reward_breakdown'].copy(),
                physics_state=self.simulator.physics_sim.get_current_state().copy()
            )
            
            self.test_history.append(result)
            self.current_step += 1
            
            if verbose:
                self._print_step_result(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 스텝 실행 중 오류 발생: {e}")
            raise e
    
    def run_multiple_steps(self, actions: List[np.ndarray], verbose: bool = True) -> List[TestResult]:
        """여러 스텝 연속 실행"""
        print(f"\n🚀 {len(actions)}개 스텝 연속 실행")
        
        results = []
        for i, action in enumerate(actions):
            print(f"\n--- Step {i+1}/{len(actions)} ---")
            result = self.step_once(action, verbose=verbose)
            results.append(result)
        
        return results
    
    def run_constant_control(self, action: np.ndarray, num_steps: int, verbose: bool = False) -> List[TestResult]:
        """일정한 제어 입력으로 여러 스텝 실행"""
        print(f"\n🔁 일정 제어로 {num_steps}스텝 실행")
        self._print_action_info(action)
        
        results = []
        for i in range(num_steps):
            if verbose or i % 10 == 0:
                print(f"\nStep {i+1}/{num_steps}")
            result = self.step_once(action, verbose=verbose)
            results.append(result)
        
        return results
    
    def _print_current_state(self):
        """현재 상태 출력"""
        physics_state = self.simulator.physics_sim.get_current_state()
        
        print("📊 현재 시뮬레이터 상태:")
        print(f"   • 온도: {[f'{t:.1f}°C' for t in physics_state['temperatures']]}")
        print(f"   • 습도: {[f'{h:.1f}%' for h in physics_state['humidities']]}")
        print(f"   • CO2: {[f'{c:.0f}ppm' for c in physics_state['co2_levels']]}")
        print(f"   • 먼지: {[f'{d:.1f}μg/m³' for d in physics_state['dust_levels']]}")
        
        print("🔧 하드웨어 상태:")
        print(f"   • 내부 서보: {[f'{s.current_angle:.1f}°' for s in self.simulator.internal_servos]}")
        print(f"   • 외부 서보: {[f'{s.current_angle:.1f}°' for s in self.simulator.external_servos]}")
        print(f"   • 소형 팬: {[f'{f.current_rpm:.0f}rpm' for f in self.simulator.small_fans]}")
        print(f"   • 대형 팬: {self.simulator.large_fan.current_rpm:.0f}rpm")
    
    def _print_action_info(self, action: np.ndarray):
        """액션 정보 출력"""
        print("🎮 제어 입력:")
        print(f"   • 펠티어: {action[0]:+.2f}")
        print(f"   • 내부서보: {[f'{a:+.2f}' for a in action[1:5]]}")
        print(f"   • 외부서보: {[f'{a:+.2f}' for a in action[5:9]]}")
        print(f"   • 소형팬: {[f'{a:+.2f}' for a in action[9:13]]}")
        print(f"   • 대형팬: {action[13]:+.2f}")
    
    def _print_step_result(self, result: TestResult):
        """스텝 결과 출력"""
        sr = result.sensor_readings
        cd = result.comfort_data
        hs = result.hardware_states
        rb = result.reward_breakdown
        
        print("📈 결과:")
        print(f"   • 온도 변화: {[f'{t:.1f}°C' for t in sr['temperatures']]}")
        print(f"   • 습도 변화: {[f'{h:.1f}%' for h in sr['humidities']]}")
        print(f"   • 쾌적도: {[f'{c:.1f}' for c in cd['comfort_scores']]} (평균: {cd['average_comfort']:.1f})")
        print(f"   • 소비전력: {hs['total_power']:.1f}W")
        print(f"   • 보상: {result.reward:.3f} (쾌적:{rb['comfort']:.3f}, 전력:{rb['power_penalty']:.3f})")
    
    def analyze_temperature_trend(self, steps: int = None) -> Dict:
        """온도 변화 추세 분석"""
        if not self.test_history:
            print("❌ 분석할 데이터가 없습니다.")
            return {}
        
        if steps is None:
            steps = len(self.test_history)
        
        data = self.test_history[-steps:]
        
        analysis = {}
        for zone in range(self.num_zones):
            temps = [r.sensor_readings['temperatures'][zone] for r in data]
            analysis[f'zone_{zone}'] = {
                'initial': temps[0],
                'final': temps[-1],
                'change': temps[-1] - temps[0],
                'max': max(temps),
                'min': min(temps),
                'avg': sum(temps) / len(temps),
                'trend': 'cooling' if temps[-1] < temps[0] else 'heating'
            }
        
        return analysis
    
    def plot_results(self, steps: int = None, save_path: str = None):
        """결과 시각화"""
        if not self.test_history:
            print("❌ 시각화할 데이터가 없습니다.")
            return
        
        if steps is None:
            steps = len(self.test_history)
        
        data = self.test_history[-steps:]
        step_nums = [r.step for r in data]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'HVAC Simulation Result (Recent {len(data)}Steps)', fontsize=16)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. 온도 변화
        ax = axes[0, 0]
        for zone in range(self.num_zones):
            temps = [r.sensor_readings['temperatures'][zone] for r in data]
            ax.plot(step_nums, temps, color=colors[zone], label=f'Zone {zone+1}', linewidth=2)
        ax.axhline(y=24.0, color='red', linestyle='--', alpha=0.7, label='Target')
        ax.set_xlabel('Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 습도 변화
        ax = axes[0, 1]
        for zone in range(self.num_zones):
            humids = [r.sensor_readings['humidities'][zone] for r in data]
            ax.plot(step_nums, humids, color=colors[zone], label=f'Zone {zone+1}', linewidth=2)
        ax.axhline(y=50.0, color='red', linestyle='--', alpha=0.7, label='Target')
        ax.set_xlabel('Step')
        ax.set_ylabel('Humidity (%)')
        ax.set_title('Humidity Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 쾌적도
        ax = axes[0, 2]
        for zone in range(self.num_zones):
            comforts = [r.comfort_data['comfort_scores'][zone] for r in data]
            ax.plot(step_nums, comforts, color=colors[zone], label=f'Zone {zone+1}', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Comfort Score')
        ax.set_title('Comfort Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 전력 소비
        ax = axes[1, 0]
        powers = [r.hardware_states['total_power'] for r in data]
        ax.plot(step_nums, powers, color='orange', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Consumption')
        ax.grid(True, alpha=0.3)
        
        # 5. 보상
        ax = axes[1, 1]
        rewards = [r.reward for r in data]
        ax.plot(step_nums, rewards, color='green', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Over Time')
        ax.grid(True, alpha=0.3)
        
        # 6. 펠티어 제어 vs 온도
        ax = axes[1, 2]
        peltier_controls = [r.action[0] for r in data]
        avg_temps = [sum(r.sensor_readings['temperatures']) / self.num_zones for r in data]
        
        ax2 = ax.twinx()
        line1 = ax.plot(step_nums, peltier_controls, 'b-', linewidth=2, label='Peltier Control')
        line2 = ax2.plot(step_nums, avg_temps, 'r-', linewidth=2, label='Avg Temperature')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Peltier Control', color='b')
        ax2.set_ylabel('Temperature (°C)', color='r')
        ax.set_title('Peltier Control vs Avg Temperature')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 그래프 저장: {save_path}")
        
        plt.show()
    
    def export_data(self, filename: str = None) -> str:
        """데이터 CSV 내보내기"""
        if not self.test_history:
            print("❌ 내보낼 데이터가 없습니다.")
            return ""
        
        if filename is None:
            filename = f"hvac_test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 데이터프레임 생성
        rows = []
        for result in self.test_history:
            row = {
                'step': result.step,
                'reward': result.reward,
                'total_power': result.hardware_states['total_power'],
            }
            
            # 액션 데이터
            row['peltier_control'] = result.action[0]
            for i in range(self.num_zones):
                row[f'internal_servo_{i}'] = result.action[1+i]
                row[f'external_servo_{i}'] = result.action[5+i]
                row[f'small_fan_{i}'] = result.action[9+i]
            row['large_fan'] = result.action[13]
            
            # 센서 데이터
            for i in range(self.num_zones):
                row[f'temperature_{i}'] = result.sensor_readings['temperatures'][i]
                row[f'humidity_{i}'] = result.sensor_readings['humidities'][i]
                row[f'comfort_{i}'] = result.comfort_data['comfort_scores'][i]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"📁 데이터 내보내기 완료: {filename}")
        print(f"   • 총 {len(rows)}개 스텝 데이터")
        print(f"   • 컬럼: {len(df.columns)}개")
        
        return filename
    
    def print_action_guide(self):
        """액션 구성 가이드 출력"""
        print("\n📋 액션 벡터 구성 가이드:")
        print("=" * 50)
        for name, info in self.action_guide.items():
            idx = info['index']
            if isinstance(idx, slice):
                idx_str = f"[{idx.start}:{idx.stop}]"
            else:
                idx_str = f"[{idx}]"
            print(f"{name:15} {idx_str:8} {str(info['range']):10} - {info['description']}")
        
        print("\n💡 사용 예시:")
        print("action = tester.create_action(")
        print("    peltier=0.5,                    # 50% 냉각")
        print("    internal_servos=[0.2, 0.3, 0.4, 0.5],  # 각 존별 내부 서보")
        print("    small_fans=[0.6, 0.7, 0.8, 0.9],       # 각 존별 소형 팬")
        print("    large_fan=0.4                   # 대형 팬")
        print(")")


# ========================================
# 사용 예시 및 테스트 시나리오들
# ========================================

def example_basic_test():
    """기본 테스트 예시"""
    print("🧪 기본 테스트 시작")
    
    # 테스터 초기화
    tester = HVACStepTester()
    
    # 초기 상태 설정
    tester.reset_simulator(
        initial_temps=[28.0, 27.0, 26.5, 28.5],
        initial_humidity=[65.0, 60.0, 55.0, 70.0]
    )
    
    # 액션 가이드 출력
    tester.print_action_guide()
    
    # 단일 스텝 테스트
    action1 = tester.create_action(peltier=0.8, small_fans=[0.5, 0.6, 0.7, 0.8])
    result1 = tester.step_once(action1)
    
    # 여러 스텝 테스트
    actions = [
        tester.create_action(peltier=0.2, small_fans=[0.3, 0.3, 0.3, 0.3]),
        tester.create_action(peltier=0.5, small_fans=[0.6, 0.6, 0.6, 0.6]),
        tester.create_action(peltier=0.8, small_fans=[0.9, 0.9, 0.9, 0.9]),
    ]
    
    results = tester.run_multiple_steps(actions)
    
    # 온도 추세 분석
    analysis = tester.analyze_temperature_trend()
    print("\n📊 온도 변화 분석:")
    for zone, data in analysis.items():
        print(f"   {zone}: {data['initial']:.1f}°C → {data['final']:.1f}°C ({data['change']:+.1f}°C, {data['trend']})")
    
    # 결과 시각화
    tester.plot_results()
    
    # 데이터 내보내기
    tester.export_data()

def example_cooling_test():
    """냉각 성능 테스트"""
    print("🧪 냉각 성능 테스트")
    
    tester = HVACStepTester()
    
    # 높은 초기 온도로 설정
    tester.reset_simulator(
        initial_temps=[30.0, 29.5, 31.0, 30.5],
        initial_humidity=[70.0, 65.0, 75.0, 68.0]
    )
    
    # 점진적 냉각 테스트
    print("\n🔸 점진적 냉각 테스트 (20스텝)")
    cooling_action = tester.create_action(
        peltier=0.7,
        internal_servos=[0.6, 0.7, 0.8, 0.6],
        small_fans=[0.8, 0.8, 0.8, 0.8],
        large_fan=0.6
    )
    
    results = tester.run_constant_control(cooling_action, 20, verbose=False)
    
    # 결과 분석
    analysis = tester.analyze_temperature_trend(20)
    print("\n📊 냉각 효과 분석:")
    for zone, data in analysis.items():
        cooling_rate = data['change'] / 20  # per step
        print(f"   {zone}: {data['change']:+.1f}°C 변화 ({cooling_rate:+.2f}°C/step)")
    
    tester.plot_results(steps=20)

def example_step_response_test():
    """스텝 응답 테스트"""
    print("🧪 스텝 응답 테스트")
    
    tester = HVACStepTester()
    tester.reset_simulator(
        initial_temps=[25.0, 25.0, 25.0, 25.0],
        initial_humidity=[50.0, 50.0, 50.0, 50.0]
    )
    
    # 스텝 응답 시퀀스
    print("\n🔸 스텝 응답 시퀀스 (제어 없음 → 최대 냉각 → 제어 없음)")
    
    # 1단계: 제어 없음 (10스텝)
    no_control = tester.create_action()
    tester.run_constant_control(no_control, 10, verbose=False)
    
    # 2단계: 최대 냉각 (15스텝)
    max_cooling = tester.create_action(
        peltier=1.0,
        small_fans=[1.0, 1.0, 1.0, 1.0],
        large_fan=1.0
    )
    tester.run_constant_control(max_cooling, 15, verbose=False)
    
    # 3단계: 다시 제어 없음 (10스텝)
    tester.run_constant_control(no_control, 10, verbose=False)
    
    tester.plot_results()

if __name__ == "__main__":
    # 기본 테스트 실행
    example_basic_test()
    
    # 추가 테스트들 (필요시 주석 해제)
    # example_cooling_test()
    # example_step_response_test()