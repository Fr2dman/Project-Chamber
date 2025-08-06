import numpy as np
import time
from typing import Tuple, Dict

from simulator.components import PeltierModel, FanModel, ServoModel
from simulator.sensors import SensorModel
from simulator.physics import PhysicsSimulator
from simulator.utils import ZoneComfortCalculator
from configs.hvac_config import target_conditions, safety_limits

class AdvancedSmartACSimulator:
    """
    고도화된 스마트 에어컨 시뮬레이터
    - 강화학습 학습 환경으로 사용됨 (reset / step 구조)
    - 센서 및 제어기 시뮬레이션 포함
    """

    def __init__(self, num_zones: int = 4):
        self.num_zones = num_zones
        self.dt = 10.0  # 제어 주기 (초)

        # 하위 시뮬레이터 구성
        self.physics_sim = PhysicsSimulator(num_zones)
        self.comfort_calcs = [ZoneComfortCalculator(f"ZONE_{i}") for i in range(self.num_zones)]

        # 하드웨어 모델 (팬, 서보, 펠티어)
        self.peltier = PeltierModel()
        self.sensors = [SensorModel() for _ in range(num_zones)]
        self.internal_servos = [ServoModel(0, 45) for _ in range(num_zones)]
        self.external_servos = [ServoModel(0, 80) for _ in range(num_zones)]
        self.small_fans = [FanModel(7000, "small") for _ in range(num_zones)]
        self.large_fan = FanModel(3300, "large")

        # 액션 스케일링을 위한 범위 정의
        self.action_ranges = {
            'internal_servo': (0, 45),
            'external_servo': (0, 80),
            'fan_pwm': (0, 90) # 예시: (action+1)*45 -> 0~90
        }

        # 상태 공간 및 액션 공간
        # 상태: 4(온도) + 4(습도) + 4(CO2) + 4(미세먼지) + 4(쾌적도) + 8(서보각도) + 5(팬속도) + 2(외부조건) = 35차원
        self.state_dim = 35
        # 액션: 1(펠티어) + 4(내부슬롯) + 4(외부슬롯) + 4(소형팬) + 1(대형팬) = 14차원
        self.action_dim = 14
        
        # 성능 추적용
        self.episode_data = {
            'rewards': [],
            'comfort_scores': [],
            'power_consumption': [],
        }

        self.reset()

    def reset(self) -> np.ndarray:
        """
        환경 초기화
        - 랜덤 초기 상태
        - 센서 필터 상태 초기화
        """
        self.physics_sim.reset()

        for i in range(self.num_zones):
            self.internal_servos[i].current_angle = 30.0
            self.external_servos[i].current_angle = 40.0
            self.sensors[i].reset(
                self.physics_sim.T[i],
                self.physics_sim.H[i],
                self.physics_sim.CO2[i],
                self.physics_sim.Dust[i]
            )

        self.time_step = 0
        self.episode_start_time = time.time()
        return self._get_state_vector()

    def set_initial_state(self, temperatures: list[float], humidities: list[float]):
        """
        수동으로 초기 온/습도 상태를 설정하고 센서 상태를 동기화합니다.
        """
        if len(temperatures) != self.num_zones or len(humidities) != self.num_zones:
            raise ValueError(f"Input lists must have length {self.num_zones}")
        
        self.physics_sim.T = np.array(temperatures, dtype=float)
        self.physics_sim.H = np.array(humidities, dtype=float)

        # 센서 필터 상태도 물리 값에 맞춰 동기화
        for i in range(self.num_zones):
            self.sensors[i].reset(
                self.physics_sim.T[i], self.physics_sim.H[i],
                self.physics_sim.CO2[i], self.physics_sim.Dust[i]
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        시뮬레이션 단일 step 실행
        - 입력: SAC agent의 action 벡터
        - 출력: (다음 상태, 보상, 종료 여부, 부가정보)
        """
        action_dict = self._parse_action(action)
        hw_states = self._update_hardware(action_dict)
        # print(hw_states)
        physics_state = self.physics_sim.update_physics(
            action_dict, hw_states['peltier'], hw_states['fans']
        )
        sensor_readings = self._read_sensors(physics_state)
        comfort_data = self._calculate_comfort(sensor_readings)
        reward, reward_breakdown = self._calculate_reward(
            sensor_readings, comfort_data, hw_states, action_dict
        )
        done = self._check_done(sensor_readings)

        # 정보 저장
        self.episode_data['rewards'].append(reward)
        self.episode_data['comfort_scores'].append(comfort_data['average_comfort'])
        self.episode_data['power_consumption'].append(hw_states['total_power'])

        self.time_step += 1
        info = {
            'sensor_readings': sensor_readings,
            'comfort_data': comfort_data,
            'reward_breakdown': reward_breakdown,
            'hardware_states': hw_states,
            'time_step': self.time_step,
        }

        return self._get_state_vector(), reward, done, info

    def _parse_action(self, action: np.ndarray) -> Dict:
        """
        SAC 에이전트로부터 받은 [-1,1] 범위의 action 벡터를 하드웨어 제어 명령으로 변환
        - 추후 실물 제어와 연동 시 이 파트를 MQTT 제어 명령으로 교체 가능
        """
        action = np.clip(action, -1, 1)

        def _scale(val, min_val, max_val):
            """[-1, 1] 범위를 [min_val, max_val] 범위로 스케일링"""
            return (val + 1) / 2 * (max_val - min_val) + min_val

        return {
            'peltier_control': action[0],
            'internal_servo_angles': _scale(action[1:5], *self.action_ranges['internal_servo']),
            'external_servo_angles': _scale(action[5:9], *self.action_ranges['external_servo']),
            'small_fan_pwm': _scale(action[9:13], *self.action_ranges['fan_pwm']),
            'large_fan_pwm': _scale(action[13], *self.action_ranges['fan_pwm'])
        }

    def _update_hardware(self, action_dict: Dict) -> Dict:
        """
        각 제어 명령을 하드웨어 모델에 적용하여 상태 업데이트
        - 이후 실제 제어 시스템 연동 시 해당 구간을 MQTT 송신 구간으로 치환 가능
        """
        avg_room_temp = float(self.physics_sim.T.mean())
        peltier_state = self.peltier.update(
            action_dict['peltier_control'],  # control (-1~1)
            avg_room_temp,                   # chamber_temp ← **수정**
            self.physics_sim.ambient_temp,   # ambient_temp ← **수정**
            self.dt                          # dt
            )       
        total_power = peltier_state['power_consumption']

        for i in range(self.num_zones):
            self.internal_servos[i].set_angle(action_dict['internal_servo_angles'][i])
            self.external_servos[i].set_angle(action_dict['external_servo_angles'][i])
            self.internal_servos[i].update(self.dt)
            self.external_servos[i].update(self.dt)

        small_fan_states = []
        for i in range(self.num_zones):
            rpm = self.small_fans[i].set_pwm(action_dict['small_fan_pwm'][i])
            fan_state = self.small_fans[i].update(rpm, self.dt)
            small_fan_states.append(fan_state)
            total_power += fan_state['power']

        rpm_large = self.large_fan.set_pwm(action_dict['large_fan_pwm'])
        large_fan_state = self.large_fan.update(rpm_large, self.dt)
        total_power += large_fan_state['power']

        return {
            'peltier': {0: peltier_state},
            'servos': {
                'internal': [servo.current_angle for servo in self.internal_servos],
                'external': [servo.current_angle for servo in self.external_servos],
            },
            'fans': {
                'small_fans': small_fan_states,
                'large_fan': large_fan_state,
            },
            'total_power': total_power
        }

    def _read_sensors(self, physics_state: Dict) -> Dict:
        """센서 값을 읽고 필터를 적용하여 반환"""
        return {
            'temperatures': [self.sensors[i].read_temperature(physics_state['temperatures'][i], self.dt) for i in range(self.num_zones)],
            'humidities': [self.sensors[i].read_humidity(physics_state['humidities'][i], self.dt) for i in range(self.num_zones)],
            'co2_levels': [self.sensors[i].read_co2(physics_state['co2_levels'][i], self.dt) for i in range(self.num_zones)],
            'dust_levels': [self.sensors[i].read_dust(physics_state['dust_levels'][i], self.dt) for i in range(self.num_zones)],
        }

    def _calculate_comfort(self, sensor_readings: Dict) -> Dict:
        """
        쾌적도 계산 모듈 호출 (PMV/PPD 기반)
        - 향후 wearable 또는 카메라 기반 TSV 보정이 가능하도록 설계 고려
        """
        scores = []
        for i in range(self.num_zones):
            res = self.comfort_calcs[i].calculate_comfort(
                temp=sensor_readings['temperatures'][i],
                rh=sensor_readings['humidities'][i],
                v=0.1                   # TODO: fan rpm 기반 풍속
            )
            scores.append(res['comfort_score'])
        return {"comfort_scores": scores, "average_comfort": np.mean(scores)}

    def _calculate_reward(self, sensor_readings: Dict, comfort_data: Dict, hardware_states: Dict, action_dict: Dict) -> Tuple[float, Dict]:
        """
        보상 함수
        - 다중 목적: 쾌적도 향상 + 에너지 절감 + 안전 유지 + 제어 부드러움
        """
        comfort_reward = np.mean([score / 100.0 for score in comfort_data['comfort_scores']])
        temp_penalty = -np.mean([(t - target_conditions['temperature'][i]) ** 2 for i, t in enumerate(sensor_readings['temperatures'])]) * 0.1
        humidity_penalty = -np.mean([(h - target_conditions['humidity'][i]) ** 2 for i, h in enumerate(sensor_readings['humidities'])]) * 0.01
        power_penalty = -hardware_states['total_power'] * 0.001
        reward = comfort_reward + temp_penalty + humidity_penalty + power_penalty

        return reward, {
            'comfort': comfort_reward,
            'temp_penalty': temp_penalty,
            'humidity_penalty': humidity_penalty,
            'power_penalty': power_penalty
        }

    def _check_done(self, sensor_readings: Dict) -> bool:
        """
        에피소드 종료 조건 판단
        - 최대 타임스텝 도달 or 안전 범위 초과
        """
        if self.time_step >= 720:
            return True

        for temp in sensor_readings['temperatures']:
            if temp < safety_limits['temperature'][0] - 2 or temp > safety_limits['temperature'][1] + 2:
                return True

        for hum in sensor_readings['humidities']:
            if hum < safety_limits['humidity'][0] - 10 or hum > safety_limits['humidity'][1] + 10:
                return True

        return False

    def _get_state_vector(self) -> np.ndarray:
        """
        환경의 현재 상태를 벡터로 반환
        - 강화학습 에이전트 입력으로 사용
        - 추후 센서 통신 연동 시 이 부분을 MQTT 수신 데이터로 교체 가능
        """
        physics_state = self.physics_sim.get_current_state()
        sensor_readings = self._read_sensors(physics_state)
        comfort_data = self._calculate_comfort(sensor_readings)

        state = []
        state += [(x - 15) / 20 for x in sensor_readings['temperatures']]
        state += [x / 100 for x in sensor_readings['humidities']]
        state += [(x - 350) / 4650 for x in sensor_readings['co2_levels']]
        state += [x / 100 for x in sensor_readings['dust_levels']]
        state += [x / 100 for x in comfort_data['comfort_scores']]
        state += [servo.current_angle / 60 for servo in self.internal_servos]
        state += [servo.current_angle / 80 for servo in self.external_servos]
        state += [fan.current_rpm / 7000 for fan in self.small_fans]
        state.append(self.large_fan.current_rpm / 3300)
        state.append((self.physics_sim.ambient_temp - 15) / 20)
        state.append(self.physics_sim.ambient_hum / 100)

        return np.array(state, dtype=np.float32)
    
    def _get_current_state(self) -> Dict:
        """
        현재 시뮬레이터 상태를 반환
        - 온도, 습도, CO2, 미세먼지, 쾌적도 등
        """
        physics_state = self.physics_sim.get_current_state()
        sensor_readings = self._read_sensors(physics_state)
        return {
            "temperatures": self.physics_sim.T.tolist(),
            "humidities": self.physics_sim.H.tolist(),  
            "co2_levels": self.physics_sim.CO2.tolist(),
            "dust_levels": self.physics_sim.Dust.tolist(),
            "comfort_scores": self._calculate_comfort(sensor_readings),
            "ambient_temp": self.physics_sim.ambient_temp,
            "ambient_hum": self.physics_sim.ambient_hum,
        }   
