import numpy as np
import time
from typing import Tuple, Dict, List

from simulator.components import PeltierModel, FanModel, ServoModel
from simulator.sensors import SensorModel
from simulator.physics import PhysicsSimulator
from simulator.utils import ZoneComfortCalculator
from configs.hvac_config import target_conditions, safety_limits, CONTROL_TERM

class AdvancedSmartACSimulator:
    """
    고도화된 스마트 에어컨 시뮬레이터
    - 강화학습 학습 환경으로 사용됨 (reset / step 구조)
    - 센서 및 제어기 시뮬레이션 포함
    """
    # =====================
    # 1. 초기화
    # =====================
    def __init__(self, num_zones: int = 4):
        self.num_zones = num_zones
        self.dt = CONTROL_TERM  # 제어 주기 (초)

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
        # 상태: 4(온도) + 4(습도) + 4(CO2) + 4(미세먼지) + 4(쾌적도) + 8(서보각도) + 5(팬속도) + 2(외부조건) + 4(영역별 TSV) = 39차원
        self.state_dim = 39
        # 액션: 1(펠티어) + 4(내부슬롯) + 4(외부슬롯) + 4(소형팬) + 1(대형팬) = 14차원
        self.action_dim = 14
        
        # --- 에피소드 로깅용 버퍼 ---
        self.episode_data: Dict[str, list] = {
            'rewards': [],
            'comfort_scores': [],
            'power_consumption': [],
        }

        # --- 보상 계산 보조 변수 ---
        self.prev_temps = np.zeros(self.num_zones, dtype=float)
        self.prev_fan_pwm = np.zeros(self.num_zones, dtype=float)
        self.prev_action = np.zeros(self.action_dim, dtype=float)
        self.current_tsv = np.zeros(self.num_zones, dtype=float)

        self.reset()

    def update_tsv(self, tsv_list: List[float]):
        if len(tsv_list) != self.num_zones:
            raise ValueError("TSV length mismatch")
        self.current_tsv = np.clip(np.asarray(tsv_list, dtype=float), -3.0, 3.0)
    
    # =====================
    # 3. 환경 초기화
    # =====================
    def reset(self) -> np.ndarray:
        """환경 초기화: 물리·센서·보조 변수 리셋"""
        self.physics_sim.reset()

        # 센서·서보 초기값
        for i in range(self.num_zones):
            self.internal_servos[i].current_angle = 30.0
            self.external_servos[i].current_angle = 40.0
            self.sensors[i].reset(
                self.physics_sim.T[i],
                self.physics_sim.H[i],
                self.physics_sim.CO2[i],
                self.physics_sim.Dust[i]
            )

        # 보조 변수 리셋
        self.prev_temps = self.physics_sim.T.copy()
        self.prev_fan_pwm = np.zeros(self.num_zones, dtype=float)
        self.prev_action = np.zeros(self.action_dim, dtype=float)
        self.current_tsv = np.zeros(self.num_zones, dtype=float)

        self.time_step = 0
        self.episode_data['rewards'].clear()
        self.episode_data['comfort_scores'].clear()
        self.episode_data['power_consumption'].clear()
        self.episode_start_time = time.time()

        return self._get_state_vector()

    # =====================
    # 4. Helper: 수동 초기 조건 세팅
    # =====================
    def set_initial_state(self, temperatures: list[float], humidities: list[float]):
        if len(temperatures) != self.num_zones or len(humidities) != self.num_zones:
            raise ValueError(f"Input lists must have length {self.num_zones}")
        self.physics_sim.T = np.array(temperatures, dtype=float)
        self.physics_sim.H = np.array(humidities, dtype=float)
        for i in range(self.num_zones):
            self.sensors[i].reset(
                self.physics_sim.T[i], self.physics_sim.H[i],
                self.physics_sim.CO2[i], self.physics_sim.Dust[i]
            )
        # prev_temps도 동기화
        self.prev_temps = self.physics_sim.T.copy()
    
    # =====================
    # 5. 메인 step 루프
    # =====================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # --- 스무스 패널티용 이전 액션 백업 ---
        prev_action_raw = self.prev_action.copy()
        self.prev_action = action.copy()

        # 5.1 Action 파싱 & HW 업데이트
        action_dict = self._parse_action(action)
        hw_states = self._update_hardware(action_dict)

        # 5.2 Physics update & 센서 읽기
        physics_state = self.physics_sim.update_physics(
            action_dict, hw_states['peltier'], hw_states['fans']
        )
        sensor_readings = self._read_sensors(physics_state)

        # 5.3 Comfort + Reward 계산
        comfort_data = self._calculate_comfort(sensor_readings)
        reward, reward_breakdown = self._calculate_reward(
            sensor_readings, comfort_data, hw_states, action_dict,
            prev_action_raw
        )

        # 5.4 종료 조건
        done = self._check_done(sensor_readings)

        # 5.5 로깅
        self.episode_data['rewards'].append(reward)
        self.episode_data['comfort_scores'].append(comfort_data['average_comfort'])
        self.episode_data['power_consumption'].append(hw_states['total_power'])

        # 5.6 next‑state 준비 (prev_* 업데이트)
        self.prev_temps = np.array(sensor_readings['temperatures'])
        self.prev_fan_pwm = np.array(action_dict['small_fan_pwm'])
        self.time_step += 1

        info = {
            'sensor_readings': sensor_readings,
            'comfort_data': comfort_data,
            'reward_breakdown': reward_breakdown,
            'hardware_states': hw_states,
            'time_step': self.time_step,
        }
        return self._get_state_vector(), reward, done, info

    # ----------------------------------------------------------------------
    # 6. Action 파싱
    # ----------------------------------------------------------------------
    def _parse_action(self, action: np.ndarray) -> Dict:
        action = np.clip(action, -1, 1)
        def _scale(val, min_val, max_val):
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
        # 펠티어 상태 업데이트
        # avg_room_temp = float(self.physics_sim.T.mean())
        # print('업데이트 됨. Peltier Control: ', action_dict['peltier_control'])

        if self.physics_sim.jet.last_Q_fan is None:
            # 첫 스텝에서는 팬 유량이 계산되지 않았으므로, 전체 평균 온도를 사용합니다.
            intake_temp = float(self.physics_sim.T.mean())
        else:
            # 이전 스텝의 팬 유량을 기반으로 가중 평균된 흡기 온도를 계산합니다.
            intake_temp = float((self.physics_sim.jet.last_Q_fan.diagonal() @ self.physics_sim.T) / (self.physics_sim.jet.last_Q_fan.diagonal().sum() + 1e-9))
        
        peltier_state = self.peltier.update(
            action_dict['peltier_control'],  # control (-1~1)
            intake_temp,                   # chamber_temp ← 에어컨에 유입되는 온도
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
    
    # ----------------------------------------------------------------------
    # 8. Sensor / Comfort helpers (기존 유지)
    # ----------------------------------------------------------------------
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
        # TSV를 comfort 계산에 함께 전달
        scores = []
        for i in range(self.num_zones):
            res = self.comfort_calcs[i].calculate_comfort(
                temp=sensor_readings['temperatures'][i],
                rh=sensor_readings['humidities'][i],
                v=0.1,  # TODO: 팬 rpm → 풍속 변환
                tsv=self.current_tsv[i]
            )
            scores.append(res['comfort_score'])
        return {"comfort_scores": scores, "average_comfort": float(np.mean(scores))}

    # ----------------------------------------------------------------------
    # 9. Reward (임시: 기존 그대로, 추후 Direction 보상으로 교체)
    # ----------------------------------------------------------------------
    def _calculate_reward(self, sensor_readings: Dict, comfort_data: Dict,
                          hw_states: Dict, action_dict: Dict, prev_action_raw: np.ndarray):
        
        temps = np.asarray(sensor_readings["temperatures"], dtype=float)  # numpy 변환

        # --- 1) TSV‑direction 보상 ---
        delta_T = np.clip((sensor_readings["temperatures"] - self.prev_temps) / 1.0, -1, 1)       # 1 °C scaling
        delta_v = np.clip((action_dict["small_fan_pwm"] - self.prev_fan_pwm) / 90.0,  -1, 1)      # 90 PWM scaling
        sign    = -np.sign(self.current_tsv)                                                      # +1 want ↑ temp if cold
        kappa   = 0.3
        R_dir   = float(np.mean(sign * delta_T + kappa * sign * delta_v))                         # [-1,1]

        # --- 2) Comfort 레벨 --- 80점 기준으로 0
        R_c = float(np.mean([(c/100.0) - 0.8 for c in comfort_data["comfort_scores"]]))           # approx [-0.8,0.2]

        # --- 3) Energy penalty (선형) ---
        P_tot   = hw_states.get("total_power", 0.0)
        P_MAX   = 1000.0                                                                          # [W] 가정, 필요시 config
        R_e     = - P_tot / P_MAX

        # --- 4) Smooth penalty ---
        R_s = -0.05 * float(np.abs(self.prev_action - prev_action_raw).sum())

        # --- 5) Safety penalty ---
        temp_max = safety_limits["temperature"]["max"]
        temp_min = safety_limits["temperature"]["min"]
        safety_violation = bool(np.any(temps > temp_max) or np.any(temps < temp_min))
        R_safe = -5.0 if safety_violation else 0.0

        # 가중합
        w_dir, w_c, w_e, w_s = 1.0, 0.6, 0.2, 0.1
        reward = w_dir*R_dir + w_c*R_c + w_e*R_e + w_s*R_s + R_safe

        breakdown = {
            "R_dir":       R_dir,
            "R_comfort":  R_c,
            "R_energy":   R_e,
            "R_smooth":   R_s,
            "R_safety":   R_safe,
            "reward":     reward,
        }
        return reward, breakdown
    
    # ----------------------------------------------------------------------
    # 10. Done 체크 (기존 유지)
    # ----------------------------------------------------------------------
    def _check_done(self, sensor_readings: Dict) -> bool:
        """
        에피소드 종료 조건 판단
        - 최대 타임스텝 도달 or 안전 범위 초과
        """
        if self.time_step >= 720:
            return True

        for temp in sensor_readings['temperatures']:
            if temp < safety_limits['temperature']["min"] - 2 or temp > safety_limits['temperature']["max"] + 5:
                return True

        for hum in sensor_readings['humidities']:
            if hum < safety_limits['humidity']["min"] - 10 or hum > safety_limits['humidity']["max"] + 30:
                return True

        return False

    # ----------------------------------------------------------------------
    # 11. 상태 벡터 생성 (TSV 포함)
    # ----------------------------------------------------------------------
    def _get_state_vector(self) -> np.ndarray:
        physics_state = self.physics_sim.get_current_state()
        sensor_readings = self._read_sensors(physics_state)
        comfort_data = self._calculate_comfort(sensor_readings)

        state = []
        # --- 센서값 정규화 (기존)
        state += [(x - 15) / 20 for x in sensor_readings['temperatures']]
        state += [x / 100 for x in sensor_readings['humidities']]
        state += [(x - 350) / 4650 for x in sensor_readings['co2_levels']]
        state += [x / 100 for x in sensor_readings['dust_levels']]
        state += [x / 100 for x in comfort_data['comfort_scores']]
        # --- NEW: TSV (-3~+3 → -1~1)
        state += [x / 3 for x in self.current_tsv]
        # --- 하드웨어 상태
        state += [servo.current_angle / 60 for servo in self.internal_servos]
        state += [servo.current_angle / 80 for servo in self.external_servos]
        state += [fan.current_rpm / 7000 for fan in self.small_fans]
        state.append(self.large_fan.current_rpm / 3300)
        state.append((self.physics_sim.ambient_temp - 15) / 20)
        state.append(self.physics_sim.ambient_hum / 100)

        return np.array(state, dtype=np.float32)
    
    # ----------------------------------------------------------------------
    # 12. 디버깅용 현재 상태 dict (기존 + TSV)
    # ----------------------------------------------------------------------
    def _get_current_state(self) -> Dict:
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
            "tsv": self.current_tsv.tolist(),
        }
