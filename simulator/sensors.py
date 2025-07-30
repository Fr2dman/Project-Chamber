import numpy as np

class BaseSensorModel:
    """센서 인터페이스 모델 - 실제 시뮬레이터용 또는 실측 대체 가능"""
    def reset(self, actual_temp: float, actual_humidity: float, actual_co2: float, actual_dust: float):
        raise NotImplementedError

    def read_temperature(self, actual_temp: float, dt: float) -> float:
        raise NotImplementedError

    def read_humidity(self, actual_humidity: float, dt: float) -> float:
        raise NotImplementedError

    def read_co2(self, actual_co2: float, dt: float) -> float:
        raise NotImplementedError

    def read_dust(self, actual_dust: float, dt: float) -> float:
        raise NotImplementedError


class SimpleSensorModel(BaseSensorModel):
    """간단한 센서 모델 (빠른 학습용)"""
    def __init__(self, noise_std=0.2):
        self.temp_filter_state = 25.0
        self.humidity_filter_state = 50.0
        self.co2_filter_state = 400.0
        self.dust_filter_state = 0.0
        self.noise_std = noise_std

    def reset(self, actual_temp: float, actual_humidity: float, actual_co2: float, actual_dust: float):
        """필터 상태를 실제 값으로 초기화합니다."""
        self.temp_filter_state = actual_temp
        self.humidity_filter_state = actual_humidity
        self.co2_filter_state = actual_co2
        self.dust_filter_state = actual_dust

    def read_temperature(self, actual_temp: float, dt: float) -> float:
        alpha = 0.1
        self.temp_filter_state += alpha * (actual_temp - self.temp_filter_state)
        return self.temp_filter_state + np.random.normal(0, self.noise_std)

    def read_humidity(self, actual_humidity: float, dt: float) -> float:
        alpha = 0.1
        self.humidity_filter_state += alpha * (actual_humidity - self.humidity_filter_state)
        return np.clip(self.humidity_filter_state + np.random.normal(0, self.noise_std), 0, 100)

    def read_co2(self, actual_co2: float, dt: float) -> float:
        alpha = 0.05
        self.co2_filter_state += alpha * (actual_co2 - self.co2_filter_state)
        return np.clip(self.co2_filter_state + np.random.normal(0, 15), 350, 10000)

    def read_dust(self, actual_dust: float, dt: float) -> float:
        alpha = 0.2
        self.dust_filter_state += alpha * (actual_dust - self.dust_filter_state)
        return max(0, self.dust_filter_state + np.random.normal(0, 0.1))


class PreciseSensorModel(BaseSensorModel):
    """실제 센서 특성을 반영한 정밀 센서 모델"""
    def __init__(self):
        self.temp_accuracy = 0.2
        self.humidity_accuracy = 2.0
        self.co2_accuracy_ppm = 50
        self.co2_accuracy_percent = 0.05

        self.temp_response_time = 30.0
        self.humidity_response_time = 8.0
        self.co2_response_time = 60.0
        self.dust_response_time = 5.0

        self.temp_filter_state = 25.0
        self.humidity_filter_state = 50.0
        self.co2_filter_state = 400.0
        self.dust_filter_state = 0.0

    def reset(self, actual_temp: float, actual_humidity: float, actual_co2: float, actual_dust: float):
        """필터 상태를 실제 값으로 초기화합니다."""
        self.temp_filter_state = actual_temp
        self.humidity_filter_state = actual_humidity
        self.co2_filter_state = actual_co2
        self.dust_filter_state = actual_dust

    def read_temperature(self, actual_temp: float, dt: float) -> float:
        tau = self.temp_response_time
        alpha = dt / (tau + dt)
        self.temp_filter_state += alpha * (actual_temp - self.temp_filter_state)
        noise = np.random.normal(0, self.temp_accuracy / 3)
        return self.temp_filter_state + noise

    def read_humidity(self, actual_humidity: float, dt: float) -> float:
        tau = self.humidity_response_time
        alpha = dt / (tau + dt)
        self.humidity_filter_state += alpha * (actual_humidity - self.humidity_filter_state)
        noise = np.random.normal(0, self.humidity_accuracy / 3)
        return np.clip(self.humidity_filter_state + noise, 0, 100)

    def read_co2(self, actual_co2: float, dt: float) -> float:
        tau = self.co2_response_time
        alpha = dt / (tau + dt)
        self.co2_filter_state += alpha * (actual_co2 - self.co2_filter_state)
        accuracy_error = self.co2_accuracy_ppm + self.co2_filter_state * self.co2_accuracy_percent
        noise = np.random.normal(0, accuracy_error / 3)
        return np.clip(self.co2_filter_state + noise, 350, 10000)

    def read_dust(self, actual_dust: float, dt: float) -> float:
        tau = self.dust_response_time
        alpha = dt / (tau + dt)
        self.dust_filter_state += alpha * (actual_dust - self.dust_filter_state)
        noise = np.random.normal(0, 0.1)
        return max(0, self.dust_filter_state + noise)


class SensorModel(BaseSensorModel):
    """모드에 따라 정밀/간단 센서모델 전환 래퍼"""
    def __init__(self, mode="simple"):
        if mode == "precise":
            self.impl = PreciseSensorModel()
        else:
            self.impl = SimpleSensorModel()

    def reset(self, actual_temp: float, actual_humidity: float, actual_co2: float, actual_dust: float):
        return self.impl.reset(actual_temp, actual_humidity, actual_co2, actual_dust)

    def read_temperature(self, actual_temp: float, dt: float) -> float:
        return self.impl.read_temperature(actual_temp, dt)

    def read_humidity(self, actual_humidity: float, dt: float) -> float:
        return self.impl.read_humidity(actual_humidity, dt)

    def read_co2(self, actual_co2: float, dt: float) -> float:
        return self.impl.read_co2(actual_co2, dt)

    def read_dust(self, actual_dust: float, dt: float) -> float:
        return self.impl.read_dust(actual_dust, dt)

# TODO: 실제 시연 환경에서는 센서 데이터를 통신으로 수신하는 RealSensorModel 클래스 구현 필요
