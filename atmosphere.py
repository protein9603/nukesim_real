# 표준 대기모델 적용
# 고도별 밀도, 압력, 온도 등

import numpy as np
from . import constants as const

# U.S. Standard Atmosphere 1976 모델 기반
# 각 대기층의 기준값(고도, 압력, 온도, 온도변화율)을 미리 정의하여
# 계산을 단순화하고 효율성을 높임.

# 고도(m), 기준 압력(Pa), 기준 온도(K), 온도변화율(K/m)
ATMOSPHERE_LAYERS = [
    # 0: 대류권 (Troposphere)
    {'base_h': 0,      'base_P': 101325.0, 'base_T': 288.15, 'lapse_rate': -0.0065},
    # 1: 성층권 (Tropopause/Stratosphere)
    {'base_h': 11000,  'base_P': 22632.1,  'base_T': 216.65, 'lapse_rate': 0.0},
    # 2: 성층권 상부 (Upper Stratosphere)
    {'base_h': 20000,  'base_P': 5474.89,  'base_T': 216.65, 'lapse_rate': 0.001},
    # 3: 중간권 (Mesosphere)
    {'base_h': 32000,  'base_P': 868.02,   'base_T': 228.65, 'lapse_rate': 0.0028},
    # 4: 중간권 상부 (Upper Mesosphere)
    {'base_h': 47000,  'base_P': 110.91,   'base_T': 270.65, 'lapse_rate': 0.0}
]

def get_properties(altitude_m):
    """
    주어진 고도(m)에서 표준 대기의 온도, 압력, 밀도를 계산.

    Args:
        altitude_m (float): 고도 (미터 단위)

    Returns:
        tuple: (온도(K), 압력(Pa), 밀도(kg/m^3))
    """
    # 입력 고도가 80km 이상이면 최상층 데이터 사용 (모델 한계)
    if altitude_m >= 80000:
        altitude_m = 79999

    # 입력 고도가 어느 층에 속하는지 찾기
    layer = None
    for l in reversed(ATMOSPHERE_LAYERS):
        if altitude_m >= l['base_h']:
            layer = l
            break

    base_h = layer['base_h']
    base_P = layer['base_P']
    base_T = layer['base_T']
    lapse_rate = layer['lapse_rate']

    # 1. 온도 계산
    temperature_k = base_T + lapse_rate * (altitude_m - base_h)

    # 2. 압력 계산
    if lapse_rate == 0:  # 등온층인 경우
        pressure_pa = base_P * np.exp(-const.G * (altitude_m - base_h) / (const.R_DRY_AIR * base_T))
    else:  # 온도 변화가 있는 층인 경우
        pressure_pa = base_P * (base_T / temperature_k) ** (const.G / (lapse_rate * const.R_DRY_AIR))

    # 3. 밀도 계산 (이상 기체 상태 방정식: P = ρRT)
    density_kg_m3 = pressure_pa / (const.R_DRY_AIR * temperature_k)

    return temperature_k, pressure_pa, density_kg_m3

# 표준대기모델 활용, 특정 고도입력 시 해당 고도의 기온, 기압, 밀도 반환.\
# 바람은 없다고 가정함.

class StandardAtmosphere:
    """
    고도(z)에 따라 기온, 기압, 밀도를 계산하는 국제 표준 대기(ISA) 모델.
    WRF 데이터 없이 Phase 2 시뮬레이션을 위한 간단한 대기 환경을 제공.
    """
    def __init__(self):
        # 해수면 기준 상수 정의
        self.T0 = 288.15      # 해수면 온도 (K)
        self.P0 = 101325.0    # 해수면 기압 (Pa)
        self.g = 9.80665      # 중력 가속도 (m/s^2)
        self.R = 287.058      # 특정 기체 상수 (J/kg·K)
        self.a = -0.0065      # 온도 감률 (K/m) - 대류권(~11km)

        # --- 단순화된 바람 프로파일 파라미터 ---
        self.WIND_SHEAR_COEFF = 0.002  # 바람 시어 계수 (1/s), 값이 클수록 고도별 바람 차이가 커짐
        self.WIND_DIRECTION_DEG = 270  # 바람 방향 (도, 270=서풍)으로 가정

        print("✅ 표준 대기(Standard Atmosphere) 모델을 초기화했습니다.")

    def get_ambient_properties(self, z):
        """
        특정 고도(z)의 대기 상태(온도, 압력, 밀도)를 계산하여 반환합니다.
        
        Args:
            z (float): 대기 정보를 알고 싶은 고도(m)

        Returns:
            dict: 해당 고도의 대기 정보 딕셔너리
        """
        # 1. 온도 계산
        # 대류권(고도 11,000m 이하) 모델을 단순 적용
        temperature = self.T0 + self.a * z
        temperature = max(temperature, 216.65) # 성층권 최저 온도 이하로 내려가지 않도록 보정

        # 2. 기압 계산
        if self.a != 0:
            pressure = self.P0 * (temperature / self.T0)**(-self.g / (self.a * self.R))
        else: # 등온층일 경우
            pressure = self.P0 * np.exp(-self.g * (z) / (self.R * self.T0))
        
        # 3. 밀도 계산 (이상기체 상태방정식: P = ρRT)
        density = pressure / (self.R * temperature)

        # 4. 고도 z에서의 바람 벡터 계산 ---
        wind_speed = z * self.WIND_SHEAR_COEFF  # 고도에 정비례하여 바람 속도 증가
        wind_rad = np.deg2rad(self.WIND_DIRECTION_DEG)
        u_wind = wind_speed * np.cos(wind_rad)
        v_wind = wind_speed * np.sin(wind_rad)

        properties = {
            'temperature_k': temperature,
            'pressure_pa': pressure,
            'density_kg_m3': density,
            'wind_vector_ms': np.array([u_wind, v_wind, 0.0])
        }
        return properties