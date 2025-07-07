# Phase1 : Fireball 형성 
# 경험식 기반
# 개별입자거동은 미고려한 상태로 개별입자 거동을 위한 값 게산 모듈임.

import numpy as np
from . import constants as const
from . import atmosphere

def get_radius_at_time(t, W_kt, rho_air_kg_m3):
    """시간(t)에 따른 화구의 반경(R)을 계산."""
    W_joules = W_kt * 4.184e12
    C = 1.2 
    radius_m = C * ((W_joules * t**2) / rho_air_kg_m3)**(1/5)
    return radius_m

def get_temperature_at_time(t, W_kt, breakaway_time):
    """
    시간(t)에 따른 화구 표면의 온도를 계산.
    계산된 breakaway 시간에 맞춰 온도가 30만K가 되도록 역산하여 일관성을 맞춤.
    """
    BREAKAWAY_TEMP_K = 300000.0
    # T = A * t^(-n) 에서, breakaway 조건을 만족하는 A를 찾음.
    # 300,000 = A * t_breakaway^(-2.5) -> A = 300,000 * t_breakaway^(2.5)
    decay_factor = 2.5 # 급격한 감소를 표현하기 위한 온도 감소게수
    A = BREAKAWAY_TEMP_K * (breakaway_time**decay_factor)
    
    temperature_k = A * (t + 1e-9)**(-decay_factor)
    return temperature_k

def get_breakaway_state(W_kt, HOB_m):
    """
    '스케일링 법칙(Scaling Law)'을 이용하여 Breakaway 시점의 상태를 계산.
    이 방식은 훨씬 더 안정적이고 물리적 현실에 가까움.
    """
    # --- 스케일링 법칙 적용 ---
    # 기준: 1kt 폭발 시 breakaway 시간은 약 0.007초
    # 시간은 W^0.4에 비례하여 스케일링됨 (출처: Glasstone & Dolan)
    T_REF_1KT = 0.007  # 1kt 기준 breakaway 시간 (초)
    SCALING_EXPONENT = 0.4
    
    breakaway_time = T_REF_1KT * (W_kt**SCALING_EXPONENT)
    

    final_temp_k = get_temperature_at_time(breakaway_time, W_kt, breakaway_time)
    
    _, _, rho_air = atmosphere.get_properties(HOB_m)
    
    breakaway_radius = get_radius_at_time(breakaway_time, W_kt, rho_air)
    breakaway_altitude = HOB_m

    
    breakaway_state = {
        'time_s': breakaway_time,
        'altitude_m': breakaway_altitude,
        'radius_m': breakaway_radius,
        'temperature_k': final_temp_k,
        'yield_kt': W_kt,
        'initial_hob_m': HOB_m
    }
    
    return breakaway_state