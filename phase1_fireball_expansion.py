# NUKE_SIM/nukesim/phase1_expansion.py
# 입자들의 초기 위치 생성, 매시간 위치 업데이트하여 개별 거동 고려한 모델 적용
# 팽창속도 + 열운동속도 + 난류속도 적용

import numpy as np
import noise
from . import phase1_fireball

def initialize_particles(num_points, hob):
    """
    폭발 원점 주변의 '작은 초기 구' 내부에 입자들을 무작위로 생성.
    이렇게 하면 모든 입자가 중심으로부터 0이 아닌 초기 거리를 갖게 되어
    팽창 시뮬레이션이 올바르게 시작가능. 현실적으로도 최초 무기의 부피가 있음.

    Args:
        num_points (int): 생성할 입자 수
        hob (float): 폭발 고도

    Returns:
        np.array: (N, 3) 모양의 입자 위치 배열
    """
    # 1. '점 구름' 방식과 동일하게 단위 구(반지름 1) 내에 균일하게 점을 생성
    r = np.random.rand(num_points)**(1/3)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(2 * np.random.rand(num_points) - 1)

    x_norm = r * np.sin(phi) * np.cos(theta)
    y_norm = r * np.sin(phi) * np.sin(theta)
    z_norm = r * np.cos(phi)

    # 2. 이 점들을 매우 작은 초기 반경(예: 0.1미터)으로 축소
    initial_radius = 0.1  # meters
    
    x = x_norm * initial_radius
    y = y_norm * initial_radius
    z = z_norm * initial_radius + hob # 폭발 고도(HOB) 오프셋 적용

    # 3. 최종적인 (N, 3) 입자 위치 배열로 결합
    particles = np.vstack((x, y, z)).T
    return particles


def update_particle_positions(particles, t, dt, W_kt, rho_air, hob, breakaway_state, enable_visual_effects=False):
    """
    '선형 팽창 속도'에 '무작위 열운동'을 추가하여 입자들의 위치를 업데이트.
    """
    # 1. 기본 유체역학적 팽창 속도 계산
    r_current = phase1_fireball.get_radius_at_time(t, W_kt, rho_air)
    if r_current < 1e-6:
        return particles, np.zeros_like(particles)
    
    r_next = phase1_fireball.get_radius_at_time(t + dt, W_kt, rho_air)
    edge_velocity = (r_next - r_current) / dt
    
    center = np.array([0, 0, hob])
    relative_positions = particles - center
    distances = np.linalg.norm(relative_positions, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        hydro_velocity_mag = edge_velocity * (distances / r_current)
    hydro_velocity_mag = np.nan_to_num(hydro_velocity_mag)

    directions = relative_positions / (distances[:, np.newaxis] + 1e-9)
    final_velocities = directions * hydro_velocity_mag[:, np.newaxis]
    
    # 2. 시각 효과 플래그가 True일 때만 난류/열운동 추가
    if enable_visual_effects:
        turbulence_strength = 30.0
        noise_scale = 5.0 / (r_current + 1e-9)
        octaves = 2
        persistence = 0.5
        
        num_particles = len(particles)
        turbulent_velocities = np.zeros_like(particles)
        time_evolution_speed = 200.0
        time_offset = t * time_evolution_speed
        
        for i in range(num_particles):
            p = particles[i] * noise_scale
            turb_x = noise.pnoise3(p[0] + time_offset, p[1], p[2], octaves=octaves, persistence=persistence, base=0)
            turb_y = noise.pnoise3(p[0] + time_offset, p[1], p[2], octaves=octaves, persistence=persistence, base=1)
            turb_z = noise.pnoise3(p[0] + time_offset, p[1], p[2], octaves=octaves, persistence=persistence, base=2)
            turbulent_velocities[i] = [turb_x, turb_y, turb_z]

        turbulent_velocities *= turbulence_strength * edge_velocity
        final_velocities += turbulent_velocities
    
    # 3. 최종 위치 업데이트
    new_positions = particles + final_velocities * dt
    
    # 위치와 함께 속도도 반환
    return new_positions, final_velocities