# 초기 : 상승과정만 모사, 원기둥 형태
# 회전력을 키워서 주변 속도장 만드는 로직 추가

import numpy as np
from scipy.special import ellipk, ellipe
from . import constants as const

class VortexRing:
    """ 버섯구름 '갓'의 물리적 상태와 움직임을 관리하는 클래스 """

    def __init__(self, p1_particles, p1_velocities, p1_summary):
        # --- Phase 1의 집계된 결과로 초기화 ---
        self.position = np.mean(p1_particles, axis=0)
        self.velocity = np.mean(p1_velocities, axis=0)
        self.radius = p1_summary['radius_m']
        self.temperature = p1_summary['temperature_k']

        # 물리 상수 및 모델 파라미터를 constants 모듈에서 가져옴
        self.R_GAS = const.R_DRY_AIR
        self.G = const.G
        self.DRAG_COEFF = const.DRAG_COEFF
        self.ENTRAINMENT_COEFF = const.ENTRAINMENT_COEFF
        self.CP_AIR = const.CP_DRY_AIR
        self.CIRCULATION_GENERATION_COEFF = const.CIRCULATION_GENERATION_COEFF
        
        # 초기 부피, 밀도, 질량 계산
        self.volume = (4/3) * np.pi * self.radius**3
        # 초기 압력은 주변 대기압과 같다고 가정
        # 실제로는 복잡하지만, 상승 초기에는 부력이 지배적이므로 단순화
        initial_pressure = 101325.0
        self.density = initial_pressure / (self.R_GAS * self.temperature)
        self.mass = self.density * self.volume
        
        # 와류의 회전 세기
        self.circulation = 0.0

    def update_state(self, dt, atmosphere):
        """ 매 시간 스텝마다 와류 고리의 상태를 업데이트합니다. """

        # 1. 현재 고도의 주변 대기 상태 가져오기
        ambient_props = atmosphere.get_ambient_properties(self.position[2])
        rho_air = ambient_props['density_kg_m3']
        
        # 2. 힘 계산 (모든 힘을 3차원 벡터로 계산)
        # 2-1. 부력 및 중력 (z축 방향으로만 작용)
        buoyancy_force_z = (rho_air - self.density) * self.volume * self.G
        gravity_force_z = -self.mass * self.G
        
        # 2-2. 항력 (속도의 반대 방향으로 작용하는 3차원 벡터)
        speed = np.linalg.norm(self.velocity)
        drag_force = -0.5 * self.DRAG_COEFF * rho_air * (np.pi * self.radius**2) * speed * self.velocity
        
        # 2-3. 모든 힘을 합쳐 최종 3차원 힘 벡터 생성
        total_force = np.array([
            drag_force[0], 
            drag_force[1], 
            buoyancy_force_z + gravity_force_z + drag_force[2]
        ])

         # --- [추가] 와류의 자체 유도 상승 속도 계산 ---
        # 실제 램-오신(Lamb-Oseen) 와류 모델에서는 로그 항이 더 붙지만, 핵심적인 관계를 단순화하여 적용
        # 회전력이 강할수록, 반경이 작을수록 자체 상승력이 커짐
        v_self_induced = self.circulation / (2 * np.pi * self.radius + 1e-9)

        # --- [수정] 총 힘 계산 시 자체 상승력을 속도에 직접 반영 ---
        # 가속도를 계산하기 전에, 현재 속도에 자체 유도 속도를 더해줌
        self.velocity[2] += v_self_induced * dt # dt 동안 자체 상승력으로 인한 속도 증가
        
        # 3. 상태 업데이트 (가속도, 속도, 위치를 3차원 벡터로 한번에 업데이트)
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # 4. 공기 유입(Entrainment) 모델
        surface_area = 4 * np.pi * self.radius**2
        entrained_mass = self.ENTRAINMENT_COEFF * surface_area * speed * rho_air * dt
        
        # 엔탈피 보존을 통해 새로운 온도 계산
        new_total_mass = self.mass + entrained_mass
        if new_total_mass > 0:
             self.temperature = (self.mass * self.temperature + entrained_mass * ambient_props['temperature_k']) / new_total_mass
        
        self.mass = new_total_mass
        # 밀도와 부피, 반경 재계산
        self.density = ambient_props['pressure_pa'] / (self.R_GAS * self.temperature)
        if self.density > 0:
            self.volume = self.mass / self.density
        self.radius = ((3 * self.volume) / (4 * np.pi))**(1/3)

        # [수정] 순환(Circulation) 업데이트: 물리 관계식 기반으로 변경
        # 바로클리닉 토크(baroclinic torque)를 부력으로 근사하여 회전력 생성률을 계산
        # dΓ/dt ≈ (g/ρ_air) * (Δρ) * R
        buoyancy_per_unit_volume = (rho_air - self.density) * self.G
        circulation_generation_rate = (buoyancy_per_unit_volume / rho_air) * self.radius
        
        self.circulation += circulation_generation_rate * dt

    def get_velocity_field_at(self, points):
        """ 
        Biot-Savart 법칙 기반으로, 특정 지점(points)에서 와류가 만드는 속도장을 계산.
        - P.G. Saffman, "Vortex Dynamics"의 공식을 참조하여 구현
        """
        if self.circulation == 0:
            return np.zeros_like(points)
            
        relative_pos = points - self.position
        r_cyl = np.linalg.norm(relative_pos[:, :2], axis=1)
        z_cyl = relative_pos[:, 2]

        # 계산 안정성을 위한 작은 값
        epsilon = 1e-9
        r_cyl += epsilon
        
        R = self.radius
        m = (4 * R * r_cyl) / ((R + r_cyl)**2 + z_cyl**2)
        
        K_m = ellipk(m)
        E_m = ellipe(m)
        
        # 속도 성분 계산
        common_factor = self.circulation / (2 * np.pi * np.sqrt((R + r_cyl)**2 + z_cyl**2))
        
        vr = common_factor * (z_cyl / r_cyl) * (((R**2 + r_cyl**2 + z_cyl**2) / ((R - r_cyl)**2 + z_cyl**2)) * E_m - K_m)
        vz = common_factor * (K_m - ((R**2 - r_cyl**2 + z_cyl**2) / ((R - r_cyl)**2 + z_cyl**2)) * E_m)
        
        # 다시 직교 좌표계 속도로 변환
        velocities = np.zeros_like(points)
        dir_xy = relative_pos[:, :2] / r_cyl[:, np.newaxis]
        velocities[:, :2] = dir_xy * vr[:, np.newaxis]
        velocities[:, 2] = vz
        
        return np.nan_to_num(velocities)