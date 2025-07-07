# fireball, 토양의 모든 입자 데이터를 관리하고 물리 법칙에 따라 위치를 업데이트하는 모듈


import numpy as np

class ParticleManager:
    """ 모든 입자의 상태를 관리하고 위치를 업데이트하는 클래스 """

    def __init__(self, p1_particles):
        """
        Phase 1에서 넘어온 초기 화구 입자들로 ParticleManager를 초기화.
        """
        self.positions = p1_particles.copy()
        
        # 입자 종류 구별 (0: 화구 입자, 1: 토양 입자)
        self.types = np.zeros(len(p1_particles), dtype=int)
        print(f"✅ ParticleManager 초기화 완료: {len(self.positions)}개 화구 입자")

    def add_soil_particles(self, num_new, radius):
        """ 지표면(z=0)의 원형 영역 내에 토양 입자를 생성하여 추가. """
        if num_new <= 0:
            return
            
        # 원형 영역 내에 무작위로 입자 생성
        r = radius * np.sqrt(np.random.rand(num_new))
        theta = 2 * np.pi * np.random.rand(num_new)
        
        new_x = r * np.cos(theta)
        new_y = r * np.sin(theta)
        new_z = np.zeros(num_new) # z=0에서 생성
        
        new_positions = np.vstack((new_x, new_y, new_z)).T
        new_types = np.ones(num_new, dtype=int) # type=1 로 설정
        
        # 기존 배열에 새로 생성된 입자들 추가
        self.positions = np.vstack((self.positions, new_positions))
        self.types = np.append(self.types, new_types)

    def update_positions(self, dt, vortex_ring, atmosphere):
        """ 모든 입자들의 위치를 업데이트합니다. """
        
        # 1. Vortex Ring이 만드는 회전 속도장을 모든 입자에 대해 계산
        flow_velocities = vortex_ring.get_velocity_field_at(self.positions)
        
        # 2. 고도에 따른 주변 바람 속도를 모든 입자에 더해줌
        # (단순화를 위해 모든 입자가 구름 중심 고도의 바람을 받는다고 가정)
        ambient_props = atmosphere.get_ambient_properties(vortex_ring.position[2])
        wind_velocity = ambient_props['wind_vector_ms']

        # 3. 토양 입자(type==1)이면서, 아직 구름 기둥(stem) 영역에 있는 경우
        is_soil = self.types == 1
        # 구름 기둥 영역 정의 (대략적으로 와류 중심보다 낮은 모든 영역)
        in_stem_zone = self.positions[:, 2] < vortex_ring.position[2]
        stem_particles_mask = is_soil & in_stem_zone
        
        # 3-1. 기둥 입자들은 강한 상승 속도를 추가로 받음
        stem_rise_velocity = np.array([0, 0, 50.0]) # 예시: 50 m/s 상승
        total_velocities = flow_velocities + wind_velocity
        total_velocities[stem_particles_mask] += stem_rise_velocity

        # 4. 최종 위치 업데이트
        self.positions += total_velocities * dt
        
        # 입자가 땅 밑으로 내려가지 않도록 보정
        self.positions[:, 2] = np.maximum(self.positions[:, 2], 0)