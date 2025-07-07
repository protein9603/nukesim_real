# Phase2 : 버섯구름 상승
# Vortex Dynamics
# WRF 미적용(설치문제로 인해 차후 다시 시도 예정) --> 표준 대기모델 적용 : 고도에 따른 기온, 기압, 밀도를 수학공식으로 간단히 계산해주는 모델임.

import numpy as np
import os
from tqdm import tqdm

import config as cfg
from .vortex_model import VortexRing
from .particle_manager import ParticleManager
from main_pkg.utils import ldm_writer, visualizer

# --- Phase 1의 속도 데이터를 인자로 추가 ---
def run_phase2_simulation(
    p1_particles,
    p1_velocities,
    p1_summary,
    atmosphere_model,
    output_folder
    ):
    """
    Phase 2: Vortex Ring 동역학을 통한 버섯구름 형성 시뮬레이션을 수행합니다.
    """
    print("\n" + "="*50)
    print("Phase 2: Vortex Dynamics 시뮬레이션을 시작합니다.")
    print(f"시나리오: {cfg.SCENARIO_NAME}")
    print("="*50)

    # --- 1. 시뮬레이션 파라미터 및 객체 초기화 ---
    dt = cfg.TIME_STEP_DT
    num_steps = int(cfg.PHASE2_SIMULATION_TIME_S / dt)

    print("[1] 시뮬레이션 객체를 초기화합니다...")
    # VortexRing과 ParticleManager 객체 생성
    vortex_ring = VortexRing(p1_particles, p1_velocities, p1_summary)
    particle_manager = ParticleManager(p1_particles)
    
    # 애니메이션 및 데이터 저장을 위한 기록용 리스트
    particle_history = []
    simulation_times = []

    # --- 2. 메인 시뮬레이션 루프 ---
    print(f"[2] 시뮬레이션을 시작합니다 (총 {cfg.PHASE2_SIMULATION_TIME_S}초, {num_steps} 스텝)")
    for step in tqdm(range(num_steps), desc="Phase 2 진행"):
        current_time = step * dt

        # 2-1. 거시적 객체(VortexRing)의 상태 업데이트 (상승, 팽창, 냉각 등)
        vortex_ring.update_state(dt, atmosphere_model)

        # 2-2. 토양 입자 유입 및 기둥 형성
        # 현재는 구름의 상승 속도와 반경에 비례하여 유입량을 동적으로 계산
        # 실제로는 Afterwinds(후폭풍), 폭발위력, 폭발고도, 지표면특성이 유입량에 영향을 미침.

       # 상승 속도가 양수일 때만 유입 발생
        if vortex_ring.velocity[2] > 0:
        # 초당 생성될 입자 수 계산
            particles_per_second = cfg.SOIL_ENTRAINMENT_RATE * vortex_ring.radius * vortex_ring.velocity[2]
            num_new_soil = int(particles_per_second * dt)
            # 초기 극한 상황에 따른 오류를 방지하기 위해 계산된 입자 수가 최대치를 넘지 않도록 제한
            num_new_soil = min(num_new_soil, cfg.MAX_SOIL_PER_STEP)
        
        if num_new_soil > 0:
            suck_in_radius = vortex_ring.radius * 0.5
            particle_manager.add_soil_particles(num_new_soil, suck_in_radius)
        
        # 2-3. 모든 입자들의 위치를 업데이트
        particle_manager.update_positions(dt, vortex_ring, atmosphere_model)
        
        # 2-4. 결과 저장
        # 1초(1/dt 스텝)마다 한 번씩 현재 상태를 기록
        if step % int(1.0 / dt) == 0:
            particle_history.append({
                'positions': particle_manager.positions.copy(),
                'types': particle_manager.types.copy()
            })
            simulation_times.append(current_time)
            
        # 안정화 조건 확인 (상승 속도가 매우 느려지면 조기 종료)
        if step > 100 and np.abs(vortex_ring.velocity[2]) < 0.1:
            print(f"\n구름이 고도 {vortex_ring.position[2]:.2f}m 에서 안정화되어 시뮬레이션을 조기 종료합니다.")
            break

    # --- 3. 최종 결과물 저장 ---
    print("\n[3] 최종 결과물을 파일로 저장합니다...")
    os.makedirs(output_folder, exist_ok=True)
    
    # 3-1. 최종 입자 상태를 CSV로 저장
    final_particles_path = os.path.join(output_folder, "phase2_final_particles.csv")
    ldm_writer.save_particle_data_phase2(
        particle_manager.positions, 
        particle_manager.types, 
        final_particles_path
    )

    # 3-2. 시뮬레이션 과정 애니메이션으로 저장
    animation_path = os.path.join(output_folder, "phase2_cloud_rise_animation.mp4")
    visualizer.create_phase2_animation(
        particle_history, 
        simulation_times, 
        animation_path
    )
    
    print(f"\nPhase 2 시뮬레이션 결과가 '{output_folder}'에 저장되었습니다.")