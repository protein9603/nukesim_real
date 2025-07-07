# main 실행 script
# phase1_fireball_expansion.py 모듈 적용 개별 입자 거동 적용.

import os
import numpy as np
import time as pytime
import pandas as pd

import config as cfg
from main_pkg import phase1_fireball, atmosphere, phase1_fireball_expansion
from main_pkg.phase2_vortex import run_phase2_simulation
from main_pkg.utils import ldm_writer, visualizer
from main_pkg.atmosphere import StandardAtmosphere
from main_pkg import atmosphere

def run_phase1_simulation():
    """
    Phase 1 (초기 화구 팽창) 시뮬레이션을 실행하고 결과물을 생성.
    """
    start_time = pytime.time()
    print("="*50)
    print("Phase 1: 초기 화구 팽창 시뮬레이션을 시작합니다.")
    print(f"시나리오: {cfg.SCENARIO_NAME}")
    print("="*50)

    # 1. 시뮬레이션 환경 설정
    print("\n[1] 시뮬레이션 환경을 설정합니다...")
    breakaway_state = phase1_fireball.get_breakaway_state(cfg.YIELD_KT, cfg.HOB_M)
    breakaway_time = breakaway_state['time_s']
    _, _, rho_air = atmosphere.get_properties(cfg.HOB_M)
    
    print(f"  - Breakaway 시간: {breakaway_time:.4f} 초")
    print(f"  - Breakaway 반경: {breakaway_state['radius_m']:.2f} 미터")
    print(f"  - Breakaway 온도: {breakaway_state['temperature_k']:.2f} K")

    # 2. 입자 초기화
    print(f"\n[2] {cfg.PARTICLE_COUNT}개의 입자를 생성합니다...")
    initial_particles = phase1_fireball_expansion.initialize_particles(cfg.PARTICLE_COUNT, cfg.HOB_M)
    
    velocities_physical = np.zeros_like(initial_particles)
    particles_physical = initial_particles.copy()

    visual_history = []
    simulation_times = np.linspace(0, breakaway_time, 100) # 100프레임으로 생성
    dt = simulation_times[1] - simulation_times[0]
    
    # 3. 메인 시뮬레이션 루프
    print("\n[3] 유체역학적 팽창을 시뮬레이션합니다...")
    num_frames = 100 
    simulation_times = np.linspace(0, breakaway_time, num_frames)
    dt = simulation_times[1] - simulation_times[0]
    temp_particles = initial_particles.copy()
    
    for t in simulation_times:
        particles_physical, velocities_physical = phase1_fireball_expansion.update_particle_positions(
            particles_physical, t, dt, cfg.YIELD_KT, rho_air, cfg.HOB_M, breakaway_state, enable_visual_effects=False
        )
        temp_particles, _ = phase1_fireball_expansion.update_particle_positions(
            temp_particles, t, dt, cfg.YIELD_KT, rho_air, cfg.HOB_M, breakaway_state, enable_visual_effects=True
        )
        visual_history.append(temp_particles.copy())

    print(f"  - {num_frames} 프레임 시뮬레이션 완료.")

    # 4. 결과물 저장
    print("\n[4] 결과물을 파일로 저장합니다...")
    output_folder = f"results/phase1_expansion_test/{cfg.SCENARIO_NAME}"
    os.makedirs(output_folder, exist_ok=True)

    # 요약 결과와 입자 데이터를 각각 저장
    summary_path = os.path.join(output_folder, "final_state_summary.csv")
    ldm_writer.save_phase1_results_to_csv(breakaway_state, summary_path) # 요약 저장용 함수 호출

    particles_path = os.path.join(output_folder, "phase1_final_state_particles.csv")
    ldm_writer.save_particle_data_to_csv(particles_physical, velocities_physical, particles_path) # 입자 저장용 함수 호출

    if cfg.SAVE_ANIMATION:
        anim_path = os.path.join(output_folder, "phase1_expansion_animation.mp4")
        visualizer.create_expansion_animation(visual_history, simulation_times, cfg.HOB_M, anim_path)
    
    end_time = pytime.time()
    print("\n" + "="*50)
    print(f"Phase 1 시뮬레이션이 성공적으로 완료되었습니다. (소요시간: {end_time - start_time:.2f}초)")
    print("="*50)
    
    # Phase 2에 필요한 모든 데이터를 반환
    return particles_physical, velocities_physical, breakaway_state

def main():
    """
    전체 핵폭발 시뮬레이션을 순차적으로 실행하는 메인 함수
    """
    # === Phase 1: 초기 화구 팽창 ===
    # 변경된 반환값에 맞춰 변수 할당
    p1_final_particles, p1_final_velocities, p1_summary_state = run_phase1_simulation()

    # === Phase 2: 버섯 구름 형성 (설정에 따라 실행) ===
    if cfg.RUN_PHASE_2:
        # 표준대기모델로 단순화된 주변환경 구축
        atmosphere_model = StandardAtmosphere()
        
        # Phase 2 결과 저장 폴더 설정
        output_folder_p2 = f"results/phase2_vortex_test/{cfg.SCENARIO_NAME}"
        
        # Phase 2 시뮬레이션 함수에 속도(velocities) 인자 추가 전달
        run_phase2_simulation(
            p1_particles=p1_final_particles,
            p1_velocities=p1_final_velocities,
            p1_summary=p1_summary_state,
            atmosphere_model=atmosphere_model,
            output_folder=output_folder_p2
        )

if __name__ == "__main__":
    main()