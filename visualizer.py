import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm

def plot_radius_vs_time(times, radii, filepath):
    """시간에 따른 화구 반경 변화 그래프를 저장하는 함수"""
    # 이 함수는 변경 사항 없습니다.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, radii, lw=2, color='red')
    ax.set_title('Fireball Radius Growth over Time (Phase 1)', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Radius (meters)', fontsize=12)
    ax.grid(True)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"반경 성장 그래프가 '{filepath}'에 저장되었습니다.")


def create_expansion_animation(particle_history, simulation_times, hob, filepath):
    """
    핵폭발 화구 팽창 과정을 3D 애니메이션으로 생성하고 저장하는 함수

    [수정 사항]
    - main.py에서 데이터 길이 일치를 보장하므로, 불필요해진 IndexError 방어 코드를 제거했습니다.
    - 코드가 더 간결하고 명확해졌습니다.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # main.py에서 길이를 맞춰주므로, particle_history의 길이로 프레임 수를 결정합니다.
    num_frames = len(particle_history)
    
    # 애니메이션 프레임이 없는 경우 오류를 방지하고 함수를 종료합니다.
    if num_frames == 0:
        print("오류: 애니메이션을 생성할 입자 데이터가 없습니다.")
        return

    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # 최종 상태의 입자 정보를 기준으로 색상과 축 범위를 미리 계산하여 효율을 높입니다.
    final_particles = particle_history[-1]
    
    # 1. 색상 설정: 중심에서 멀어질수록 어두워지는 'hot' 컬러맵 적용
    center_point = np.array([0, 0, hob])
    distances = np.linalg.norm(final_particles - center_point, axis=1)
    max_dist = np.max(distances)
    if max_dist < 1e-6: max_dist = 1.0 # 0으로 나누기 방지
    
    intensity = 1 - (distances / max_dist)**2
    colors = cm.hot(intensity)

    # 2. 축 범위 설정: 최종 화구 크기 기준으로 여유 공간을 20% 주어 설정
    all_particles = np.concatenate(particle_history, axis=0)
    x_min, x_max = np.min(all_particles[:, 0]), np.max(all_particles[:, 0])
    y_min, y_max = np.min(all_particles[:, 1]), np.max(all_particles[:, 1])
    z_min, z_max = np.min(all_particles[:, 2]), np.max(all_particles[:, 2])

    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min
    max_range = np.max([range_x, range_y, range_z]) * 0.6 # 약간의 여유


    mid_x = (x_min + x_max) * 0.5
    mid_y = (y_min + y_max) * 0.5
    mid_z = (z_min + z_max) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect((1, 1, 1))

    # 애니메이션 성능을 위해 scatter 객체를 미리 생성
    pbar = tqdm(total=num_frames, desc="화구 팽창 애니메이션 렌더링", unit="frame")
    scatter_plot = ax.scatter([], [], [], s=5, c=[], cmap=cm.hot)

    def update(frame):
        particles = particle_history[-1] # 마지막 프레임의 색상 기준으로 고정
        center_point = np.array([0, 0, hob])
        distances = np.linalg.norm(particles - center_point, axis=1)
        max_dist = np.max(distances)
        if max_dist < 1e-6: max_dist = 1.0
        intensity = 1 - (distances / max_dist)**2
        colors = cm.hot(intensity)
        # frame에 해당하는 입자 데이터로 업데이트
        particles = particle_history[frame]
        scatter_plot._offsets3d = (particles[:, 0], particles[:, 1], particles[:, 2])
        # 색상은 최초 계산된 값을 계속 사용
        scatter_plot.set_color(colors) 

        # 축 라벨 및 타이틀 설정
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        ax.tick_params(axis='z', colors='white', labelsize=8)
        
        # main.py에서 Z축 보정을 했으므로, 실제 데이터 비율에 맞게 aspect 설정
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

        ax.set_title(f'Time: {simulation_times[frame]:.4f} s', color='white', y=0.95)
        pbar.update(1)
        return scatter_plot,

    # blit=False는 3D plot에서 안정성을 높여주므로 그대로 사용합니다.
    ani = FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=False)

    ani.save(filepath, writer='ffmpeg', fps=20, dpi=150, savefig_kwargs={'facecolor': 'black'})
    pbar.close()
    plt.close(fig)
    print(f"\n입자 거동 애니메이션이 '{filepath}'에 저장되었습니다.")

def create_phase2_animation(particle_history, simulation_times, filepath):
    """ 버섯구름 상승 및 형성 과정을 3D 애니메이션으로 생성. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig = plt.figure(figsize=(10, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # 전체 시뮬레이션 동안의 최대 좌표 범위를 계산하여 축 고정
    all_positions = np.vstack([frame['positions'] for frame in particle_history])
    max_x = np.max(np.abs(all_positions[:, 0])) * 1.1
    max_y = np.max(np.abs(all_positions[:, 1])) * 1.1
    max_z = np.max(all_positions[:, 2]) * 1.1
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.set_zlim(0, max_z)

    # 입자 종류에 따른 색상 맵
    # 화구(0): hot, 토양(1): copper
    colors = [cm.hot, cm.copper]
    
    # 애니메이션 성능을 위해 scatter 객체를 미리 생성
    scatter_plot = ax.scatter([], [], [], s=5)

    pbar = tqdm(total=len(particle_history), desc="Phase 2 애니메이션 렌더링", unit="frame")

    def update(frame_index):
        data = particle_history[frame_index]
        positions = data['positions']
        types = data['types']
        
        # 입자 타입에 따라 색상 결정
        particle_colors = np.empty((len(positions), 4))
        # 화구 입자 (type 0)
        mask_fireball = types == 0
        distances_fireball = np.linalg.norm(positions[mask_fireball], axis=1)
        if len(distances_fireball) > 0:
             # 중심에서 멀수록 어두워지는 효과
            intensity = 1 - (distances_fireball / (distances_fireball.max() + 1e-9))**2
            particle_colors[mask_fireball] = colors[0](intensity)

        # 토양 입자 (type 1)
        mask_soil = types == 1
        if len(positions[mask_soil]) > 0:
            # z 고도에 따라 색을 다르게 표현
            intensity = positions[mask_soil][:, 2] / max_z
            particle_colors[mask_soil] = colors[1](intensity)

        scatter_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scatter_plot.set_facecolor(particle_colors)

        ax.set_title(f'Phase 2: Cloud Rise | Time: {simulation_times[frame_index]:.2f} s', color='white')
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        pbar.update(1)
        return scatter_plot,

    ani = FuncAnimation(fig, update, frames=len(particle_history), interval=50, blit=False)
    ani.save(filepath, writer='ffmpeg', fps=20, dpi=150, savefig_kwargs={'facecolor': 'black'})
    pbar.close()
    plt.close(fig)
    print(f"\nPhase 2 애니메이션이 '{filepath}'에 저장되었습니다.")