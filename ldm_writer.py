# LDM 입력 파일 생성

import csv
import os
import numpy as np
import pandas as pd 

def save_phase1_results_to_csv(summary_data, filepath):
    """Phase 1의 요약 결과를 CSV 파일로 저장."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(summary_data.keys())
        writer.writerow(summary_data.values())
    print(f"요약 결과가 '{filepath}'에 저장되었습니다.")


def save_particle_data_to_csv(positions, velocities, filepath):
    """ Numpy 배열 형태의 입자 위치와 속도 데이터를 CSV 파일로 저장. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 위치와 속도 배열을 수평으로 합침
    data = np.hstack((positions, velocities))
    # pandas DataFrame을 사용하여 헤더와 함께 저장
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    df.to_csv(filepath, index=False)
    
    print(f"최종 입자 상태(위치, 속도)가 '{filepath}'에 저장되었습니다.")

def save_particle_data_phase2(positions, types, filepath):
    """ Phase 2의 최종 입자 상태(위치, 타입)를 CSV 파일로 저장. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 위치와 타입을 합쳐서 DataFrame 생성
    data = {
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'type': types
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    print(f"Phase 2 최종 입자 상태가 '{filepath}'에 저장되었습니다.")