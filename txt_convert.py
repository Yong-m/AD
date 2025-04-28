import os
import pandas as pd
import numpy as np

# 설정
camera_dir = r"E:\dataset\testing\camera_image"   # parquet 파일 폴더
output_txt_path = r"C:\Users\User\PycharmProjects\AD\data\train_txt_only"  # 저장할 위치
os.makedirs(output_txt_path, exist_ok=True)

# ✅ 만들어진 mp4는 그대로 두고, txt만 다시 생성
camera_files = sorted([
    f for f in os.listdir(camera_dir) if f.endswith(".parquet")
])

# Frame 데이터 모아두기
frame_list = []

print(f"총 {len(camera_files)}개의 parquet 파일에서 속도 데이터 추출 시작")

for f_idx, filename in enumerate(camera_files):
    file_path = os.path.join(camera_dir, filename)
    try:
        df = pd.read_parquet(file_path)

        for idx in range(len(df)):
            row = df.iloc[idx]

            # ✅ Front 카메라만 사용
            if row['key.camera_name'] != 1:
                continue

            try:
                # vx, vy 가져오기
                vx = row['[CameraImageComponent].velocity.linear_velocity.x']
                vy = row['[CameraImageComponent].velocity.linear_velocity.y']

                # ✅ magnitude 계산 (진짜 speed)
                speed = np.sqrt(vx**2 + vy**2)

                # 비정상 데이터는 skip
                if np.isnan(speed) or speed > 50:
                    continue

                # Frame 추가
                frame_list.append(speed)

            except Exception as inner_e:
                print(f"⚠️ 내부 row 에러 (파일 {filename}): {inner_e}")

    except Exception as e:
        print(f"⚠️ 파일 로드 에러: {filename} → {e}")

print(f"✅ 총 {len(frame_list)}개의 속도 데이터 수집 완료")

# ✅ txt 파일로 저장
output_txt_file = os.path.join(output_txt_path, "train_speed.txt")

with open(output_txt_file, "w") as f:
    for speed in frame_list:
        f.write(f"{speed:.4f}\n")

print(f"✅ Speed txt 저장 완료: {output_txt_file}")
