import os
import cv2
import pandas as pd
import numpy as np

# 설정
camera_dir = r"E:\dataset\testing\camera_image"  # parquet 파일 폴더 (test용)
output_dir = r"C:\Users\User\PycharmProjects\AD\data\test"
os.makedirs(output_dir, exist_ok=True)

IMG_WIDTH = 1280
IMG_HEIGHT = 720
FPS = 10

video_path = os.path.join(output_dir, "test.mp4")
txt_path = os.path.join(output_dir, "test.txt")

frame_list = []

# parquet 파일 정렬
camera_files = sorted([
    f for f in os.listdir(camera_dir) if f.endswith(".parquet")
])

print(f"총 {len(camera_files)}개의 .parquet 파일 로딩 시작")

for f_idx, filename in enumerate(camera_files):
    file_path = os.path.join(camera_dir, filename)
    try:
        df = pd.read_parquet(file_path)

        for idx in range(len(df)):
            row = df.iloc[idx]

            # ✅ FRONT 카메라만 사용
            if row['key.camera_name'] != 1:
                continue

            try:
                timestamp = row['key.frame_timestamp_micros']
                img_data = row['[CameraImageComponent].image']
                vx = row['[CameraImageComponent].velocity.linear_velocity.x']
                vy = row['[CameraImageComponent].velocity.linear_velocity.y']
                speed = np.sqrt(vx ** 2 + vy ** 2)

                frame_list.append({
                    "timestamp": timestamp,
                    "image": img_data,
                    "speed": speed
                })

            except Exception as inner_e:
                print(f"⚠️ 내부 row 에러 (파일 {filename}): {inner_e}")

    except Exception as e:
        print(f"⚠️ 파일 로드 에러: {filename} → {e}")

print(f"총 {len(frame_list)}개의 전면 카메라 프레임 수집 완료")

# ✅ timestamp 기준 정렬
frame_list.sort(key=lambda x: x['timestamp'])

# 비디오 + 텍스트 저장
video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (IMG_WIDTH, IMG_HEIGHT)
)
label_file = open(txt_path, "w")

for f in frame_list:
    img_np = np.frombuffer(f["image"], dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        continue

    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) if (img.shape[1], img.shape[0]) != (IMG_WIDTH, IMG_HEIGHT) else img
    video_writer.write(img_resized)
    label_file.write(f"{f['speed']:.4f}\n")

label_file.close()
video_writer.release()

print(f"✅ Test 변환 완료: {video_path}, {txt_path}")
