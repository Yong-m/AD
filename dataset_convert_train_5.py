import os
import cv2
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc

# # 설정
# camera_dir = r"E:\dataset\camera_image"  # parquet 파일 폴더
# output_dir = r"D:\data\multi_view"
# camera_dir = r"E:\dataset\testing\camera_image"  # parquet 파일 폴더
# output_dir = r"D:\data\test"
camera_dir = r"E:\dataset\validation"  # parquet 파일 폴더
output_dir = r"D:\data\validation"
os.makedirs(output_dir, exist_ok=True)

IMG_WIDTH = 1280
IMG_HEIGHT = 720
FPS = 10
FRAMES_PER_VIDEO = 5000

camera_id_to_name = {
    1: "front",
    2: "left",
    3: "right",
    4: "rear_left",
    5: "rear_right",
}

# parquet 파일 리스트
camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith(".parquet")])
print(f"총 {len(camera_files)}개의 .parquet 파일 로딩 시작")

label_output_dir = os.path.join(output_dir, "train_labels")
os.makedirs(label_output_dir, exist_ok=True)

for f_idx, filename in enumerate(camera_files):
    file_path = os.path.join(camera_dir, filename)
    print(f"\n▶ 처리 중: {filename}")

    view_frames = {name: [] for name in camera_id_to_name.values()}
    front_frame_info = []
    label_list = []

    try:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=500, columns=[
            'key.camera_name', 'key.frame_timestamp_micros',
            '[CameraImageComponent].image',
            '[CameraImageComponent].velocity.linear_velocity.x',
            '[CameraImageComponent].velocity.linear_velocity.y',
            '[CameraImageComponent].velocity.angular_velocity.z']):
            try:
                df = batch.to_pandas()
            except Exception as e:
                print(f"⚠️ batch.to_pandas 에러: {e}")
                break

            for idx in range(len(df)):
                row = df.iloc[idx]
                cam_id = row['key.camera_name']
                cam_name = camera_id_to_name.get(cam_id, None)
                if cam_name is None:
                    continue

                try:
                    timestamp = row['key.frame_timestamp_micros']
                    img_data = row['[CameraImageComponent].image']
                    vx = row['[CameraImageComponent].velocity.linear_velocity.x']
                    vy = row['[CameraImageComponent].velocity.linear_velocity.y']
                    speed = np.sqrt(vx ** 2 + vy ** 2)
                    angular_z = row['[CameraImageComponent].velocity.angular_velocity.z']

                    view_frames[cam_name].append({
                        "timestamp": timestamp,
                        "image": img_data,
                    })

                    if cam_name == "front":
                        front_frame_info.append({
                            "timestamp": timestamp,
                            "speed": speed,
                            "angular_vel_z": angular_z
                        })
                except Exception as inner_e:
                    print(f"⚠️ row 처리 에러: {inner_e}")

    except Exception as e:
        print(f"⚠️ 파일 로드 실패: {filename} → {e}")
        continue

    # 각 뷰별로 mp4 저장
    for cam_name, frames in view_frames.items():
        if not frames:
            continue
        frames.sort(key=lambda x: x['timestamp'])

        cam_dir = os.path.join(output_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        video_path = os.path.join(cam_dir, f"{filename.replace('.parquet', '.mp4')}")

        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            (IMG_WIDTH, IMG_HEIGHT)
        )

        for f in frames:
            try:
                img_np = np.frombuffer(f["image"], dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) \
                    if (img.shape[1], img.shape[0]) != (IMG_WIDTH, IMG_HEIGHT) else img
                video_writer.write(img_resized)
            except Exception as im_err:
                print(f"⚠️ 이미지 디코딩 에러: {im_err}")

        video_writer.release()
        print(f"✅ {cam_name} → {video_path} 저장 완료")

    # front 기준 라벨 계산 및 저장
    if front_frame_info:
        front_frame_info.sort(key=lambda x: x['timestamp'])
        for i in range(len(front_frame_info) - 1):
            cur = front_frame_info[i]
            nxt = front_frame_info[i + 1]
            dt = (nxt['timestamp'] - cur['timestamp']) / 1e6
            accel = 0.0 if dt == 0 else (nxt['speed'] - cur['speed']) / dt
            label_list.append((cur['speed'], cur['angular_vel_z'], accel))

        label_filename = filename.replace(".parquet", ".txt")
        label_path = os.path.join(label_output_dir, label_filename)
        with open(label_path, "w") as label_file:
            for speed, angular_z, accel in label_list:
                label_file.write(f"{speed:.4f} {angular_z:.6f} {accel:.6f}\n")

        print(f"✅ front → {label_path} 저장 완료")

    # 메모리 정리
    del view_frames, df, batch, front_frame_info, label_list
    gc.collect()

print("\n✅ 모든 변환 완료: 각 parquet 파일별 mp4 및 라벨 분리 저장됨")