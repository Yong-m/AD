import cv2
import numpy as np
import os

CAMERA_ORDER = ['front', 'left', 'right', 'rear_left', 'rear_right']
input_base = r"D:\data\test"
video_id = "11987368976578218644_1340_000_1360_000"
output_path = r"C:\Users\User\PycharmProjects\AD\data\input.mp4"

frames = []

# 각 카메라에서 마지막 프레임 추출
for cam in CAMERA_ORDER:
    path = os.path.join(input_base, cam, f"{video_id}.mp4")
    cap = cv2.VideoCapture(path)
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
    cap.release()

    if last_frame is None:
        raise RuntimeError(f"No frame found in {path}")
    if last_frame.ndim == 2 or last_frame.shape[2] == 1:
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)

    resized = cv2.resize(last_frame, (848, 480))
    frames.append(resized)

# 마지막 view: rear (없는 경우 시커먼 화면)
black_frame = np.zeros((480, 848, 3), dtype=np.uint8)
frames.append(black_frame)

# 저장
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 1.0, (848, 480))
for f in frames:
    out.write(f)
out.release()

print(f"✅ Saved multiview input as {output_path}")
