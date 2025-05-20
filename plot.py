import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from train_multiview import VelocityDerivedAccelLoss, CAMERA_ORDER, IMG_WIDTH, IMG_HEIGHT, N_FRAMES, OUTPUT_DIM

FRAME_WEIGHTS = {1: 0.25, 2: 0.5, 3: 0.25}
N_VIEWS = len(CAMERA_ORDER)

# 모델 로드
model = tf.keras.models.load_model(
    "multiview_model_flattened_valsmooth.keras",
    custom_objects={'VelocityDerivedAccelLoss': VelocityDerivedAccelLoss}
)

video_root = 'D:/data/test'
label_root = 'D:/data/test/test_labels'

prefixes = sorted([
    fname.replace('.mp4', '')
    for fname in os.listdir(os.path.join(video_root, 'front'))
    if fname.endswith('.mp4')
])

# 프레임 하나씩 예측하는 함수
def load_test_clip(prefix):
    frames_per_view = []
    for cam in CAMERA_ORDER:
        path = os.path.join(video_root, cam, prefix + '.mp4')
        cap = cv2.VideoCapture(path)
        view_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            view_frames.append(frame)
        cap.release()
        frames_per_view.append(view_frames)

    label_path = os.path.join(label_root, prefix + '.txt')
    labels = np.loadtxt(label_path)
    min_len = min([len(v) for v in frames_per_view] + [len(labels)])
    samples = []
    for i in range(min_len - N_FRAMES):
        clip = [np.stack(view_frames[i:i+N_FRAMES], axis=0) for view_frames in frames_per_view]
        x = np.stack(clip, axis=0)
        y = labels[i + N_FRAMES - 1]
        samples.append((x, y, i))
    return samples

# 누적 저장용
all_y_true, all_y_pred = [], []
pos_gt_dict = {i: [] for i in range(N_FRAMES)}
pos_pred_dict = {i: [] for i in range(N_FRAMES)}

for prefix in prefixes:
    print(f"🔍 Processing {prefix}")
    samples = load_test_clip(prefix)
    pred_dict = {}
    gt_dict = {}

    for x, y, start in samples:
        x_batch = np.expand_dims(x, axis=0)
        pred = model.predict(x_batch, verbose=0)[0]

        # 프레임 위치별 예측 저장
        for pos in range(N_FRAMES):
            pos_gt_dict[pos].append(y)
            pos_pred_dict[pos].append(pred)

        # smoothing용
        for rel_idx, w in FRAME_WEIGHTS.items():
            idx = start + rel_idx
            if idx not in pred_dict:
                pred_dict[idx] = [(pred, w)]
                gt_dict[idx] = y
            else:
                pred_dict[idx].append((pred, w))

    for key in sorted(pred_dict.keys()):
        preds = pred_dict[key]
        weighted = sum(p * w for p, w in preds) / sum(w for _, w in preds)
        all_y_pred.append(weighted)
        all_y_true.append(gt_dict[key])

# ✅ 전체 smoothed plot
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

plt.figure(figsize=(10, 4))
plt.plot(all_y_true[:, 0], label='True Speed', linestyle='--')
plt.plot(all_y_pred[:, 0], label='Smoothed Predicted Speed', linestyle='-')
plt.title("Smoothed Speed Prediction vs Ground Truth")
plt.xlabel("Frame Index")
plt.ylabel("Speed")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ✅ 위치별 프레임 plot
for pos in range(N_FRAMES):
    gt = np.array(pos_gt_dict[pos])
    pred = np.array(pos_pred_dict[pos])
    plt.figure(figsize=(10, 4))
    plt.plot(gt[:, 0], label='True Speed', linestyle='--')
    plt.plot(pred[:, 0], label='Predicted Speed', linestyle='-')
    plt.title(f"Speed Prediction at Frame Position {pos}")
    plt.xlabel("Sample Index")
    plt.ylabel("Speed")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
