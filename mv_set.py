import os
import shutil
from sklearn.model_selection import train_test_split

# 설정
VAL_ROOT = 'D:/data/validation'
TRAIN_ROOT = 'D:/data/multi_view'
LABEL_NAME = 'val_labels'

CAMERA_ORDER = ['front', 'left', 'right', 'rear_left', 'rear_right']

# Step 1: prefix 수집
prefixes = []
front_dir = os.path.join(VAL_ROOT, 'front')
for fname in os.listdir(front_dir):
    if fname.endswith('.mp4'):
        prefix = fname.replace('.mp4', '')
        label_path = os.path.join(VAL_ROOT, LABEL_NAME, prefix + '.txt')
        if os.path.isfile(label_path):  # 라벨도 있어야 함
            valid = all(os.path.isfile(os.path.join(VAL_ROOT, cam, prefix + '.mp4')) for cam in CAMERA_ORDER)
            if valid:
                prefixes.append(prefix)

# Step 2: train/val 나누기
train_prefixes, remain_val_prefixes = train_test_split(prefixes, test_size=0.25, random_state=42)

print(f"총 prefix 수: {len(prefixes)}")
print(f"Train으로 이동할 prefix 수: {len(train_prefixes)}")
print(f"Validation에 남길 prefix 수: {len(remain_val_prefixes)}")

# Step 3: 파일 이동
for prefix in train_prefixes:
    # 각 view의 mp4 파일 이동
    for cam in CAMERA_ORDER:
        src = os.path.join(VAL_ROOT, cam, prefix + '.mp4')
        dst_dir = os.path.join(TRAIN_ROOT, cam)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, prefix + '.mp4')
        shutil.move(src, dst)

    # txt 라벨 이동
    src_label = os.path.join(VAL_ROOT, LABEL_NAME, prefix + '.txt')
    dst_label_dir = os.path.join(TRAIN_ROOT, 'train_labels')
    os.makedirs(dst_label_dir, exist_ok=True)
    dst_label = os.path.join(dst_label_dir, prefix + '.txt')
    shutil.move(src_label, dst_label)

print("✅ 파일 이동 완료")
