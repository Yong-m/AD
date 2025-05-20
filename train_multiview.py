import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# 기본 설정
IMG_WIDTH, IMG_HEIGHT = 200, 66
N_FRAMES = 5
OUTPUT_DIM = 3
CAMERA_ORDER = ['front', 'left', 'right', 'rear_left', 'rear_right']
N_VIEWS = len(CAMERA_ORDER)
BATCH_SIZE = 2
SCALER_PATH = "label_scaler.pkl"

# ✅ Custom Loss
class VelocityDerivedAccelLoss(tf.keras.losses.Loss):
    def __init__(self, speed_weight=0.5, ang_weight=0.25, acc_weight=0.25,
                 var_weight=0.5, var_threshold=2.0, **kwargs):
        super().__init__(**kwargs)
        self.base_weights = tf.constant([speed_weight, ang_weight, acc_weight], dtype=tf.float32)
        self.var_weight = var_weight
        self.var_threshold = var_threshold

    def call(self, y_true, y_pred):
        base_error = tf.square(y_true - y_pred)
        base_loss = tf.reduce_sum(self.base_weights * base_error, axis=-1)
        v_pred = y_pred[:, 0]
        v_var = tf.math.reduce_variance(v_pred)
        var_penalty = tf.maximum(0.0, v_var - self.var_threshold)
        return base_loss + self.var_weight * var_penalty

# ✅ DataLoader
class MultiViewSequence(tf.keras.utils.Sequence):
    def __init__(self, video_root, label_root, mode='train', batch_size=2, n_chunks=10, scaler=None):
        self.video_root = video_root
        self.label_root = label_root
        self.mode = mode
        self.batch_size = batch_size
        self.n_chunks = n_chunks
        self.scaler = scaler  # StandardScaler

        self.video_groups = self._collect_groups()
        if mode == 'train':
            self.chunk_index = 0
            self._load_chunk()
        else:
            self.samples = self._load_all()

    def _collect_groups(self):
        prefixes = set()
        for cam in CAMERA_ORDER:
            cam_dir = os.path.join(self.video_root, cam)
            for fname in os.listdir(cam_dir):
                if fname.endswith('.mp4'):
                    prefixes.add(fname.replace('.mp4', ''))
        return sorted(list(prefixes))

    def _load_chunk(self):
        self.chunk_prefixes = self.video_groups[
            len(self.video_groups) * self.chunk_index // self.n_chunks:
            len(self.video_groups) * (self.chunk_index + 1) // self.n_chunks
        ]
        self.samples = self._load_specific(self.chunk_prefixes)

    def on_epoch_end(self):
        if self.mode == 'train':
            self.chunk_index = (self.chunk_index + 1) % self.n_chunks
            self._load_chunk()

    def _load_all(self):
        return self._load_specific(self.video_groups)

    def _load_specific(self, prefixes):
        samples = []
        for prefix in prefixes:
            frames_per_view = []
            for cam in CAMERA_ORDER:
                path = os.path.join(self.video_root, cam, prefix + '.mp4')
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

            label_path = os.path.join(self.label_root, prefix + '.txt')
            labels = np.loadtxt(label_path)
            min_len = min([len(v) for v in frames_per_view] + [len(labels)])
            for i in range(min_len - N_FRAMES):
                samples.append((frames_per_view, labels, i))
        return samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            frames_per_view, labels, start = self.samples[i]
            clip = [np.stack(view_frames[start:start+N_FRAMES], axis=0) for view_frames in frames_per_view]
            x = np.stack(clip, axis=0)
            y = labels[start + N_FRAMES - 1]
            x_batch.append(x)
            y_batch.append(y)

        y_array = np.array(y_batch)
        if self.scaler:
            y_array = self.scaler.transform(y_array)
        return np.array(x_batch), y_array

# ✅ 모델 정의
def get_multiview_model():
    input_shape = (N_VIEWS, N_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3)
    inputs = tf.keras.Input(shape=input_shape)

    def shared_3d_cnn():
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.GlobalAveragePooling3D()
        ])

    shared_cnn = shared_3d_cnn()
    features = tf.keras.layers.TimeDistributed(shared_cnn)(inputs)
    x = tf.keras.layers.Flatten()(features)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(OUTPUT_DIM)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=VelocityDerivedAccelLoss(), metrics=['mae'])
    return model

# ✅ 예측 시각화
def plot_predictions(y_true, y_pred, labels=["Speed", "Angular Z", "Acceleration"]):
    for i, label in enumerate(labels):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:, i], label=f'True {label}', linestyle='--')
        plt.plot(y_pred[:, i], label=f'Predicted {label}', linestyle='-')
        plt.title(f'{label} Prediction vs Ground Truth')
        plt.xlabel('Frame Index')
        plt.ylabel(label)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# ✅ 학습 실행
def main():
    # Step 1: StandardScaler 학습
    print("📊 Scaler 학습 중...")
    all_labels = []
    label_root = 'D:/data/multi_view/train_labels'
    for fname in os.listdir(label_root):
        if fname.endswith('.txt'):
            path = os.path.join(label_root, fname)
            labels = np.loadtxt(path)
            if labels.ndim == 1:
                labels = labels[np.newaxis, :]
            all_labels.append(labels)
    all_labels = np.concatenate(all_labels, axis=0)
    scaler = StandardScaler().fit(all_labels)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ 저장 완료: {SCALER_PATH}")

    # Step 2: 모델 및 데이터
    model = get_multiview_model()

    train_seq = MultiViewSequence(
        video_root='D:/data/multi_view',
        label_root='D:/data/multi_view/train_labels',
        mode='train',
        scaler=scaler
    )

    val_seq = MultiViewSequence(
        video_root='D:/data/validation',
        label_root='D:/data/validation/val_labels',
        mode='val',
        scaler=scaler
    )

    test_seq = MultiViewSequence(
        video_root='D:/data/test',
        label_root='D:/data/test/test_labels',
        mode='test',
        scaler=scaler
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_seq, validation_data=val_seq, epochs=50, callbacks=[early_stop])
    model.save('multiview_model_normalized.keras')

    print("\n✅ 모델 저장 완료. 테스트 성능:")
    model.evaluate(test_seq)

    # Step 3: 시각화
    all_y_true, all_y_pred = [], []
    for x_batch, y_batch in test_seq:
        preds = model.predict(x_batch, verbose=0)
        all_y_pred.extend(scaler.inverse_transform(preds))
        all_y_true.extend(scaler.inverse_transform(y_batch))

    plot_predictions(np.array(all_y_true), np.array(all_y_pred))


if __name__ == "__main__":
    main()
