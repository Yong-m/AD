import os
import cv2
import numpy as np
import tensorflow as tf

# Configuration
IMG_WIDTH, IMG_HEIGHT = 200, 66
N_FRAMES = 5
OUTPUT_DIM = 3
CAMERA_ORDER = ['front', 'left', 'right', 'rear_left', 'rear_right']
N_VIEWS = len(CAMERA_ORDER)
BATCH_SIZE = 2
DT = 0.1  # time step (assuming 10 FPS)

# Custom loss function
class MultiOutputSmoothLoss(tf.keras.losses.Loss):
    def __init__(self, base_weight=1.0, smooth_weight=0.5, acc_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_weight = base_weight
        self.smooth_weight = smooth_weight
        self.acc_weight = acc_weight

    def call(self, y_true, y_pred):
        base_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        v = y_pred[:, :, 0]
        smoothness = tf.reduce_mean(tf.square(v[:, 1:] - v[:, :-1]))
        acc_pred = (v[:, 1:] - v[:, :-1]) / DT
        acc_gt = y_true[:, 1:, 2]
        acc_loss = tf.reduce_mean(tf.square(acc_pred - acc_gt))
        return self.base_weight * base_loss + self.smooth_weight * smoothness + self.acc_weight * acc_loss

# Data loader
class MultiViewSequence(tf.keras.utils.Sequence):
    def __init__(self, video_root, label_root, mode='train', batch_size=2, n_chunks=10):
        self.video_root = video_root
        self.label_root = label_root
        self.mode = mode
        self.batch_size = batch_size
        self.n_chunks = n_chunks
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
            for i in range(min_len - N_FRAMES + 1):
                x_clip = [np.stack(view_frames[i:i+N_FRAMES], axis=0) for view_frames in frames_per_view]
                y_clip = labels[i:i+N_FRAMES]
                samples.append((np.stack(x_clip, axis=0), y_clip))
        return samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            x, y = self.samples[i]
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

# LSTM model
def get_multioutput_lstm_model():
    inputs = tf.keras.Input(shape=(N_VIEWS, N_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3))

    def shared_3d_cnn():
        return tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.GlobalAveragePooling3D()
        ])

    shared_cnn = shared_3d_cnn()
    view_features = tf.keras.layers.TimeDistributed(shared_cnn)(inputs)
    x = tf.keras.layers.Reshape((N_FRAMES, -1))(view_features)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(OUTPUT_DIM))(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=MultiOutputSmoothLoss(), metrics=['mae'])
    return model

# Train & Evaluate
def main():
    model = get_multioutput_lstm_model()

    train_seq = MultiViewSequence(
        video_root='D:/data/multi_view',
        label_root='D:/data/multi_view/train_labels',
        mode='train'
    )
    val_seq = MultiViewSequence(
        video_root='D:/data/validation',
        label_root='D:/data/validation/val_labels',
        mode='val'
    )
    test_seq = MultiViewSequence(
        video_root='D:/data/test',
        label_root='D:/data/test/test_labels',
        mode='test'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_seq, validation_data=val_seq, epochs=30, callbacks=[early_stop])
    model.save('multiview_model_multioutput_lstm.keras')

    print("\n✅ 모델 저장 완료. 테스트 성능:")
    model.evaluate(test_seq)

if __name__ == '__main__':
    main()
