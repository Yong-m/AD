import os
import cv2
import numpy as np
import tensorflow as tf

class WaymoDatasetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, mode='train', img_width=200, img_height=66):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height

        self.images = []
        self.labels = []

        if self.mode == 'train':
            # ✅ 모든 train_*.mp4, train_*.txt 파일 로드
            video_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('train_') and f.endswith('.mp4')])
            txt_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('train_') and f.endswith('.txt')])

            assert len(video_files) == len(txt_files), "Mismatch between train videos and labels."

            for vfile, tfile in zip(video_files, txt_files):
                video_path = os.path.join(self.data_dir, vfile)
                label_path = os.path.join(self.data_dir, tfile)

                print(f"✅ Loading {vfile} and {tfile}...")
                imgs, lbls = self._load_video_and_labels(video_path, label_path)
                self.images.extend(imgs)
                self.labels.extend(lbls)

        else:
            # ✅ validation/test는 test.mp4 하나
            self.video_path = os.path.join(self.data_dir, 'test.mp4')
            self.label_path = os.path.join(self.data_dir, 'test.txt')

            self.speeds = self._load_labels(self.label_path)
            self.n_frames = len(self.speeds)

            if self.mode == 'val':
                self.start_idx = 0
                self.end_idx = self.n_frames // 2
            elif self.mode == 'test':
                self.start_idx = self.n_frames // 2
                self.end_idx = self.n_frames
            else:
                raise ValueError("Invalid mode! Choose 'train', 'val', or 'test'.")

            self.indexes = np.arange(self.start_idx, self.end_idx - 1)

            imgs, lbls = self._load_video_and_labels(self.video_path, self.label_path, self.indexes)
            self.images.extend(imgs)
            self.labels.extend(lbls)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        print(f"✅ {self.mode} dataset loaded: {len(self.images)} samples")

    def _load_labels(self, label_path):
        with open(label_path, 'r') as f:
            speeds = [float(line.strip()) for line in f]
        return speeds

    def _load_video_and_labels(self, video_path, label_path, indexes=None):
        images = []
        labels = []

        speeds = self._load_labels(label_path)
        n_frames = len(speeds)

        if indexes is None:
            indexes = np.arange(0, n_frames - 1)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"🚨 Cannot open video file: {video_path}")

        for idx in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()

            if not ret1 or not ret2:
                continue

            img = self.calc_optical_flow(frame1, frame2)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0

            v1 = speeds[idx]
            v2 = speeds[idx+1]
            mean_speed = 0.5 * (v1 + v2)

            images.append(img)
            labels.append(mean_speed)

        cap.release()

        return images, labels

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        x_batch = self.images[start:end]
        y_batch = self.labels[start:end]

        return x_batch, y_batch

    def calc_optical_flow(self, frame1, frame2):
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=1,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.3,
            flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr
