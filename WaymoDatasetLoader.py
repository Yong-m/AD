import os
import threading
import cv2
import numpy as np
import tensorflow as tf

class WaymoDatasetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, mode='train', img_width=200, img_height=66,
                 prefetch_size=20000, prefetch_threshold=0.7):
        """
        data_dir: 'data/train' 또는 'data/test'
        mode: 'train', 'val', 'test'
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.prefetch_size = prefetch_size
        self.prefetch_threshold = int(prefetch_size * prefetch_threshold)

        self.cached_images = []
        self.cached_labels = []
        self.cache_pointer = 0
        self.prefetching = False
        self.lock = threading.Lock()

        if self.mode == 'train':
            # train은 여러 train_*.mp4, train_*.txt
            self.video_txt_pairs = self._load_train_pairs()
            self.indexes = np.arange(len(self.video_txt_pairs))
        else:
            # val/test는 test.mp4, test.txt
            self.video_path = os.path.join(data_dir, 'test.mp4')
            self.label_path = os.path.join(data_dir, 'test.txt')

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

        # ✅ 처음 prefetch
        self._prefetch()

    def _load_train_pairs(self):
        video_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".mp4")])
        txt_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".txt")])

        assert len(video_files) == len(txt_files), "Mismatch between train videos and txt files!"

        pairs = []
        for v, t in zip(video_files, txt_files):
            v_path = os.path.join(self.data_dir, v)
            t_path = os.path.join(self.data_dir, t)
            speeds = self._load_labels(t_path)
            pairs.append((v_path, speeds))

        return pairs

    def _load_labels(self, label_path):
        with open(label_path, 'r') as f:
            speeds = [float(line.strip()) for line in f]
        return speeds

    def _prefetch(self):
        def prefetch_worker():
            new_images = []
            new_labels = []

            if self.mode == 'train':
                remaining = self.prefetch_size
                while remaining > 0:
                    video_idx = np.random.choice(self.indexes)
                    video_path, speeds = self.video_txt_pairs[video_idx]

                    cap = cv2.VideoCapture(video_path)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if n_frames < 2:
                        cap.release()
                        continue

                    start_idx = np.random.randint(0, n_frames - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

                    ret1, frame1 = cap.read()
                    ret2, frame2 = cap.read()
                    cap.release()

                    if not ret1 or not ret2:
                        continue

                    img = self.calc_optical_flow(frame1, frame2)
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    img = img.astype(np.float32) / 255.0

                    v1 = speeds[start_idx]
                    v2 = speeds[start_idx+1]
                    mean_speed = 0.5 * (v1 + v2)

                    new_images.append(img)
                    new_labels.append(mean_speed)

                    remaining -= 1

            else:
                cap = cv2.VideoCapture(self.video_path)
                for idx in self.indexes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret1, frame1 = cap.read()
                    ret2, frame2 = cap.read()

                    if not ret1 or not ret2:
                        continue

                    img = self.calc_optical_flow(frame1, frame2)
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    img = img.astype(np.float32) / 255.0

                    v1 = self.speeds[idx]
                    v2 = self.speeds[idx+1]
                    mean_speed = 0.5 * (v1 + v2)

                    new_images.append(img)
                    new_labels.append(mean_speed)

                    if len(new_images) >= self.prefetch_size:
                        break

                cap.release()

            with self.lock:
                self.cached_images = new_images
                self.cached_labels = new_labels
                self.cache_pointer = 0
                self.prefetching = False
            print(f"✅ Prefetch 완료 ({len(new_images)} frames)")

        self.prefetching = True
        threading.Thread(target=prefetch_worker).start()

    def __len__(self):
        return int(np.floor(self.prefetch_size / self.batch_size))

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        with self.lock:
            if self.cache_pointer + self.batch_size >= len(self.cached_images):
                while self.prefetching:
                    pass  # Prefetch가 끝날 때까지 기다림
                self._prefetch()

            if (not self.prefetching) and (self.cache_pointer >= self.prefetch_threshold):
                self._prefetch()

            x_batch = np.array(self.cached_images[self.cache_pointer:self.cache_pointer+self.batch_size])
            y_batch = np.array(self.cached_labels[self.cache_pointer:self.cache_pointer+self.batch_size])

            self.cache_pointer += self.batch_size

        return x_batch, y_batch

    def calc_optical_flow(self, frame1, frame2):
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 1, 15, 2, 5, 1.3, 0
        )

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr
