import os
import cv2
import numpy as np
import tensorflow as tf

class WaymoDatasetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, mode='train', img_width=200, img_height=66, n_frames=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.n_frames = n_frames

        self.frames, self.speeds = self._load_all_frames_and_speeds()
        self.n_total_frames = len(self.frames)

        self.indexes = np.arange(self.n_total_frames - self.n_frames)

        print(f"✅ 전체 frame 메모리 로드 완료: {self.n_total_frames} frames")

    def _load_all_frames_and_speeds(self):
        video_files = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.mp4')
        ])
        txt_files = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.txt')
        ])
        assert len(video_files) == len(txt_files), "Video와 Label 파일 개수가 다릅니다."

        frames = []
        speeds = []

        for vfile, tfile in zip(video_files, txt_files):
            video_path = os.path.join(self.data_dir, vfile)
            label_path = os.path.join(self.data_dir, tfile)

            with open(label_path, 'r') as f:
                video_speeds = [float(line.strip()) for line in f]

            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            speeds.extend(video_speeds)
            cap.release()

        return np.array(frames), np.array(speeds)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indexes = self.indexes[start:end]

        x_batch = []
        y_batch = []

        for start_idx in batch_indexes:
            frames_stack = self.frames[start_idx:start_idx+self.n_frames]  # (n_frames, 66, 200, 3)

            speeds_slice = self.speeds[start_idx:start_idx+self.n_frames]
            mean_speeds = []
            for j in range(self.n_frames - 1):
                mean_speed = 0.5 * (speeds_slice[j] + speeds_slice[j+1])
                mean_speeds.append(mean_speed)

            x_batch.append(frames_stack)
            y_batch.append(mean_speeds)

        return np.array(x_batch), np.array(y_batch)
