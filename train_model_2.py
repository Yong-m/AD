import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

# 하이퍼파라미터 설정
TEST_SIZE = 0.2
EPOCHS = 10
IMG_WIDTH = 200
IMG_HEIGHT = 66
FPS = 20

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train_model_rgb.py model_directory")

    saved_model_dir = sys.argv[1]

    # 데이터 경로
    video_filename = os.path.join('data', 'train.mp4')
    label_filename = os.path.join('data', 'train.txt')

    # Load data
    images, labels = load_data(video_filename, label_filename)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, shuffle=False
    )

    # Build model
    model = get_rgb_model()

    # Train model
    print("Training model")
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate
    test_loss = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_loss:.4f}")

    # Save model
    model_file = os.path.join('saved_models', saved_model_dir)
    model.save(model_file)
    print(f"Model saved to {model_file}.")

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Predict
    y_pred = model.predict(x_test)

    # 평가지표 계산
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot loss
    plt.plot(history.history['loss'], label='train loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

    # Predict and plot Ground Truth vs Prediction
    y_pred = model.predict(x_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.flatten(), label='Ground Truth Speed', linestyle='--')
    plt.plot(y_pred.flatten(), label='Predicted Speed', linestyle='-')
    plt.title('Ground Truth vs Predicted Speed on Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()

def load_data(video_filename, label_filename):
    """
    Load RGB frames and labels.
    """

    cap = cv2.VideoCapture(video_filename)
    speeds = load_labels(label_filename)

    images = []
    labels = []

    print("Loading frames (RGB only)")
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        # ✅ Optical Flow 없이 RGB 프레임 그대로 사용
        img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        try:
            v = next(speeds)
        except StopIteration:
            break

        images.append(img)
        labels.append(v)

    cap.release()
    cv2.destroyAllWindows()

    print("Loading Frames Done!")

    return np.array(images), np.array(labels)

def load_labels(filename):
    """
    Yield speeds from txt file line by line
    """

    for row in open(filename, 'r'):
        yield float(row)

def get_rgb_model():
    """
    Build CNN model for RGB input
    """

    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=input_shape),

        tf.keras.layers.Conv2D(32, (5, 5), activation='elu', strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='elu', strides=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='elu', strides=(2, 2)),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(1, name='output')
    ])

    model.compile(optimizer='adam', loss='mse')

    return model

if __name__ == "__main__":
    main()
