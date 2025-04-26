import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from WaymoDatasetLoader import WaymoDatasetLoader

# 설정
IMG_WIDTH = 200
IMG_HEIGHT = 66
MAX_EPOCHS = 30

def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train_model.py model_directory")

    saved_model_dir = sys.argv[1]

    # ✅ 데이터 로더 준비
    train_loader = WaymoDatasetLoader(
        data_dir="data/train",
        batch_size=32,
        mode='train',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )
    val_loader = WaymoDatasetLoader(
        data_dir="data/test",
        batch_size=32,
        mode='val',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )
    test_loader = WaymoDatasetLoader(
        data_dir="data/test",
        batch_size=32,
        mode='test',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )

    print(f"Dataset ready: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

    # 모델 준비
    model = get_model()

    # ✅ EarlyStopping 콜백 추가
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # 모델 학습
    print("Training model...")
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    # 테스트 데이터 평가
    print("Evaluating on test set...")
    test_loss = model.evaluate(test_loader, verbose=2)
    print(f"Test loss: {test_loss:.4f}")

    # 모델 저장
    model_file = os.path.join('saved_models', saved_model_dir)
    os.makedirs('saved_models', exist_ok=True)
    model.save(model_file)
    print(f"Model saved to {model_file}.")

    # ✅ Test Set 예측
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_loader:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 평가지표 출력
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"✅ Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Loss 시각화
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Ground Truth vs Prediction 비교
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Ground Truth Speed', linestyle='--')
    plt.plot(y_pred, label='Predicted Speed', linestyle='-')
    plt.title('Ground Truth vs Predicted Speed on Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()

def get_model():
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=input_shape),
        tf.keras.layers.Conv2D(24, (5, 5), activation='elu', strides=(2, 2)),
        tf.keras.layers.Conv2D(36, (5, 5), activation='elu', strides=(2, 2)),
        tf.keras.layers.Conv2D(48, (5, 5), activation='elu', strides=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', strides=(1, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', strides=(1, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='elu'),
        tf.keras.layers.Dense(50, activation='elu'),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(1, name='output')
    ])

    model.compile(optimizer='adam', loss="mse")
    return model

if __name__ == "__main__":
    main()
