import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from WaymoDatasetLoader_3D import WaymoDatasetLoader
from get_model_3dcnn_with_attention import get_model_3dcnn_with_attention
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

IMG_WIDTH = 200
IMG_HEIGHT = 66
N_FRAMES = 5
MAX_EPOCHS = 30

def main():
    saved_model_dir = get_new_version_dir()

    train_loader = WaymoDatasetLoader(
        data_dir="data/train",
        batch_size=32,
        mode='train',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        n_frames=N_FRAMES
    )
    val_loader = WaymoDatasetLoader(
        data_dir="data/test",
        batch_size=32,
        mode='val',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        n_frames=N_FRAMES
    )
    test_loader = WaymoDatasetLoader(
        data_dir="data/test",
        batch_size=32,
        mode='test',
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        n_frames=N_FRAMES
    )

    model = get_model_3dcnn_with_attention(
        n_frames=N_FRAMES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    model.save(saved_model_dir)
    print(f"✅ Model saved to {saved_model_dir}")

    evaluate_model(model, test_loader)
    plot_loss(history)

def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_loader:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    print(f"✅ Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # ✅ Plot 추가
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.flatten(), label='Ground Truth Speed', linestyle='--')
    plt.plot(y_pred.flatten(), label='Predicted Speed', linestyle='-')
    plt.title('Ground Truth vs Predicted Speed on Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid()
    plt.show()

def get_new_version_dir(base_dir='saved_models'):
    os.makedirs(base_dir, exist_ok=True)
    existing_versions = [
        int(d.replace('version_', '')) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('version_')
    ]
    next_version = max(existing_versions) + 1 if existing_versions else 0
    new_version_dir = os.path.join(base_dir, f"version_{next_version}")
    os.makedirs(new_version_dir, exist_ok=True)
    return new_version_dir

if __name__ == "__main__":
    main()
