import tensorflow as tf
import numpy as np
import itertools
import gc

from train_multiview import MultiViewSequence, CAMERA_ORDER, IMG_WIDTH, IMG_HEIGHT, N_FRAMES, OUTPUT_DIM

class VelocitySmoothVarianceLoss(tf.keras.losses.Loss):
    def __init__(self, speed_weight=0.5, ang_weight=0.25, acc_weight=0.25,
                 var_weight=0.2, var_threshold=4.0, **kwargs):
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

def build_model(config):
    input_shape = (len(CAMERA_ORDER), N_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3)
    inputs = tf.keras.Input(shape=input_shape)

    def shared_3d_cnn():
        layers = []
        for _ in range(config['cnn_depth']):
            layers.append(tf.keras.layers.Conv3D(config['cnn_filters'], config['cnn_kernel'],
                                                  activation='relu', padding='same'))
            layers.append(tf.keras.layers.MaxPooling3D(config['pool_size']))
        layers.append(tf.keras.layers.GlobalAveragePooling3D())
        return tf.keras.Sequential(layers)

    shared_cnn = shared_3d_cnn()
    view_features = tf.keras.layers.TimeDistributed(shared_cnn)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(view_features)

    if config['dropout'] > 0.0:
        x = tf.keras.layers.Dropout(config['dropout'])(x)

    for _ in range(config['mlp_layers']):
        x = tf.keras.layers.Dense(config['dense_units'], activation='relu')(x)

    output = tf.keras.layers.Dense(OUTPUT_DIM)(x)
    model = tf.keras.Model(inputs, output)

    loss_fn = VelocitySmoothVarianceLoss(
        var_weight=config['var_weight'],
        var_threshold=config['var_threshold']
    )

    optimizer = getattr(tf.keras.optimizers, config['optimizer'])(learning_rate=config['lr'])

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    return model


def run_experiment(config, train_seq, val_seq, test_seq, epochs=10):
    tf.keras.backend.clear_session()
    model = build_model(config)

    model.fit(train_seq, validation_data=val_seq, epochs=epochs, verbose=0)

    test_loss, test_mae = model.evaluate(test_seq, verbose=0)

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return test_loss, test_mae

def grid_search(train_seq, val_seq, test_seq, param_grid, epochs=10):
    best_score = float('inf')
    best_config = None

    keys, values = zip(*param_grid.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"🔍 총 {len(all_configs)}개 config 탐색 시작")

    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] config: {config}")

        test_loss, test_mae = run_experiment(config, train_seq, val_seq, test_seq, epochs)
        print(f"🧪 test_loss={test_loss:.4f}, test_mae={test_mae:.4f}")

        if test_loss < best_score:
            best_score = test_loss
            best_config = config

    print(f"\n✅ 최적 config: {best_config} | test_loss={best_score:.4f}")
    return best_config

def main():
    train_seq = MultiViewSequence("D:/data/multi_view", "D:/data/multi_view/train_labels", mode='train', batch_size=4)
    val_seq_temp = MultiViewSequence("D:/data/validation", "D:/data/validation/val_labels", mode='val', batch_size=1)
    val_prefixes = val_seq_temp.video_groups
    val_seq = MultiViewSequence("D:/data/validation", "D:/data/validation/val_labels", mode='val', prefixes=val_prefixes)
    test_seq_temp = MultiViewSequence("D:/data/test", "D:/data/test/test_labels", mode='test', batch_size=1)
    test_prefixes = test_seq_temp.video_groups
    test_seq = MultiViewSequence("D:/data/test", "D:/data/test/test_labels", mode='test', prefixes=test_prefixes)

    param_grid = {
        'cnn_filters': [32, 64],
        'cnn_kernel': [(3, 3, 3)],
        'pool_size': [(1, 2, 2), (2, 2, 2)],
        'cnn_depth': [1, 2],
        'mlp_layers': [1, 2],
        'dense_units': [64, 128],
        'dropout': [0.0, 0.2],
        'var_weight': [0.0, 0.1, 0.5],
        'var_threshold': [2.0, 4.0],
        'optimizer': ['Adam'],
        'lr': [1e-3, 1e-4]
    }

    best_config = grid_search(train_seq, val_seq, test_seq, param_grid, epochs=10)

if __name__ == "__main__":
    main()
