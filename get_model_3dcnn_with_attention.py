import tensorflow as tf

def MaskedAttentionBlock(x):
    attention = tf.keras.layers.Conv3D(
        filters=1, kernel_size=(1, 1, 1), activation='sigmoid'
    )(x)
    attended = tf.keras.layers.Multiply()([x, attention])
    return attended

def get_model_3dcnn_with_attention(n_frames=5, img_height=66, img_width=200, channels=3):
    input_shape = (n_frames, img_height, img_width, channels)

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(32, (3, 5, 5), activation='relu', strides=(1, 2, 2), padding='same')(inputs)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)

    x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', strides=(1, 2, 2), padding='same')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', strides=(1, 2, 2), padding='same')(x)

    # ✅ Masked Attention
    x = MaskedAttentionBlock(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_frames - 1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse')

    return model
