import tensorflow as tf

paths = [
    "models/gesture_cnn_model.keras",
    "models/gesture_cnn_model.h5",
]

for path in paths:
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print(f"SUCCESS: {path}")
        model.summary()
    except Exception as e:
        print(f"FAILED: {path}")
        print(e)
        print("-" * 50)