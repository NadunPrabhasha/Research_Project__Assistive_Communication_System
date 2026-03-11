import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

CLASSES = ["food", "happy", "help", "toilet", "water"]
IMG_SIZE = 224

MODEL_DIR = "models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "gesture_cnn_weights.weights.h5")

model = None
model_status = "Model missing"


def build_model():

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(CLASSES), activation="softmax")
    ])

    return model


def load_prediction_model():
    global model, model_status

    try:
        if not os.path.exists(WEIGHTS_PATH):
            model_status = f"Weights not found: {WEIGHTS_PATH}"
            print(model_status)
            return

        model = build_model()
        model.load_weights(WEIGHTS_PATH)

        model_status = "Gesture model loaded successfully"
        print(model_status)

    except Exception as e:
        model = None
        model_status = f"Model load error: {e}"
        print(model_status)


load_prediction_model()


def preprocess_frame(frame):

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    return img


def predict_frame(frame):

    if model is None:
        return "Model missing", 0.0

    try:
        processed = preprocess_frame(frame)

        predictions = model.predict(processed, verbose=0)[0]

        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])

        label = CLASSES[class_index]

        return label, confidence

    except Exception as e:
        print("Prediction error:", e)
        return "Prediction error", 0.0