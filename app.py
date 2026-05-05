from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import pyttsx3
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
import os
import logging
import atexit
import threading
from collections import deque
import predictor
import numpy as np

from gtts import gTTS
import pygame
import tempfile

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# -------------------------
# Camera setup - Opens webcam using OpenCV
# -------------------------

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


# -------------------------
# UPDATED TEXT-TO-SPEECH SETUP
# -------------------------

"""
DELETED OLD SPEECH PART:

engine = pyttsx3.init()
engine.setProperty("rate", 150)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.warning(f"Speech error: {e}")
        finally:
            speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

Reason for removing:
The shared pyttsx3 engine and queue may work only one time in Flask.
The updated version below creates a fresh pyttsx3 engine inside a safe speech thread.
"""

speech_lock = threading.Lock()
last_spoken_text = ""
last_spoken_time = 0
SPEAK_INTERVAL = 2.5


def speak_text(text):
    global last_spoken_text, last_spoken_time

    if not text:
        return

    current_time = time.time()

    if text == last_spoken_text and current_time - last_spoken_time < SPEAK_INTERVAL:
        return

    last_spoken_text = text
    last_spoken_time = current_time

    language = selected_language

    speech_thread = threading.Thread(target=_speak_worker, args=(text, language))
    speech_thread.daemon = True
    speech_thread.start()


def _speak_worker(text, language):
    """
    Multilingual TTS worker.
    English uses offline pyttsx3.
    Sinhala and Tamil use online gTTS.
    """
    with speech_lock:
        try:
            if language == "en":
                speak_with_pyttsx3(text)
            elif language in ["si", "ta"]:
                speak_with_gtts(text, language)
            else:
                speak_with_pyttsx3(text)

        except Exception as e:
            logging.warning(f"TTS Error: {e}")

def speak_with_pyttsx3(text):
    """
    Offline English voice using pyttsx3.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def speak_with_gtts(text, language):
    """
    Online Sinhala/Tamil voice using gTTS.
    Requires internet connection.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name

    tts = gTTS(text=text, lang=language)
    tts.save(temp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    try:
        os.remove(temp_path)
    except Exception:
        pass


# -------------------------
# Prediction state
# -------------------------

latest_prediction = {
    "label": "Waiting...",
    "confidence": 0.0,
    "sentence": "Waiting for gesture...",
    "model_status": predictor.model_status,
    "detector_status": "Checking detector..."
}

hand_detector_status = "Unknown"
history = deque(maxlen=10)

last_spoken = ""
last_hand_seen_time = 0
current_stable_label = ""
ready_for_next_speech = True

CONFIDENCE_THRESHOLD = 0.75
MIN_STABLE_FRAMES = 6
NO_HAND_RESET_TIME = 1.2


# Current selected language
# en = English, si = Sinhala, ta = Tamil
selected_language = "en"


MODEL_PATH = os.path.join("models", "hand_landmarker.task")

hand_landmarker = None
legacy_hands = None
legacy_mp_draw = None
legacy_mp_hands_lib = None


# -------------------------
# Cleanup
# -------------------------

def cleanup():
    global camera, legacy_hands, hand_landmarker

    try:
        if camera is not None and camera.isOpened():
            camera.release()
    except Exception:
        pass

    try:
        if legacy_hands is not None:
            legacy_hands.close()
    except Exception:
        pass

    try:
        if hand_landmarker is not None:
            hand_landmarker.close()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


atexit.register(cleanup)


# -------------------------
# MediaPipe hand detector setup
# -------------------------

try:
    if os.path.isfile(MODEL_PATH):
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        hand_options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        hand_landmarker = HandLandmarker.create_from_options(hand_options)
        logging.info(f"Loaded HandLandmarker from {MODEL_PATH}")
        hand_detector_status = f"Tasks HandLandmarker loaded from {MODEL_PATH}"
    else:
        raise FileNotFoundError(f"Hand landmarker model not found: {MODEL_PATH}")

except Exception as e:
    logging.warning(f"Tasks HandLandmarker failed: {e}")

    if hasattr(mp, "solutions"):
        try:
            legacy_mp_hands_lib = mp.solutions.hands
            legacy_hands = legacy_mp_hands_lib.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            legacy_mp_draw = mp.solutions.drawing_utils
            logging.info("Using legacy mp.solutions.hands fallback")
            hand_detector_status = "Legacy mp.solutions.hands loaded"
        except Exception as legacy_error:
            logging.warning(f"Legacy MediaPipe also failed: {legacy_error}")
            hand_detector_status = f"Legacy MediaPipe failed: {legacy_error}"


# -------------------------
# Gesture sentence mapping
# -------------------------

LANGUAGE_SENTENCES = {
    "en": {
        "happy": "I am happy",
        "help": "I need help",
        "food": "I need food",
        "water": "I need water",
        "toilet": "I need to go to the toilet"
    },
    "si": {
        "happy": "මට සතුටුයි",
        "help": "මට උදව් අවශ්‍යයි",
        "food": "මට කෑම අවශ්‍යයි",
        "water": "මට වතුර අවශ්‍යයි",
        "toilet": "මට වැසිකිළියට යන්න අවශ්‍යයි"
    },
    "ta": {
        "happy": "நான் மகிழ்ச்சியாக இருக்கிறேன்",
        "help": "எனக்கு உதவி வேண்டும்",
        "food": "எனக்கு உணவு வேண்டும்",
        "water": "எனக்கு தண்ணீர் வேண்டும்",
        "toilet": "நான் கழிப்பறைக்கு செல்ல வேண்டும்"
    }
}


def gesture_to_sentence(label):
    language_map = LANGUAGE_SENTENCES.get(selected_language, LANGUAGE_SENTENCES["en"])
    return language_map.get(label, "Waiting for gesture...")


# -------------------------
# Gesture stability controls
# -------------------------

def reset_interaction_state():
    global history, last_spoken, current_stable_label, ready_for_next_speech
    history.clear()
    last_spoken = ""
    current_stable_label = ""
    ready_for_next_speech = True


def reset_if_no_hand_long_enough():
    global last_hand_seen_time
    if time.time() - last_hand_seen_time > NO_HAND_RESET_TIME:
        reset_interaction_state()
    else:
        history.clear()


def draw_bbox_and_predict(frame, x_list, y_list, w, h):
    global history

    x_min = max(min(x_list) - 20, 0)
    y_min = max(min(y_list) - 20, 0)
    x_max = min(max(x_list) + 20, w)
    y_max = min(max(y_list) + 20, h)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
##
    hand_crop = frame[y_min:y_max, x_min:x_max]

    if hand_crop.size == 0:
        return "Detecting...", 0.0, None, None
##
    label, confidence = predictor.predict_frame(hand_crop)

    if confidence >= CONFIDENCE_THRESHOLD and label not in ["Model missing", "Prediction error"]:
        history.append(label)
    else:
        history.clear()
        return "Detecting...", confidence, x_min, y_min

    if len(history) >= MIN_STABLE_FRAMES:
        stable_label = max(set(history), key=history.count)
        return stable_label, confidence, x_min, y_min

    return "Detecting...", confidence, x_min, y_min


def process_stable_speech(display_label):
    global current_stable_label, ready_for_next_speech, last_spoken

    if display_label == "Detecting...":
        return  

    sentence = gesture_to_sentence(display_label)

    if display_label != current_stable_label:
        current_stable_label = display_label
        ready_for_next_speech = True

    if ready_for_next_speech:
        speak_text(sentence)
        last_spoken = display_label
        ready_for_next_speech = False


# -------------------------
# Video frame generation
# -------------------------

def generate_frames():
    global latest_prediction
    global last_hand_seen_time

    while True:
        success, frame = camera.read()
        if not success:
            logging.warning("Camera read failed, retrying...")
            time.sleep(0.1)
            continue

        try:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            display_label = "Detecting..."
            confidence = 0.0

            if hand_landmarker is not None:
                try:
                    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                    results = hand_landmarker.detect(mp_image)

                    if results.hand_landmarks:
                        last_hand_seen_time = time.time()

                        for hand_landmarks in results.hand_landmarks:
                            for lm in hand_landmarks:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                            x_list = [int(lm.x * w) for lm in hand_landmarks]
                            y_list = [int(lm.y * h) for lm in hand_landmarks]

                            display_label, confidence, x_min, y_min = draw_bbox_and_predict(
                                frame, x_list, y_list, w, h
                            )

                            if x_min is not None and y_min is not None and display_label != "Detecting...":
                                cv2.putText(
                                    frame,
                                    display_label,
                                    (x_min, max(y_min - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )

                            process_stable_speech(display_label)
                    else:
                        reset_if_no_hand_long_enough()

                except Exception as e:
                    logging.exception(f"HandLandmarker detection error: {e}")
                    reset_if_no_hand_long_enough()

            elif legacy_hands is not None:
                try:
                    res = legacy_hands.process(rgb)

                    if res.multi_hand_landmarks:
                        last_hand_seen_time = time.time()

                        for hand_landmarks in res.multi_hand_landmarks:
                            legacy_mp_draw.draw_landmarks(
                                frame,
                                hand_landmarks,
                                legacy_mp_hands_lib.HAND_CONNECTIONS
                            )

                            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                            display_label, confidence, x_min, y_min = draw_bbox_and_predict(
                                frame, x_list, y_list, w, h
                            )

                            if x_min is not None and y_min is not None and display_label != "Detecting...":
                                cv2.putText(
                                    frame,
                                    display_label,
                                    (x_min, max(y_min - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )

                            process_stable_speech(display_label)
                    else:
                        reset_if_no_hand_long_enough()

                except Exception as e:
                    logging.exception(f"Legacy MediaPipe detection error: {e}")
                    reset_if_no_hand_long_enough()

            else:
                reset_if_no_hand_long_enough()
                display_label = "Hand detector missing"
                confidence = 0.0
                cv2.putText(
                    frame,
                    "Hand detector not loaded - see console.",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            if hand_landmarker is not None:
                detector_status = hand_detector_status
            elif legacy_hands is not None:
                detector_status = hand_detector_status
            else:
                detector_status = "Hand detector not loaded"

            latest_prediction = {
                "label": display_label,
                "confidence": round(confidence * 100, 2),
                "sentence": gesture_to_sentence(display_label) if display_label != "Detecting..." else "Waiting for gesture...",
                "model_status": predictor.model_status,
                "detector_status": detector_status
            }

            text = f"{display_label} ({confidence * 100:.2f}%)"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                logging.warning("Failed to encode frame, sending blank placeholder")
                raise RuntimeError("Failed to encode frame")

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        except Exception:
            logging.exception("Unhandled error while processing frame; yielding placeholder image")
            try:
                ph_h, ph_w = (480, 640)
                placeholder = np.full((ph_h, ph_w, 3), 240, dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Stream error - see server logs",
                    (10, ph_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                ret, buffer = cv2.imencode(".jpg", placeholder)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
            except Exception:
                logging.exception("Failed to create placeholder frame")

            time.sleep(0.2)
            continue


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/gesture")
def gesture_page():
    return render_template("index.html")


@app.route("/symbols")
def symbol_page():
    return render_template("symbols.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_prediction")
def get_prediction():
    return jsonify(latest_prediction)


@app.route("/set_language", methods=["POST"])
def set_language():
    global selected_language

    data = request.get_json(silent=True) or {}
    language = data.get("language", "en")

    if language not in ["en", "si", "ta"]:
        return jsonify({
            "status": "error",
            "message": "Unsupported language"
        }), 400

    selected_language = language

    return jsonify({
        "status": "success",
        "language": selected_language
    })



@app.route("/speak_symbol", methods=["POST"])
def speak_symbol():
    data = request.get_json(silent=True) or {}

    # New method: receive symbol label, for example "food"
    label = data.get("label", "").strip().lower()

    # Convert label to selected language sentence
    if label:
        sentence = gesture_to_sentence(label)
    else:
        # Backup method: if old frontend sends direct text
        sentence = data.get("text", "").strip() or data.get("sentence", "").strip()

    if not sentence or sentence == "Waiting for gesture...":
        return jsonify({
            "status": "error",
            "message": "No valid symbol provided"
        }), 400

    speak_text(sentence)

    return jsonify({
        "status": "success",
        "label": label,
        "language": selected_language,
        "sentence": sentence
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)