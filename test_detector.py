import os
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join("models", "hand_landmarker.task")

try:
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    print("HandLandmarker loaded successfully")
except Exception as e:
    print(f"Failed to load HandLandmarker: {e}")

# Test legacy
try:
    legacy_mp_hands_lib = mp.solutions.hands
    legacy_hands = legacy_mp_hands_lib.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    print("Legacy MediaPipe loaded successfully")
except Exception as e:
    print(f"Failed to load Legacy MediaPipe: {e}")