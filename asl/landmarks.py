"""Hand landmark extraction: webcam frame in, 21 (x, y, z) hand landmarks out."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

NUM_LANDMARKS = 21

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


class HandLandmarkExtractor:
    """Wraps MediaPipe's HandLandmarker in VIDEO mode (see D7 in architecture.md)."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        # Verify model exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmark model not found at {model_path}.")

        # Configuration
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )

        # Load the model
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Create a timestamp
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def extract(self, frame_bgr):
        """Return the hand's 21 landmarks as a (21, 3) float32 array, or None if no hand is found."""
        # Turn OpenCV's BGR format to MediaPipe's RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe image so it can process it
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Generate a timestamp in milliseconds
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        # Make sure two timestamps don't land in the same millisecond
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        # Call the landmarker on the frame to run hand detection and landmark extraction
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # If no hand was found, return None
        if not result.hand_landmarks:
            return None

        # If a hand was found, pull out the 21 landmark points and extract the (x,y,z) coordinates for each point
        hand = result.hand_landmarks[0]
        # Convert them into a NumPy array of shape (21,3)
        return np.array([[point.x, point.y, point.z] for point in hand], dtype=np.float32)

    def close(self):
        """Release MediaPipe's resources."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    with HandLandmarkExtractor() as extractor:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            landmarks = extractor.extract(frame)
            if landmarks is not None:
                height, width = frame.shape[:2]
                for x, y, _ in landmarks:
                    cv2.circle(frame, (int(x * width), int(y * height)), 4, (0, 255, 0), -1)
                status = f"Hand found: {landmarks.shape}"
            else:
                status = "No hand"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Landmark test (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()