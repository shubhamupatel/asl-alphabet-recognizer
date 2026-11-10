"""Recognize ASL letters live from the webcam (see D39, D40, and D41 in architecture.md).

Shows the predicted letter and its confidence, averaged over recent frames.
Press Esc to quit.
"""

import argparse
from collections import deque

import cv2
import numpy as np
import torch

from asl.features import landmarks_to_features
from asl.labels import LETTERS
from asl.landmarks import HandLandmarkExtractor
from asl.model import DEFAULT_CHECKPOINT_PATH, load_checkpoint

# Average the probabilities over this many recent frames (D39)
SMOOTHING_FRAMES = 10
# Below this averaged confidence, show "No letter detected" instead of a guess (D40)
CONFIDENCE_THRESHOLD = 0.7
# Esc quits, matching collect_data.py
ESC_KEY = 27


def parse_args():
    """Read the signing hand and the model file from the command line (D41)."""
    parser = argparse.ArgumentParser(description="Recognize ASL letters live from the webcam.")
    parser.add_argument("--hand", choices=["right", "left"], required=True, help="the hand you're signing with")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHECKPOINT_PATH,
        help="the saved model to run (default: models/letter_classifier.pt)",
    )
    return parser.parse_args()


def letter_probabilities(model, landmarks, is_left_hand):
    """Return the model's 24 letter probabilities for one frame's landmarks."""
    features = landmarks_to_features(landmarks, is_left_hand)
    # The model expects a batch, so add a batch dimension: (63,) becomes (1, 63)
    batch = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
    # Softmax is applied only here, where a confidence is needed (D24)
    return torch.softmax(logits, dim=1)[0].numpy()

def main():
    args = parse_args()
    # The declared hand decides mirroring for every frame (D19)
    is_left_hand = args.hand == "left"
    model, _ = load_checkpoint(args.model)
    # The most recent frames' probabilities. Once full, adding one drops the oldest (D39)
    recent = deque(maxlen=SMOOTHING_FRAMES)

    capture = cv2.VideoCapture(0)
    try:
        with HandLandmarkExtractor() as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                detection = extractor.extract(frame)
                if detection is None:
                    # Start fresh when the hand returns, so an old letter can't carry over (D39)
                    recent.clear()
                    status = "No hand detected"
                else:
                    # MediaPipe's per-frame handedness guess is ignored in favor of the declared hand (D19)
                    landmarks, _ = detection
                    recent.append(letter_probabilities(model, landmarks, is_left_hand))
                    for x, y, _ in landmarks:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                    # Only a full window counts as an average over the last 10 frames (D39)
                    if len(recent) < SMOOTHING_FRAMES:
                        status = "No letter detected"
                    else:
                        average = np.mean(recent, axis=0)
                        best = int(average.argmax())
                        if average[best] >= CONFIDENCE_THRESHOLD:
                            status = f"{LETTERS[best]}  {average[best]:.0%}"
                        else:
                            status = "No letter detected"

                # Landmarks were drawn on the original frame, so they stay on the hand after mirroring.
                # Text is drawn after mirroring so it reads normally (D22)
                display = cv2.flip(frame, 1)
                cv2.putText(display, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                hand_label = f"{args.hand.capitalize()} hand"
                cv2.putText(display, hand_label, (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("ASL letters (Esc to quit)", display)
                if cv2.waitKey(1) & 0xFF == ESC_KEY:
                    break
    finally:
        # Runs however the loop ends: Esc, a camera failure, Ctrl+C, or a crash
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()