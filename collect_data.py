"""Record labeled hand landmark samples from the webcam (see D20 and D21 in architecture.md).

Press a letter key to record a burst of that letter. Press Esc to save the session and quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from asl.labels import LETTERS
from asl.landmarks import HandLandmarkExtractor

# Anchor paths to this script's folder so it runs correctly from anywhere
PROJECT_DIR = Path(__file__).resolve().parent
# Recordings are original data, kept apart from regenerable files (D21)
RECORDINGS_DIR = PROJECT_DIR / "data" / "recordings"
# Time to form the sign before recording starts, so transition frames aren't saved (D20)
COUNTDOWN_SECONDS = 2.0
# Esc quits, because "q" is a letter we need to record
ESC_KEY = 27

# The three recording states (D20)
IDLE = "idle"
COUNTDOWN = "countdown"
RECORDING = "recording"


def key_to_letter(key):
    """Return the letter for a pressed key, or None if it isn't a letter we record."""
    letter = chr(key).upper()
    return letter if letter in LETTERS else None

def parse_args():
    """Read the session name, signing hand, and burst size from the command line."""
    parser = argparse.ArgumentParser(description="Record labeled hand landmark samples from the webcam.")
    parser.add_argument("session", help="name for this recording session, such as session1")
    parser.add_argument("--hand", choices=["right", "left"], required=True, help="the hand you're signing with")
    parser.add_argument("--samples", type=int, default=200, help="samples to record per burst (default: 200)")
    return parser.parse_args()

def save_session(output_path, all_landmarks, letters, is_left_hand, session):
    """Save the completed bursts to this session's file (D18, D21). Saves nothing if no burst finished."""
    if not letters:
        print("No completed bursts, so nothing was saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks=np.stack(all_landmarks),
        is_left_hand=np.full(len(letters), is_left_hand),
        letters=np.array(letters),
        sources=np.full(len(letters), session),
    )
    print(f"Saved {len(letters)} samples to {output_path}")

def main():
    args = parse_args()

    # Recordings can't be regenerated, so never overwrite an existing session
    output_path = RECORDINGS_DIR / f"{args.session}.npz"
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists. Choose a different session name.")

    # The declared hand applies to every sample in this session (D19)
    is_left_hand = args.hand == "left"

    # Session-level lists, which only grow when a burst finishes
    all_landmarks = []
    letters = []

    # Recording state (D20): idle → countdown → recording → idle
    state = IDLE
    current_letter = None
    countdown_start = 0.0
    burst = []

    capture = cv2.VideoCapture(0)
    try:
        with HandLandmarkExtractor() as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # Extract on every frame, even when idle, so VIDEO mode keeps tracking the hand
                detection = extractor.extract(frame)

                # Draw landmarks on the original frame, then mirror it for display only (D22).
                # Landmarks always come from the unmirrored frame, so the data is unaffected
                if detection is not None:
                    landmarks, _ = detection
                    for x, y, _ in landmarks:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                display = cv2.flip(frame, 1)

                # Status text is drawn after mirroring so it reads normally
                if state == IDLE:
                    status = f"Press a letter to record ({len(letters)} samples this session)"
                elif state == COUNTDOWN:
                    remaining = COUNTDOWN_SECONDS - (time.monotonic() - countdown_start)
                    status = f"Get ready: {current_letter} in {remaining:.1f}s"
                else:
                    status = f"Recording {current_letter}: {len(burst)}/{args.samples}"
                if detection is None:
                    status += "  (no hand)"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Recording (Esc to save and quit)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ESC_KEY:
                    break

                if state == IDLE:
                    # A letter key starts a countdown for that letter. Other keys are ignored
                    letter = key_to_letter(key)
                    if letter is not None:
                        current_letter = letter
                        countdown_start = time.monotonic()
                        state = COUNTDOWN
                        print(f"Get ready: {current_letter}")

                elif state == COUNTDOWN:
                    # Start recording once the countdown has run out
                    if time.monotonic() - countdown_start >= COUNTDOWN_SECONDS:
                        burst = []
                        state = RECORDING
                        print(f"Recording {current_letter}...")

                elif state == RECORDING:
                    # Only frames with a detected hand count toward the burst (D20).
                    # MediaPipe's handedness guess is ignored in favor of the declared hand (D19)
                    if detection is not None:
                        landmarks, _ = detection
                        burst.append(landmarks)
                    # A finished burst joins the session, then we go back to waiting for a key
                    if len(burst) >= args.samples:
                        all_landmarks.extend(burst)
                        letters.extend([current_letter] * len(burst))
                        state = IDLE
                        print(f"Finished {current_letter}: {len(letters)} samples so far this session")
    finally:
        # Runs however the loop ends: Esc, a camera failure, Ctrl+C, or a crash
        capture.release()
        cv2.destroyAllWindows()
        save_session(output_path, all_landmarks, letters, is_left_hand, args.session)


if __name__ == "__main__":
    main()