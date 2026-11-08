"""Feature processing: raw hand landmarks in, normalized feature vector out."""

import numpy as np

WRIST = 0
MIDDLE_FINGER_MCP = 9
X_AXIS = 0


def normalize_landmarks(landmarks):
    """Remove hand position and size from a (21, 3) landmark array (see D9 in architecture.md).

    Positions become relative to the wrist and are divided by palm length, so the same
    hand shape gives the same numbers anywhere in the frame and at any distance.
    Rotation is deliberately left alone, because hand orientation distinguishes some
    letters, such as H from U (see D10).
    """
    centered = landmarks - landmarks[WRIST]
    palm_length = np.linalg.norm(centered[MIDDLE_FINGER_MCP])
    return centered / palm_length

def mirror_to_right_hand(normalized_landmarks):
    """Mirror a normalized left hand so it matches right-hand form (see D11 in architecture.md)."""
    mirrored = normalized_landmarks.copy()
    mirrored[:, X_AXIS] = -mirrored[:, X_AXIS]
    return mirrored