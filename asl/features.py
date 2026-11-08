"""Feature processing: raw hand landmarks in, normalized feature vector out."""

import numpy as np

WRIST = 0
MIDDLE_FINGER_MCP = 9


def normalize_landmarks(landmarks):
    """Remove hand position and size from a (21, 3) landmark array (see D9 in architecture.md).

    Positions become relative to the wrist and are divided by palm length, so the same
    hand shape gives the same numbers anywhere in the frame and at any distance.
    """
    centered = landmarks - landmarks[WRIST]
    palm_length = np.linalg.norm(centered[MIDDLE_FINGER_MCP])
    return centered / palm_length