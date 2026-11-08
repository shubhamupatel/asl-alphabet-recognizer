"""Feature processing: raw hand landmarks in, normalized feature vector out."""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_FINGER_MCP = 9
X_AXIS = 0

# Each landmark contributes x, y, and z
NUM_FEATURES = NUM_LANDMARKS * 3


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

def landmarks_to_features(landmarks, is_left_hand):
    """Turn raw (21, 3) landmarks into the 63-number feature vector the classifier uses.

    Normalizes position and size (D9), mirrors left hands to right-hand form (D11),
    and flattens the result. Keeps z (D14) and the wrist's zeros (D15).
    """
    normalized = normalize_landmarks(landmarks)
    if is_left_hand:
        normalized = mirror_to_right_hand(normalized)
    return normalized.flatten()