"""Saved data loading: .npz files in, features, labels, and sources out (see D26 and D27 in architecture.md)."""

import numpy as np

from asl.features import landmarks_to_features
from asl.labels import LETTER_TO_INDEX


def load_file(path):
    """Load one saved .npz file (D18) and return its features, labels, and sources."""
    # The with block closes the file once the arrays are read into memory
    with np.load(path) as data:
        landmarks = data["landmarks"]
        is_left_hand = data["is_left_hand"]
        letters = data["letters"]
        sources = data["sources"]

    # Every sample goes through the same function the live demo uses (D17)
    features = np.stack(
        [landmarks_to_features(sample, left) for sample, left in zip(landmarks, is_left_hand)]
    )
    # Computed at load time so indices always match the current label set (D18).
    # PyTorch's cross-entropy loss expects class indices as 64-bit integers
    labels = np.array([LETTER_TO_INDEX[letter] for letter in letters], dtype=np.int64)
    return features, labels, sources

def load_files(paths):
    """Load several .npz files and join them into one set of features, labels, and sources."""
    loaded = [load_file(path) for path in paths]
    features = np.concatenate([file_features for file_features, _, _ in loaded])
    labels = np.concatenate([file_labels for _, file_labels, _ in loaded])
    sources = np.concatenate([file_sources for _, _, file_sources in loaded])
    return features, labels, sources