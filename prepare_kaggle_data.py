"""Convert the Kaggle ASL Alphabet photos into landmark data (see D12 and D16–D18 in architecture.md).

Runs every photo in the dataset's letter folders through MediaPipe in IMAGE mode
and saves the detected hands to data/processed/kaggle.npz.
"""

from pathlib import Path

import cv2
import numpy as np

from asl.labels import LETTERS
from asl.landmarks import HandLandmarkExtractor

# Anchor paths to this script's folder so it runs correctly from anywhere
PROJECT_DIR = Path(__file__).resolve().parent
# The folder that directly contains the letter folders (A, B, C, ...)
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "kaggle_asl_alphabet" / "asl_alphabet_train" / "asl_alphabet_train"
# Where the extracted landmark data gets saved
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "kaggle.npz"
# Stored with every sample so it keeps its origin after merging with other sources (D17)
SOURCE = "kaggle"
# The Kaggle signer is treated as right-handed (D19), so no sample from this source is mirrored
SIGNER_IS_LEFT_HANDED = False


def main():
    # One list per saved array, filled one photo at a time
    # (appending to lists is fast; growing a NumPy array would copy it every time)
    all_landmarks = []
    left_hand_flags = []
    letters = []
    sources = []

    # Create one IMAGE-mode extractor (D16) and reuse it for every photo,
    # since loading the model is the expensive part
    with HandLandmarkExtractor(video_mode=False) as extractor:
        # Loop over our own letter list, which automatically skips J, Z, space, del, and nothing
        for letter in LETTERS:
            letter_dir = DATASET_DIR / letter
            # A missing letter folder almost certainly means DATASET_DIR is wrong, so stop loudly
            if not letter_dir.is_dir():
                raise FileNotFoundError(f"No folder for letter {letter} at {letter_dir}")

            # Find this letter's photos in a fixed order, skipping any non-.jpg files
            image_paths = sorted(letter_dir.glob("*.jpg"))
            found = 0
            for image_path in image_paths:
                # Load the photo as a BGR array. imread returns None instead of raising an error
                # when a file can't be read, so check for that explicitly
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                # Extract the raw (21, 3) pixel landmarks
                detection = extractor.extract(image)
                # Skip photos where MediaPipe couldn't find a hand
                if detection is None:
                    continue

                # Record this sample's four pieces of information
                # Keep the landmarks. MediaPipe's per-frame handedness guess is ignored (D19)
                landmarks, _ = detection
                all_landmarks.append(landmarks)
                left_hand_flags.append(SIGNER_IS_LEFT_HANDED)
                letters.append(letter)
                sources.append(SOURCE)
                found += 1

            # Report this letter's detection rate, to spot letters that came out short
            print(f"{letter}: hand found in {found} of {len(image_paths)} photos")

    # Create data/processed/ if it doesn't exist yet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Convert the lists to arrays and save them as named arrays in one compressed file (D18)
    # np.stack combines the N separate (21, 3) arrays into one (N, 21, 3) array
    np.savez_compressed(
        OUTPUT_PATH,
        landmarks=np.stack(all_landmarks),
        is_left_hand=np.array(left_hand_flags),
        letters=np.array(letters),
        sources=np.array(sources),
    )
    print(f"Saved {len(letters)} samples to {OUTPUT_PATH}")


# Run main() only when this file is executed directly, not when it's imported
if __name__ == "__main__":
    main()