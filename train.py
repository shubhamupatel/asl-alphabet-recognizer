"""Train the letter classifier, with early stopping on validation data (see D23 and D25 in architecture.md).

Trains on the Kaggle data plus my training session, and checks the validation session
after every epoch. The test session is never loaded here. Only evaluate.py reads it.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from asl.data import load_files

# Anchor paths to this script's folder so it runs correctly from anywhere
PROJECT_DIR = Path(__file__).resolve().parent
KAGGLE_PATH = PROJECT_DIR / "data" / "processed" / "kaggle.npz"
RECORDINGS_DIR = PROJECT_DIR / "data" / "recordings"
# Roles are assigned by file (D23). The test session is deliberately missing from both lists
TRAIN_PATHS = [KAGGLE_PATH, RECORDINGS_DIR / "train1.npz"]
VAL_PATHS = [RECORDINGS_DIR / "val1.npz"]
# A fixed seed makes runs repeatable, so comparing D25's two sampling modes changes only the sampling
SEED = 0

# Samples per gradient step (D28). Validation reuses it, which changes speed but not results
BATCH_SIZE = 64


def parse_args():
    """Read the sampling mode from the command line (D25)."""
    parser = argparse.ArgumentParser(description="Train the letter classifier.")
    parser.add_argument(
        "--sampling",
        choices=["balanced", "natural"],
        default="balanced",
        help="balanced draws every source and letter equally often; natural is the unweighted baseline (default: balanced)",
    )
    return parser.parse_args()

def sample_weights(labels, sources):
    """Weight each sample by 1 / the size of its source-and-letter group (D25)."""
    # A group is a source and a letter together, so Kaggle's A and train1's A are counted separately
    groups = list(zip(sources.tolist(), labels.tolist()))
    group_sizes = Counter(groups)
    return [1 / group_sizes[group] for group in groups]


def make_train_loader(features, labels, sources, sampling):
    """Build the training DataLoader, balanced by weighted sampling or in the natural mix (D25)."""
    # from_numpy shares memory with the arrays instead of copying, and keeps their float32 and int64 types
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    if sampling == "balanced":
        # Drawing with replacement lets samples from small groups come up repeatedly.
        # One epoch is as many draws as there are training samples (D29)
        sampler = WeightedRandomSampler(sample_weights(labels, sources), num_samples=len(labels), replacement=True)
        return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def main():
    args = parse_args()
    # PyTorch's random number generator drives initial weights, dropout, and batch order
    torch.manual_seed(SEED)

    train_features, train_labels, train_sources = load_files(TRAIN_PATHS)
    # Validation sources aren't needed, since validation is never reweighted (D25)
    val_features, val_labels, _ = load_files(VAL_PATHS)
    print(f"Training on {len(train_labels)} samples, validating on {len(val_labels)}")

    train_loader = make_train_loader(train_features, train_labels, train_sources, args.sampling)
    # Validation is only measured, so it's never shuffled or reweighted
    val_dataset = TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()