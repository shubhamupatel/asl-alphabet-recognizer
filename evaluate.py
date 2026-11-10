"""Evaluate a trained letter classifier on one recording session (see D23, D34, and D35 in architecture.md).

The test session is evaluated once, after every decision is final (D23).
To check this script itself, run it on val1, since validation data can be used freely.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from asl.data import load_files
from asl.labels import LETTERS, NUM_CLASSES
from asl.model import DEFAULT_CHECKPOINT_PATH, load_checkpoint, predict

# Anchor paths to this script's folder so it runs correctly from anywhere
PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "data" / "recordings"
# Nothing learns here, so batch size affects only speed (D28)
BATCH_SIZE = 64
# How many of the most common confusions the report lists (D35)
NUM_CONFUSIONS_SHOWN = 10


def parse_args():
    """Read the session to evaluate and the model file from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate the letter classifier on one recording session.")
    parser.add_argument("session", help="recording session to evaluate, such as val1 or test1")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHECKPOINT_PATH,
        help="the saved model to evaluate (default: models/letter_classifier.pt)",
    )
    parser.add_argument(
        "--heatmap",
        help="also save the confusion matrix as an image at this path, such as docs/confusion_matrix.png (D43)",
    )
    return parser.parse_args()

def confusion_matrix(true_labels, predictions):
    """Count how often each true letter (row) was predicted as each letter (column)."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true, predicted in zip(true_labels.tolist(), predictions.tolist()):
        matrix[true, predicted] += 1
    return matrix


def print_report(matrix):
    """Print accuracy per letter, worst first, then the most common confusions (D35)."""
    # The diagonal holds each letter's correct predictions, and each row's sum is that letter's sample count
    correct_per_letter = np.diag(matrix)
    samples_per_letter = matrix.sum(axis=1)
    accuracy_per_letter = correct_per_letter / samples_per_letter

    print("\nAccuracy per letter, worst first:")
    # A stable sort keeps tied letters in alphabetical order
    for index in np.argsort(accuracy_per_letter, kind="stable"):
        print(
            f"  {LETTERS[index]}  {accuracy_per_letter[index]:6.1%}  "
            f"({correct_per_letter[index]} of {samples_per_letter[index]})"
        )

    # Every nonzero cell off the diagonal is one specific mistake: a true letter predicted as another
    confusions = [
        (matrix[true, predicted], LETTERS[true], LETTERS[predicted])
        for true in range(NUM_CLASSES)
        for predicted in range(NUM_CLASSES)
        if true != predicted and matrix[true, predicted] > 0
    ]
    confusions.sort(key=lambda confusion: confusion[0], reverse=True)

    print(f"\nMost common confusions (true -> predicted), top {NUM_CONFUSIONS_SHOWN}:")
    if not confusions:
        print("  None")
    for count, true_letter, predicted_letter in confusions[:NUM_CONFUSIONS_SHOWN]:
        print(f"  {true_letter} -> {predicted_letter}: {count}")

def save_heatmap(matrix, path, title):
    """Save the confusion matrix as an image, shaded by row percentage, with counts in nonzero cells (D43)."""
    # Each row divided by its total: the share of that true letter's samples landing in each cell
    row_shares = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(row_shares, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Share of the true letter's samples")

    ax.set_xticks(range(NUM_CLASSES), labels=LETTERS)
    ax.set_yticks(range(NUM_CLASSES), labels=LETTERS)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(title)

    for true in range(NUM_CLASSES):
        for predicted in range(NUM_CLASSES):
            count = matrix[true, predicted]
            if count > 0:
                # White text on dark cells and black on light ones, so every count stays readable
                color = "white" if row_shares[true, predicted] > 0.5 else "black"
                ax.text(predicted, true, str(count), ha="center", va="center", fontsize=7, color=color)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    # Free the figure's memory, since nothing displays it
    plt.close(fig)

def main():
    args = parse_args()

    model, training_info = load_checkpoint(args.model)
    print(
        f"Model: best epoch {training_info['best_epoch']} of a {training_info['sampling']} run, "
        f"validation accuracy {training_info['val_accuracy']:.1%}"
    )

    features, labels, _ = load_files([RECORDINGS_DIR / f"{args.session}.npz"])
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    logits, true_labels = predict(model, DataLoader(dataset, batch_size=BATCH_SIZE))
    # The highest of the 24 scores is each sample's predicted letter
    predictions = logits.argmax(dim=1)

    correct = (predictions == true_labels).sum().item()
    total = len(true_labels)
    print(f"Session {args.session}: {correct} of {total} correct ({correct / total:.1%})")

    print_report(confusion_matrix(true_labels, predictions))

    matrix = confusion_matrix(true_labels, predictions)
    print_report(matrix)
    if args.heatmap:
        title = f"Confusion matrix on {args.session}: {correct} of {total} correct ({correct / total:.1%})"
        save_heatmap(matrix, args.heatmap, title)
        print(f"\nSaved the heatmap to {args.heatmap}")

if __name__ == "__main__":
    main()