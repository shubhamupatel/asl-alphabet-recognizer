"""Train the letter classifier, with early stopping on validation data (see D23 and D25 in architecture.md).

Trains on the Kaggle data plus my training session, and checks the validation session
after every epoch. The test session is never loaded here. Only evaluate.py reads it.
"""

import argparse
import copy
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from asl.data import load_files
from asl.labels import LETTERS
from asl.model import LetterClassifier

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
# Adam's standard default (D30). Tunable on validation data, never test data
LEARNING_RATE = 0.001
# Stop after this many epochs in a row without a lower validation loss (D31)
PATIENCE = 5
# Safety net in case tiny improvements keep resetting the patience counter (D31)
MAX_EPOCHS = 100
# Every run overwrites this file (D32)
MODEL_PATH = PROJECT_DIR / "models" / "letter_classifier.pt"
# The model's constructor settings (D24), saved with the weights so the model can be rebuilt exactly (D32)
MODEL_SETTINGS = {"first_hidden": 128, "second_hidden": 64, "dropout": 0.2}


def parse_args():
    """Read the sampling mode from the command line (D25)."""
    parser = argparse.ArgumentParser(description="Train the letter classifier.")
    parser.add_argument(
        "--sampling",
        choices=["balanced", "natural"],
        default="natural",
        help="balanced draws every source and letter equally often; natural is the unweighted baseline (default: natural)",
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

def train_one_epoch(model, loader, loss_fn, optimizer):
    """Run one epoch of the training phase and return the average loss per sample.
    
    For each batch:

    1. Unpack the (features, labels) pair
    2. optimizer.zero_grad(), to clear the previous batch’s gradients
    3. logits = model(features), the forward pass
    4. loss = loss_fn(logits, labels)
    5. loss.backward(), which computes every weight’s gradient
    6. optimizer.step(), which lets Adam move each weight
    7. Add this batch’s loss to a running total
    """
    # Turns dropout on (D24)
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, labels in loader:
        # PyTorch adds new gradients onto old ones, so the previous batch's must be cleared first
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        # Compute every weight's gradient, then let Adam move each weight (D30)
        loss.backward()
        optimizer.step()

        # loss is this batch's average, so weight it by the batch's size, since the last batch is smaller
        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples

def validate(model, loader, loss_fn):
    """Measure the model on validation data and return its average loss per sample and its accuracy."""
    # Turns dropout off, so every neuron contributes and the same input always gives the same output
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # Nothing learns here, so skip recording the operations backpropagation would need
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = loss_fn(logits, labels)
            # The highest of the 24 scores is the predicted letter.
            # Softmax isn't needed, since it never changes which score is highest
            predictions = logits.argmax(dim=1)

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples

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

    model = LetterClassifier(**MODEL_SETTINGS)
    # Takes raw scores (logits) and applies softmax internally (D24)
    loss_fn = nn.CrossEntropyLoss()
    # The optimizer gets direct access to every weight and bias the model tracks
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Infinity, so the first epoch always counts as an improvement
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = validate(model, val_loader, loss_fn)
        summary = f"Epoch {epoch:3d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val accuracy {val_accuracy:.1%}"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            # A deep copy, because state_dict() returns the live tensors that every optimizer step changes
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            summary += "  (best so far)"
        else:
            epochs_without_improvement += 1
        print(summary)

        if epochs_without_improvement >= PATIENCE:
            print(f"Stopping: no improvement for {PATIENCE} epochs")
            break

    if epochs_without_improvement < PATIENCE:
        print(f"Reached the {MAX_EPOCHS}-epoch limit")
    print(f"Best: epoch {best_epoch}, val loss {best_val_loss:.4f}, val accuracy {best_val_accuracy:.1%}")

    checkpoint = {
        "weights": best_weights,
        "model_settings": MODEL_SETTINGS,
        # The order of LETTERS is what each of the 24 outputs means, so it travels with the weights
        "letters": LETTERS,
        "training_info": {
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_accuracy": best_val_accuracy,
            "sampling": args.sampling,
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved the best model to {MODEL_PATH}")

if __name__ == "__main__":
    main()