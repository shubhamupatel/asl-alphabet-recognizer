"""The letter classifier (see D24 in architecture.md), plus saving, loading, and running it (D32, D34)."""

from pathlib import Path

import torch
from torch import nn

from asl.features import NUM_FEATURES
from asl.labels import LETTERS, NUM_CLASSES


class LetterClassifier(nn.Module):
    """Maps a batch of 63-number feature vectors to 24 letter scores (logits) each."""

    def __init__(self, first_hidden=128, second_hidden=64, dropout=0.2):
        super().__init__()
        # 63 -> 128 -> 64 -> 24
        self.layers = nn.Sequential(
            nn.Linear(NUM_FEATURES, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, NUM_CLASSES),
        )

    def forward(self, features):
        return self.layers(features)


def save_checkpoint(path, weights, model_settings, training_info):
    """Save trained weights with everything needed to rebuild and interpret the model (D32)."""
    checkpoint = {
        "weights": weights,
        "model_settings": model_settings,
        # The order of LETTERS is what each of the 24 outputs means, so it travels with the weights
        "letters": LETTERS,
        "training_info": training_info,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """Rebuild a saved classifier, ready to run, and return it with its training info (D32)."""
    # weights_only refuses anything but tensors and plain data, so a tampered file can't run code
    checkpoint = torch.load(path, weights_only=True)
    # A different letter order would silently mislabel every prediction, so stop instead
    if checkpoint["letters"] != LETTERS:
        raise ValueError(f"{path} was trained on letters {checkpoint['letters']}, but the current labels are {LETTERS}")
    model = LetterClassifier(**checkpoint["model_settings"])
    # Raises an error if any saved weight doesn't fit the rebuilt model
    model.load_state_dict(checkpoint["weights"])
    # Ready to run, even for callers that never use predict(), like the live demo
    model.eval()
    return model, checkpoint["training_info"]


def predict(model, loader):
    """Run the model over every batch and return all scores (logits) and true labels, in loader order (D34)."""
    # Turns dropout off
    model.eval()
    all_logits = []
    all_labels = []
    # Nothing learns here, so skip recording the operations backpropagation would need
    with torch.no_grad():
        for features, labels in loader:
            all_logits.append(model(features))
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)