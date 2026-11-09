"""The letter classifier: a small multilayer perceptron (see D24 in architecture.md)."""

from torch import nn

from asl.features import NUM_FEATURES
from asl.labels import NUM_CLASSES


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