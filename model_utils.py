"""Shared preprocessing and model definition for VoxAssist."""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch
from torch import nn

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text without requiring an external NLP package."""
    return TOKEN_PATTERN.findall(text.lower())


def extract_features(text: str) -> list[str]:
    """Return binary unigram and bigram features for one utterance."""
    tokens = tokenize(text)
    features = [f"uni:{token}" for token in tokens]
    features.extend(f"bi:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return features


def vectorize(text: str, vocabulary: Sequence[str]) -> torch.Tensor:
    """Create a multi-hot input vector in the order defined by vocabulary."""
    positions = {feature: i for i, feature in enumerate(vocabulary)}
    vector = torch.zeros(len(vocabulary), dtype=torch.float32)
    for feature in set(extract_features(text)):
        position = positions.get(feature)
        if position is not None:
            vector[position] = 1.0
    return vector


class IntentNet(nn.Module):
    """A small multilayer perceptron for supervised intent classification."""

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def make_training_features(texts: Iterable[str], vocabulary: Sequence[str]) -> torch.Tensor:
    rows = [vectorize(text, vocabulary) for text in texts]
    if not rows:
        return torch.empty((0, len(vocabulary)), dtype=torch.float32)
    return torch.stack(rows)
