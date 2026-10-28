"""Shared CLINC150 loading, feature handling, and OOS evaluation helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch

from model_utils import IntentNet, extract_features, make_training_features, vectorize

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def load_clinc(path: Path) -> dict[str, list[list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
    if set(required) - set(data):
        raise ValueError("CLINC150 data is missing one or more official splits")
    return {split: data[split] for split in required}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def features(text: str) -> list[str]:
    tokens = tokenize(text)
    return [f"uni:{token}" for token in tokens] + [f"bi:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def make_vocabulary(texts: Iterable[str]) -> list[str]:
    return sorted({feature for text in texts for feature in features(text)})


def vectorize_clinc(text: str, vocabulary: list[str]) -> torch.Tensor:
    positions = {feature: i for i, feature in enumerate(vocabulary)}
    output = torch.zeros(len(vocabulary), dtype=torch.float32)
    for feature in set(features(text)):
        if feature in positions:
            output[positions[feature]] = 1.0
    return output


def make_features(texts: Iterable[str], vocabulary: list[str]) -> torch.Tensor:
    rows = [vectorize_clinc(text, vocabulary) for text in texts]
    return torch.stack(rows) if rows else torch.empty((0, len(vocabulary)), dtype=torch.float32)


def labels_for(rows: list[list[str]]) -> tuple[list[str], list[str]]:
    return [row[0] for row in rows], [row[1] for row in rows]


def predict_mlp(model: IntentNet, x: torch.Tensor, tags: list[str]) -> tuple[list[str], list[float]]:
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    values, indices = probs.max(dim=1)
    return [tags[int(i)] for i in indices], [float(v) for v in values]
