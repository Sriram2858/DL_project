"""Train and evaluate the VoxAssist intent classifier.

Run from the project root with: python train.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from model_utils import IntentNet, extract_features, make_training_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "intents.json"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SEED = 42


def load_dataset() -> tuple[list[str], list[int], list[str], dict[str, list[str]], list[str]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    intents = data["intents"]
    tags = [intent["tag"] for intent in intents]
    responses = {intent["tag"]: intent["responses"] for intent in intents}
    texts: list[str] = []
    labels: list[int] = []
    for class_index, intent in enumerate(intents):
        for pattern in intent["patterns"]:
            if not pattern.strip():
                continue
            texts.append(pattern.strip())
            labels.append(class_index)
    vocabulary = sorted({feature for text in texts for feature in extract_features(text)})
    return texts, labels, tags, responses, vocabulary


def split_by_class(labels: list[int], validation_fraction: float = 0.20) -> tuple[list[int], list[int]]:
    """Make a reproducible stratified split, holding out examples from each class."""
    rng = random.Random(SEED)
    class_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        class_indices.setdefault(label, []).append(index)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in class_indices.values():
        rng.shuffle(indices)
        count = max(1, round(len(indices) * validation_fraction)) if len(indices) > 2 else 1
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def train_network(
    features: torch.Tensor,
    labels: torch.Tensor,
    input_size: int,
    num_classes: int,
    epochs: int,
    seed: int = SEED,
) -> tuple[IntentNet, list[float]]:
    torch.manual_seed(seed)
    model = IntentNet(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return model, losses


def accuracy(model: IntentNet, features: torch.Tensor, labels: torch.Tensor) -> float:
    if len(labels) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    source_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    pattern_responses = {
        item["tag"]: item.get("pattern_responses", {})
        for item in source_data["intents"]
    }
    texts, labels, tags, responses, _ = load_dataset()
    train_indices, validation_indices = split_by_class(labels)

    # Evaluate on examples excluded from the evaluation model's training set.
    train_texts = [texts[i] for i in train_indices]
    train_labels_list = [labels[i] for i in train_indices]
    validation_texts = [texts[i] for i in validation_indices]
    validation_labels_list = [labels[i] for i in validation_indices]
    evaluation_vocabulary = sorted({feature for text in train_texts for feature in extract_features(text)})
    x_train = make_training_features(train_texts, evaluation_vocabulary)
    y_train = torch.tensor(train_labels_list, dtype=torch.long)
    x_validation = make_training_features(validation_texts, evaluation_vocabulary)
    y_validation = torch.tensor(validation_labels_list, dtype=torch.long)
    evaluation_model, evaluation_losses = train_network(
        x_train, y_train, len(evaluation_vocabulary), len(tags), epochs=40, seed=SEED
    )
    heldout_accuracy = accuracy(evaluation_model, x_validation, y_validation)

    # Fit the deployable model on every labelled example after measuring held-out accuracy.
    full_vocabulary = sorted({feature for text in texts for feature in extract_features(text)})
    x_all = make_training_features(texts, full_vocabulary)
    y_all = torch.tensor(labels, dtype=torch.long)
    final_model, final_losses = train_network(
        x_all, y_all, len(full_vocabulary), len(tags), epochs=45, seed=SEED
    )
    train_accuracy = accuracy(final_model, x_all, y_all)

    torch.save(final_model.state_dict(), MODEL_PATH)
    metadata = {
        "model_name": "VoxAssist IntentNet",
        "model_type": "Feed-forward neural network (MLP)",
        "feature_method": "Binary unigram and bigram multi-hot vector",
        "hidden_layers": [128, 64],
        "vocabulary": full_vocabulary,
        "tags": tags,
        "responses": responses,
        "pattern_responses": pattern_responses,
        "fallback_responses": source_data["fallback_responses"],
        "confidence_threshold": 0.46,
        "training_examples": len(texts),
        "num_intents": len(tags),
        "seed": SEED,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "evaluation_method": "Reproducible stratified holdout; examples used for validation are excluded from the evaluation model's fit.",
        "training_examples_total": len(texts),
        "training_examples_in_holdout_experiment": len(train_indices),
        "heldout_examples": len(validation_indices),
        "num_intents": len(tags),
        "vocabulary_size_final_model": len(full_vocabulary),
        "heldout_accuracy": round(heldout_accuracy, 4),
        "final_model_training_accuracy": round(train_accuracy, 4),
        "evaluation_model_final_loss": round(evaluation_losses[-1], 6),
        "evaluation_epochs": len(evaluation_losses),
        "final_training_epochs": len(final_losses),
        "final_training_loss": round(final_losses[-1], 6),
        "per_intent_pattern_counts": dict(Counter(tags[label] for label in labels)),
        "note": "This is a small course-project dataset. Treat holdout metrics as a sanity check, not as evidence of production-level speech understanding.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nVoxAssist model training complete")
    print("--------------------------------")
    print(f"Intent categories : {len(tags)}")
    print(f"Training examples : {len(texts)}")
    print(f"Vocabulary size   : {len(full_vocabulary)}")
    print(f"Held-out accuracy : {heldout_accuracy:.1%} ({len(validation_indices)} examples)")
    print(f"Train accuracy    : {train_accuracy:.1%}")
    print(f"Model saved       : {MODEL_PATH}")
    print(f"Metadata saved    : {METADATA_PATH}")
    print(f"Metrics saved     : {METRICS_PATH}\n")


if __name__ == "__main__":
    main()
