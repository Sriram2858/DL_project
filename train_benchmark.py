"""Train and evaluate the VoxAssist MLP and a TF-IDF + LinearSVC baseline on CLINC150.

Only official train/validation/test splits are used. The test split is read once for
the final report and is never used for fitting or threshold selection.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from benchmark_utils import (features, labels_for, load_clinc, make_features,
                             make_vocabulary, predict_mlp, vectorize_clinc)
from model_utils import IntentNet

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/clinc150/data_full.json"
ARTIFACTS = ROOT / "artifacts"
SEED = 42


def train_mlp(texts: list[str], labels: list[int], vocab: list[str], classes: list[str]) -> tuple[IntentNet, float]:
    torch.manual_seed(SEED)
    model = IntentNet(len(vocab), len(classes))
    x = make_features(texts, vocab)
    y = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(45):
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
    return model, time.perf_counter() - started


def scores_mlp(model, texts: list[str], vocab: list[str], classes: list[str]):
    x = make_features(texts, vocab)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    oos_index = classes.index("oos")
    id_scores, id_indices = torch.cat((probabilities[:, :oos_index], probabilities[:, oos_index + 1:]), dim=1).max(dim=1)
    id_classes = classes[:oos_index] + classes[oos_index + 1:]
    return [id_classes[int(i)] for i in id_indices], probabilities[:, oos_index].tolist(), id_scores.tolist()


def scores_svm(pipeline, texts: list[str], classes: list[str]):
    decision = pipeline.decision_function(texts)
    oos_index = list(classes).index("oos")
    id_decision = decision[:, :oos_index] if oos_index == len(classes) - 1 else __import__("numpy").delete(decision, oos_index, axis=1)
    indices = id_decision.argmax(axis=1)
    return [classes[:oos_index][int(i)] if oos_index == len(classes) - 1 else [c for c in classes if c != "oos"][int(i)] for i in indices], decision[:, oos_index].tolist(), id_decision.max(axis=1).tolist()


def choose_threshold(oos_scores: list[float], id_scores: list[float], oos_labels: list[bool]) -> float:
    """Choose a low in-domain confidence cutoff on official validation rows."""
    del oos_scores
    candidates = sorted(set(id_scores))
    best = (0.0, 0.0)
    best_fallback = (1.0, 0.0)
    for threshold in candidates:
        predicted = [score <= threshold for score in id_scores]
        truth = oos_labels
        f1 = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)[2]
        false_rejection = sum(pred and not actual for pred, actual in zip(predicted, truth)) / max(sum(not actual for actual in truth), 1)
        if false_rejection < best_fallback[0]:
            best_fallback = (false_rejection, threshold)
        if false_rejection <= 0.10 and f1 > best[0]:
            best = (f1, threshold)
    return float(best[1] if best[0] else best_fallback[1])


def evaluate(name, true_labels, predicted_id, oos_scores, id_scores, threshold, classes):
    actual = [label if label != "oos" else "oos" for label in true_labels]
    predicted = ["oos" if score <= threshold else label for label, score in zip(predicted_id, id_scores)]
    del oos_scores
    report = classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0)
    id_mask = [label != "oos" for label in actual]
    in_domain_accuracy = accuracy_score([a for a, keep in zip(actual, id_mask) if keep], [p for p, keep in zip(predicted, id_mask) if keep])
    oos_truth = [a == "oos" for a in actual]
    oos_pred = [p == "oos" for p in predicted]
    p, r, f, _ = precision_recall_fscore_support(oos_truth, oos_pred, average="binary", zero_division=0)
    false_rejection = sum(p == "oos" for p, keep in zip(predicted, id_mask) if keep) / max(sum(id_mask), 1)
    cm = confusion_matrix(actual, predicted, labels=classes)
    pairs = []
    for i, row in enumerate(cm):
        for j, count in enumerate(row):
            if i != j and count:
                pairs.append((int(count), classes[i], classes[j]))
    pairs.sort(reverse=True)
    return {"model": name, "in_domain_accuracy": in_domain_accuracy, "overall_accuracy": accuracy_score(actual, predicted),
            "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"], "oos_precision": p, "oos_recall": r, "oos_f1": f,
            "oos_false_rejection_rate": false_rejection, "oos_threshold": threshold,
            "top_confusions": [{"count": n, "actual": a, "predicted": b} for n, a, b in pairs[:10]]}


def main() -> None:
    data = load_clinc(DATA_PATH)
    train_rows = data["train"] + data["oos_train"]
    val_rows = data["val"] + data["oos_val"]
    test_rows = data["test"] + data["oos_test"]
    train_texts, train_labels = labels_for(train_rows)
    val_texts, val_labels = labels_for(val_rows)
    test_texts, test_labels = labels_for(test_rows)
    classes = sorted({label for _, label in data["train"]} | {"oos"})
    label_index = {label: i for i, label in enumerate(classes)}
    numeric = [label_index[label] for label in train_labels]
    vocab = make_vocabulary(train_texts)
    ARTIFACTS.mkdir(exist_ok=True)

    mlp, mlp_time = train_mlp(train_texts, numeric, vocab, classes)
    mlp_val_id, mlp_val_oos, mlp_val_id_scores = scores_mlp(mlp, val_texts, vocab, classes)
    mlp_threshold = choose_threshold(mlp_val_oos, mlp_val_id_scores, [label == "oos" for label in val_labels])
    mlp_test_id, mlp_test_oos, mlp_test_id_scores = scores_mlp(mlp, test_texts, vocab, classes)
    torch.save(mlp.state_dict(), ARTIFACTS / "clinc_mlp_model.pt")
    (ARTIFACTS / "clinc_mlp_metadata.json").write_text(json.dumps({"vocabulary": vocab, "tags": classes, "id_confidence_threshold": mlp_threshold, "feature_method": "binary unigram and bigram multi-hot", "training_rows": len(train_rows)}, indent=2), encoding="utf-8")

    svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)), ("classifier", LinearSVC(random_state=SEED))])
    started = time.perf_counter(); svm.fit(train_texts, train_labels); svm_time = time.perf_counter() - started
    svm_val_id, svm_val_oos, _ = scores_svm(svm, val_texts, classes)
    svm_val_id_scores = svm.decision_function(val_texts)[:, :-1].max(axis=1).tolist()
    svm_val_threshold = choose_threshold(svm_val_oos, svm_val_id_scores, [label == "oos" for label in val_labels])
    svm_test_id, svm_test_oos, svm_test_id_scores = scores_svm(svm, test_texts, classes)
    joblib.dump(svm, ARTIFACTS / "clinc_svm.joblib")
    (ARTIFACTS / "clinc_svm_metadata.json").write_text(json.dumps({"tags": classes, "id_confidence_threshold": svm_val_threshold, "feature_method": "TF-IDF word unigrams and bigrams", "training_rows": len(train_rows)}, indent=2), encoding="utf-8")

    mlp_metrics = evaluate("mlp", test_labels, mlp_test_id, mlp_test_oos, mlp_test_id_scores, mlp_threshold, classes)
    svm_metrics = evaluate("svm", test_labels, svm_test_id, svm_test_oos, svm_test_id_scores, svm_val_threshold, classes)
    for metrics, elapsed, files in ((mlp_metrics, mlp_time, ["clinc_mlp_model.pt", "clinc_mlp_metadata.json"]), (svm_metrics, svm_time, ["clinc_svm.joblib", "clinc_svm_metadata.json"])):
        metrics["training_time_seconds"] = elapsed
        metrics["artifact_size_bytes"] = sum((ARTIFACTS / file).stat().st_size for file in files)
    output = {"dataset": "CLINC150 data_full.json", "citation": "Larson et al., OOS evaluation dataset, CC BY 4.0", "splits": {key: len(value) for key, value in data.items()}, "models": [mlp_metrics, svm_metrics]}
    (ARTIFACTS / "comparison_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (ARTIFACTS / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key, value in mlp_metrics.items() if not isinstance(value, list)]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: value for key, value in metrics.items() if key in fields} for metrics in (mlp_metrics, svm_metrics)])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
