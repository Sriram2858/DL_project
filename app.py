"""Flask server for VoxAssist: speech-friendly neural-network chatbot."""
from __future__ import annotations

import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch
import joblib
from flask import Flask, jsonify, render_template, request

from model_utils import IntentNet, vectorize
from benchmark_utils import vectorize_clinc
from llm_provider import KnowledgeRetriever, LLMProviderError, OllamaProvider

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
MAX_MESSAGE_LENGTH = 500

app = Flask(
    __name__,
    template_folder=str(ROOT / "frontend" / "templates"),
    static_folder=str(ROOT / "frontend" / "static"),
)

model: IntentNet | None = None
metadata: dict | None = None
benchmark_model = None
model_kind = "legacy"
authored_responses: dict[str, list[str]] = {}
authored_patterns: dict[str, str] = {}
provider_name = "classifier"
llm_provider: OllamaProvider | None = None
retrieval_enabled = False


def clean_response_text(text: str) -> str:
    """Remove Markdown markers that browsers may read aloud as punctuation names."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)
    cleaned = cleaned.replace(r"\div", " divided by ").replace(r"\times", " times ").replace(r"\cdot", " multiplied by ")
    cleaned = cleaned.replace(r"\leq", " <= ").replace(r"\geq", " >= ").replace(r"\neq", " != ")
    cleaned = re.sub(r"[`*_~]", "", cleaned)
    cleaned = re.sub(r"\basterisks?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def load_model() -> None:
    """Load the trained model and its matching preprocessing metadata."""
    global model, metadata, benchmark_model, model_kind, authored_responses, authored_patterns
    # The authored intents model is the default conversational classifier.
    # CLINC150 MLP/SVM remain selectable benchmark models.
    model_kind = os.environ.get("VOXASSIST_MODEL", "legacy").lower()
    try:
        authored = json.loads((ROOT / "intents.json").read_text(encoding="utf-8"))
        authored_responses = {item["tag"]: item["responses"] for item in authored["intents"]}
        authored_patterns = {
            pattern.strip().lower(): item["tag"]
            for item in authored["intents"]
            for pattern in item.get("patterns", [])
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        authored_responses = {}
        authored_patterns = {}
    if model_kind in {"mlp", "svm"}:
        try:
            authored = json.loads((ROOT / "intents.json").read_text(encoding="utf-8"))
            authored_responses = {item["tag"]: item["responses"] for item in authored["intents"]}
            if model_kind == "mlp":
                metadata = json.loads((ARTIFACT_DIR / "clinc_mlp_metadata.json").read_text(encoding="utf-8"))
                benchmark_model = IntentNet(len(metadata["vocabulary"]), len(metadata["tags"]))
                benchmark_model.load_state_dict(torch.load(ARTIFACT_DIR / "clinc_mlp_model.pt", map_location="cpu", weights_only=True))
                benchmark_model.eval()
            else:
                metadata = json.loads((ARTIFACT_DIR / "clinc_svm_metadata.json").read_text(encoding="utf-8"))
                benchmark_model = joblib.load(ARTIFACT_DIR / "clinc_svm.joblib")
            model = None
            return
        except (FileNotFoundError, KeyError, RuntimeError, ValueError):
            model, metadata, benchmark_model = None, None, None
            return
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        model, metadata = None, None
        return
    loaded_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    loaded_model = IntentNet(len(loaded_metadata["vocabulary"]), len(loaded_metadata["tags"]))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state)
    loaded_model.eval()
    model, metadata = loaded_model, loaded_metadata
    benchmark_model = None


def load_provider() -> None:
    """Configure response generation separately from intent classification."""
    global provider_name, llm_provider, retrieval_enabled
    provider_name = os.environ.get("VOXASSIST_PROVIDER", "classifier").lower()
    llm_provider = None
    retrieval_enabled = os.environ.get("VOXASSIST_RETRIEVAL", "0").lower() in {"1", "true", "yes", "on"}
    if provider_name not in {"ollama", "hybrid"}:
        return
    retriever = None
    if retrieval_enabled:
        try:
            retriever = KnowledgeRetriever(ROOT / "knowledge", os.environ.get("VOXASSIST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        except Exception as exc:
            raise LLMProviderError(f"Knowledge retrieval could not start: {exc}") from exc
    try:
        llm_provider = OllamaProvider(retriever=retriever)
    except (TypeError, ValueError) as exc:
        raise LLMProviderError(f"Invalid Ollama configuration: {exc}") from exc


def classify_message(message: str) -> dict:
    """Return a predicted intent, confidence, and a response."""
    if metadata is None or (model is None and benchmark_model is None):
        command = "python train_benchmark.py" if model_kind in {"mlp", "svm"} else "python train.py"
        raise RuntimeError(f"The selected model is not trained yet. Run {command} first.")
    direct_tag = authored_patterns.get(message.strip().lower())
    if direct_tag and model_kind == "legacy":
        pattern_response = select_pattern_response(message, direct_tag, metadata)
        choices = authored_responses.get(direct_tag, [])
        return {
            "response": clean_response_text(pattern_response or (random.choice(choices) if choices else "I recognized that request.")),
            "intent": direct_tag,
            "intent_label": direct_tag.replace("_", " ").title(),
            "confidence": 1.0,
            "recognized": True,
        }
    if model_kind in {"mlp", "svm"}:
        tags = [tag for tag in metadata["tags"] if tag != "oos"]
        if model_kind == "mlp":
            features = vectorize_clinc(message, metadata["vocabulary"]).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(benchmark_model(features), dim=1)[0]
            oos_index = metadata["tags"].index("oos")
            oos_score = float(probabilities[oos_index].item())
            id_indices = [index for index, tag in enumerate(metadata["tags"]) if tag != "oos"]
            predicted_tag = metadata["tags"][id_indices[int(probabilities[id_indices].argmax().item())]]
            confidence_value = float(probabilities[id_indices].max().item())
        else:
            decision = benchmark_model.decision_function([message])[0]
            oos_index = metadata["tags"].index("oos")
            id_indices = [index for index, tag in enumerate(metadata["tags"]) if tag != "oos"]
            predicted_tag = metadata["tags"][id_indices[int(decision[id_indices].argmax())]]
            oos_score = float(decision[oos_index])
            confidence_value = float(decision[id_indices].max())
        threshold = float(metadata.get("id_confidence_threshold", metadata.get("oos_threshold", metadata.get("confidence_threshold", 0.0))))
        if confidence_value <= threshold:
            return {"response": clean_response_text("I could not confidently map that to one of VoxAssist's supported topics. Please rephrase or ask about the project."), "intent": "fallback", "intent_label": "Out of scope", "confidence": round(confidence_value, 4), "recognized": False}
        choices = authored_responses.get(predicted_tag)
        response = clean_response_text(random.choice(choices) if choices else f"I recognized the CLINC150 intent '{predicted_tag}', but no response template is configured for it yet.")
        return {"response": response, "intent": predicted_tag, "intent_label": predicted_tag.replace("_", " ").title(), "confidence": round(confidence_value, 4), "recognized": True}
    features = vectorize(message, metadata["vocabulary"]).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[0]
    confidence, predicted_index = torch.max(probabilities, dim=0)
    confidence_value = float(confidence.item())
    predicted_tag = metadata["tags"][int(predicted_index.item())]

    if confidence_value < float(metadata.get("confidence_threshold", 0.42)):
        response = clean_response_text(random.choice(metadata.get("fallback_responses") or ["I didn't understand that. Could you rephrase it?"]))
        return {
            "response": response,
            "intent": "fallback",
            "intent_label": "Not confidently recognized",
            "confidence": round(confidence_value, 4),
            "recognized": False,
        }

    pattern_response = select_pattern_response(message, predicted_tag, metadata)
    if pattern_response:
        response = clean_response_text(pattern_response)
    else:
        choices = metadata["responses"].get(predicted_tag, [])
        response = clean_response_text(random.choice(choices) if choices else "I recognized your message, but I don't have a response for that intent yet.")
    label = predicted_tag.replace("_", " ").title()
    return {
        "response": clean_response_text(response),
        "intent": predicted_tag,
        "intent_label": label,
        "confidence": round(confidence_value, 4),
        "recognized": True,
    }


def select_pattern_response(message: str, tag: str, metadata: dict) -> str | None:
    """Pick the response for the closest authored pattern in the predicted intent."""
    choices = metadata.get("pattern_responses", {}).get(tag, {})
    if not choices:
        return None

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    query = normalize(message)
    query_words = set(query.split())
    best_pattern = None
    best_score = 0.0
    for pattern in choices:
        candidate = normalize(pattern)
        candidate_words = set(candidate.split())
        union = query_words | candidate_words
        overlap = len(query_words & candidate_words) / len(union) if union else 0.0
        sequence = SequenceMatcher(None, query, candidate).ratio()
        score = 0.55 * overlap + 0.45 * sequence
        if score > best_score:
            best_score = score
            best_pattern = pattern
    if best_pattern is None or best_score < 0.28:
        return None
    return choices[best_pattern]


def generate_message(message: str) -> dict:
    """Generate a free-form answer; this path does not claim classifier output."""
    if llm_provider is None:
        raise LLMProviderError("The Ollama provider is not configured. Set VOXASSIST_PROVIDER=ollama.")
    response, passages = llm_provider.generate(message)
    return {
        "response": clean_response_text(response),
        "intent": "llm_response",
        "intent_label": "Generated response",
        "confidence": None,
        "recognized": True,
        "provider": "ollama",
        "retrieved_sources": [item.get("source") for item in passages],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    loaded = model is not None or benchmark_model is not None
    if provider_name == "ollama":
        loaded = llm_provider is not None
    return jsonify({
        "status": "ok" if loaded else "model_not_trained",
        "model_loaded": loaded,
        "model_selection": model_kind,
        "provider": provider_name,
        "llm_model": llm_provider.model if llm_provider else None,
        "retrieval_enabled": retrieval_enabled,
        "model_name": metadata.get("model_name") if metadata else None,
        "intents": len(metadata["tags"]) if metadata else 0,
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400
    message = message.strip()
    if not message:
        return jsonify({"error": "Please enter or speak a message first."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Messages must be {MAX_MESSAGE_LENGTH} characters or fewer."}), 400
    try:
        if provider_name == "ollama":
            result = generate_message(message)
        elif provider_name == "hybrid":
            try:
                classified = classify_message(message)
            except RuntimeError:
                classified = None
            has_authored_response = bool(classified and classified.get("recognized") and classified.get("intent") in authored_responses)
            if has_authored_response:
                classified["provider"] = "classifier"
                result = classified
            else:
                try:
                    result = generate_message(message)
                except LLMProviderError:
                    if classified is None:
                        raise
                    classified["provider"] = "classifier_fallback"
                    result = classified
        else:
            result = classify_message(message)
    except (RuntimeError, LLMProviderError) as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(result)


# Load already-trained artifacts at startup. Training is intentionally a separate step.
load_model()
try:
    load_provider()
except LLMProviderError as exc:
    # Keep the server and health route available so the UI can show a useful setup error.
    llm_provider = None
    print(f"LLM provider setup failed: {exc}")

if __name__ == "__main__":
    if provider_name == "classifier" and model is None and benchmark_model is None:
        print("No trained model found. Run 'python train.py' before starting the app.")
    print("Starting VoxAssist at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
