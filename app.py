"""Flask server for VoxAssist: speech-friendly neural-network chatbot."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

from model_utils import IntentNet, vectorize

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


def load_model() -> None:
    """Load the trained model and its matching preprocessing metadata."""
    global model, metadata
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        model, metadata = None, None
        return
    loaded_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    loaded_model = IntentNet(len(loaded_metadata["vocabulary"]), len(loaded_metadata["tags"]))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state)
    loaded_model.eval()
    model, metadata = loaded_model, loaded_metadata


def classify_message(message: str) -> dict:
    """Return a predicted intent, confidence, and a response."""
    if model is None or metadata is None:
        raise RuntimeError("The model is not trained yet. Run python train.py first.")
    features = vectorize(message, metadata["vocabulary"]).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[0]
    confidence, predicted_index = torch.max(probabilities, dim=0)
    confidence_value = float(confidence.item())
    predicted_tag = metadata["tags"][int(predicted_index.item())]

    if confidence_value < float(metadata.get("confidence_threshold", 0.42)):
        response = random.choice(metadata.get("fallback_responses") or ["I didn't understand that. Could you rephrase it?"])
        return {
            "response": response,
            "intent": "fallback",
            "intent_label": "Not confidently recognized",
            "confidence": round(confidence_value, 4),
            "recognized": False,
        }

    choices = metadata["responses"].get(predicted_tag, [])
    response = random.choice(choices) if choices else "I recognized your message, but I don't have a response for that intent yet."
    label = predicted_tag.replace("_", " ").title()
    return {
        "response": response,
        "intent": predicted_tag,
        "intent_label": label,
        "confidence": round(confidence_value, 4),
        "recognized": True,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok" if model is not None else "model_not_trained",
        "model_loaded": model is not None,
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
        result = classify_message(message)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(result)


# Load already-trained artifacts at startup. Training is intentionally a separate step.
load_model()

if __name__ == "__main__":
    if model is None:
        print("No trained model found. Run 'python train.py' before starting the app.")
    print("Starting VoxAssist at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
