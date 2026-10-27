"""Lightweight checks for the trained VoxAssist model and dataset."""
import json
import unittest
from pathlib import Path

import torch

from model_utils import IntentNet, vectorize

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class VoxAssistModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "intents.json").read_text(encoding="utf-8"))
        cls.metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
        cls.model = IntentNet(len(cls.metadata["vocabulary"]), len(cls.metadata["tags"]))
        cls.model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True))
        cls.model.eval()

    def predict(self, text):
        with torch.no_grad():
            probabilities = torch.softmax(self.model(vectorize(text, self.metadata["vocabulary"]).unsqueeze(0)), dim=1)[0]
        index = int(probabilities.argmax().item())
        return self.metadata["tags"][index], float(probabilities[index].item())

    def test_intents_dataset_is_not_empty(self):
        self.assertGreaterEqual(len(self.data["intents"]), 10)
        self.assertTrue(all(intent["patterns"] and intent["responses"] for intent in self.data["intents"]))

    def test_model_and_metadata_dimensions_match(self):
        self.assertEqual(len(self.metadata["tags"]), self.model.layers[-1].out_features)
        self.assertEqual(len(self.metadata["vocabulary"]), self.model.layers[0].in_features)

    def test_common_prompts_map_to_expected_intents(self):
        examples = {
            "Explain deep learning": "deep_learning",
            "How is the model trained?": "model_training",
            "What is speech recognition?": "speech_recognition",
            "Explain this project": "project_overview",
            "How do I fix my mic?": "microphone_troubleshooting",
        }
        for message, expected_intent in examples.items():
            with self.subTest(message=message):
                predicted, confidence = self.predict(message)
                self.assertEqual(predicted, expected_intent)
                self.assertGreater(confidence, self.metadata["confidence_threshold"])

    def test_vectorizer_returns_expected_size(self):
        vector = vectorize("hello neural network", self.metadata["vocabulary"])
        self.assertEqual(vector.shape[0], len(self.metadata["vocabulary"]))
        self.assertTrue(torch.all((vector == 0) | (vector == 1)).item())


if __name__ == "__main__":
    unittest.main()
