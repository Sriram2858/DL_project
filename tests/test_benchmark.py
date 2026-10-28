import json
import os
import unittest
from pathlib import Path

from benchmark_utils import load_clinc

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_clinc(ROOT / "data/clinc150/data_full.json")

    def test_official_splits_are_preserved(self):
        self.assertEqual(set(self.data), {"train", "val", "test", "oos_train", "oos_val", "oos_test"})
        self.assertEqual(len(self.data["train"]), 15000)
        self.assertEqual(len(self.data["test"]), 4500)
        self.assertTrue(all(label != "oos" for _, label in self.data["train"]))
        self.assertTrue(all(label == "oos" for _, label in self.data["oos_test"]))

    def test_artifacts_and_metrics_if_benchmark_has_run(self):
        metrics = ROOT / "artifacts/comparison_metrics.json"
        if not metrics.exists():
            self.skipTest("Run python train_benchmark.py to create benchmark artifacts")
        data = json.loads(metrics.read_text(encoding="utf-8"))
        self.assertEqual({model["model"] for model in data["models"]}, {"mlp", "svm"})
        self.assertTrue((ROOT / "artifacts/clinc_mlp_model.pt").exists())
        self.assertTrue((ROOT / "artifacts/clinc_svm.joblib").exists())

    def test_authored_intent_set_is_expanded(self):
        authored = json.loads((ROOT / "intents.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(authored["intents"]), 50)
        self.assertEqual(len({item["tag"] for item in authored["intents"]}), len(authored["intents"]))

    def test_flask_api_compatibility_and_validation(self):
        import app
        client = app.app.test_client()
        health = client.get("/api/health")
        self.assertIn(health.status_code, (200, 503))
        self.assertEqual(client.post("/api/chat", json={"message": ""}).status_code, 400)
        self.assertEqual(client.post("/api/chat", json={"message": "x" * 501}).status_code, 400)

    def test_expanded_authored_intents_return_authored_replies(self):
        import app
        client = app.app.test_client()
        coin = client.post("/api/chat", json={"message": "flip a coin"})
        self.assertEqual(coin.status_code, 200)
        self.assertEqual(coin.json["intent"], "coin_flip")
        pattern_reply = app.metadata["pattern_responses"]["coin_flip"]["flip a coin"]
        self.assertEqual(coin.json["response"], pattern_reply)


if __name__ == "__main__":
    unittest.main()
