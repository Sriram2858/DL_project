import json
import unittest
from unittest.mock import patch
from urllib.error import URLError
from types import SimpleNamespace

from llm_provider import LLMProviderError, OllamaProvider, format_retrieved_context
from app import clean_response_text


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class LLMProviderTests(unittest.TestCase):
    def test_response_cleaner_removes_markdown_symbols_and_word(self):
        self.assertEqual(clean_response_text("**Hello** *there* — an asterisk"), "Hello there — an")

    def test_context_formatting_is_bounded_and_source_labelled(self):
        context = format_retrieved_context([{"source": "guide.md", "text": "Use the local server."}, {"source": "large.md", "text": "x" * 100}], max_chars=60)
        self.assertIn("guide.md", context)
        self.assertLessEqual(len(context), 60)

    @patch("llm_provider.urlopen")
    def test_ollama_chat_request_and_response(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse({"message": {"content": "A generated answer."}})
        provider = OllamaProvider(model="qwen3.5:2b", timeout=2)
        answer, passages = provider.generate("Explain this")
        self.assertEqual(answer, "A generated answer.")
        self.assertEqual(passages, [])
        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3.5:2b")
        self.assertFalse(body["stream"])
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 2)
        self.assertIn("not predicting an intent", body["messages"][0]["content"])

    @patch("llm_provider.urlopen", side_effect=URLError("connection refused"))
    def test_unavailable_ollama_is_helpful(self, _mocked_urlopen):
        with self.assertRaisesRegex(LLMProviderError, "Ollama is unavailable"):
            OllamaProvider().generate("hello")

    @patch("llm_provider.urlopen", side_effect=TimeoutError())
    def test_timeout_is_reported(self, _mocked_urlopen):
        with self.assertRaises(LLMProviderError):
            OllamaProvider(timeout=0.01).generate("hello")

    def test_flask_api_uses_llm_contract_and_provider_errors(self):
        import app
        original_provider, original_client = app.provider_name, app.llm_provider
        try:
            app.provider_name = "ollama"
            app.llm_provider = SimpleNamespace(generate=lambda _message: ("generated", [{"source": "guide.md"}]), model="qwen3.5:2b")
            client = app.app.test_client()
            response = client.post("/api/chat", json={"message": "hello"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["intent"], "llm_response")
            self.assertEqual(response.json["provider"], "ollama")
            app.llm_provider = SimpleNamespace(generate=lambda _message: (_ for _ in ()).throw(LLMProviderError("Ollama is unavailable")), model="qwen3.5:2b")
            response = client.post("/api/chat", json={"message": "hello"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("unavailable", response.json["error"])
        finally:
            app.provider_name, app.llm_provider = original_provider, original_client

    def test_hybrid_mode_uses_generation_for_unmatched_classifier_intent(self):
        import app
        original_provider, original_client = app.provider_name, app.llm_provider
        try:
            app.provider_name = "hybrid"
            app.authored_responses = {"greeting": ["Hello"]}
            app.llm_provider = SimpleNamespace(generate=lambda _message: ("general answer", []), model="qwen3.5:2b")
            with patch.object(app, "classify_message", return_value={"response": "generic", "intent": "unknown_clinc_intent", "recognized": True}):
                response = app.app.test_client().post("/api/chat", json={"message": "a general question"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["response"], "general answer")
            self.assertEqual(response.json["intent"], "llm_response")
        finally:
            app.provider_name, app.llm_provider = original_provider, original_client


if __name__ == "__main__":
    unittest.main()
