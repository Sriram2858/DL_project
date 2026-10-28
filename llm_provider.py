"""Local Ollama response generation and optional lightweight knowledge retrieval."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProviderError(RuntimeError):
    """A user-facing provider or configuration failure."""


def format_retrieved_context(passages: list[dict[str, str]], max_chars: int = 6000) -> str:
    """Format approved passages for a prompt without creating a large database."""
    sections: list[str] = []
    used = 0
    for index, passage in enumerate(passages, 1):
        source = passage.get("source", "knowledge file")
        text = passage.get("text", "").strip()
        if not text:
            continue
        section = f"[Source {index}: {source}]\n{text}"
        if used + len(section) > max_chars:
            break
        sections.append(section)
        used += len(section) + 2
    return "\n\n".join(sections)


class KnowledgeRetriever:
    """Optional sentence-transformer retrieval over a small local knowledge folder."""

    def __init__(self, directory: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError as exc:
            raise LLMProviderError("Retrieval is enabled, but sentence-transformers is not installed. Install requirements-llm.txt.") from exc
        self._np = np
        self.model = SentenceTransformer(model_name, device="cpu")
        self.passages = self._read_passages(directory)
        if not self.passages:
            self.embeddings = None
        else:
            self.embeddings = self.model.encode([item["text"] for item in self.passages], normalize_embeddings=True)

    @staticmethod
    def _read_passages(directory: Path) -> list[dict[str, str]]:
        passages: list[dict[str, str]] = []
        if not directory.exists():
            return passages
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for chunk in (part.strip() for part in text.split("\n\n")):
                if chunk:
                    passages.append({"source": str(path), "text": chunk})
        return passages

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        if self.embeddings is None:
            return []
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ vector
        indices = scores.argsort()[::-1][:top_k]
        return [{**self.passages[int(index)], "score": f"{float(scores[index]):.4f}"} for index in indices]


class OllamaProvider:
    """Ollama's local /api/chat provider using only Python's standard library."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float | None = None, retriever: KnowledgeRetriever | None = None) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
        self.retriever = retriever

    def generate(self, message: str) -> tuple[str, list[dict[str, str]]]:
        passages = self.retriever.search(message) if self.retriever else []
        context = format_retrieved_context(passages)
        system = ("You are the response assistant for a speech chatbot. Answer clearly and concisely. "
                  "You are generating a free-form response, not predicting an intent label. "
                  "Do not invent project-specific facts, files, metrics, or capabilities. "
                  "If evidence is supplied, use it and distinguish supported facts from unsupported claims. "
                  "If the evidence is insufficient, say that you do not have enough information.")
        if context:
            system += f"\n\nApproved knowledge passages:\n{context}"
        payload = {"model": self.model, "stream": False, "think": False, "options": {"num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "512"))}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}]}
        request = Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMProviderError(f"Ollama returned HTTP {exc.code}. Is model '{self.model}' installed? {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMProviderError(f"Ollama is unavailable at {self.base_url}. Start Ollama and run 'ollama pull {self.model}'.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMProviderError("Ollama returned an invalid response.") from exc
        content = ((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise LLMProviderError("Ollama returned no message content.")
        return content, passages
