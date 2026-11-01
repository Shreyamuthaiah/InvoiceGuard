"""Ollama backend: free, local, no API key. Default model qwen2.5:7b.

Uses Ollama's /api/chat with `format` set to the JSON schema, which constrains decoding
to schema-valid JSON. The schema only constrains output, so the field guide is also put
in the prompt; otherwise the model never sees what each field means.
"""

from __future__ import annotations

import base64
import json
import time

import httpx

from invoiceguard.extract.base import ExtractionResult, finish, image_message, user_message
from invoiceguard.extract.jsonschema import PROMPT, SCHEMA  # noqa: F401 (re-exported)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaExtractor:
    provider = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout_s: float = 900,
                 client: httpx.Client | None = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_s)

    def extract_text(self, document_text: str, hint: str | None = None) -> ExtractionResult:
        return self._run({"role": "user", "content": user_message(document_text, hint)})

    def extract_image(self, image: bytes, media_type: str = "image/jpeg", hint: str | None = None) -> ExtractionResult:
        """Needs a vision model (e.g. qwen2.5vl, llava); text-only models will fail or guess."""
        return self._run({"role": "user", "content": image_message(hint),
                          "images": [base64.b64encode(image).decode()]})

    def _run(self, user: dict) -> ExtractionResult:
        result = ExtractionResult(invoice=None, raw=None, model=self.model, provider=self.provider)
        start = time.perf_counter()
        try:
            resp = self.client.post(f"{self.host}/api/chat", json={
                "model": self.model,
                "messages": [{"role": "system", "content": PROMPT}, user],
                "format": SCHEMA,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            })
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError:
            result.error = f"cannot reach Ollama at {self.host} (is `ollama serve` running?)"
            return result
        except httpx.TimeoutException:
            result.error = "Ollama request timed out"
            return result
        except httpx.HTTPStatusError as e:
            result.error = f"Ollama error {e.response.status_code}: {e.response.text[:200]}"
            return result
        finally:
            result.latency_s = time.perf_counter() - start

        result.input_tokens = body.get("prompt_eval_count", 0) or 0
        result.output_tokens = body.get("eval_count", 0) or 0
        if body.get("done_reason") == "length":
            result.error = "output truncated (context/length limit)"
            return result
        content = (body.get("message") or {}).get("content")
        if content:
            try:
                json.loads(content)
            except json.JSONDecodeError:
                result.error = "model returned invalid JSON"
                return result
        return finish(result, content)
