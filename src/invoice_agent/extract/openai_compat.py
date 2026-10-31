"""Backend for free hosted APIs that speak the OpenAI chat-completions protocol.

Presets: Google Gemini (default; free tier for Flash models) and Groq (free tier).
Both accept `response_format={"type": "json_schema", ...}` for schema-constrained JSON.
API keys come from environment variables and are never logged or written to disk.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from invoice_agent.extract.base import ExtractionResult, finish, user_message
from invoice_agent.extract.jsonschema import PROMPT, SCHEMA, close_objects


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: str
    default_model: str
    rpm: int  # stay under the free tier's requests-per-minute limit
    strict: bool  # send "strict": true with the schema


PRESETS = {
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                     "gemini-3.8-flash", rpm=10, strict=False),
    "groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", rpm=3, strict=True),
}

RETRY_STATUSES = {429, 500, 502, 503, 504}


def read_key(name: str, env_file: Path | None = None) -> str | None:
    """Environment variable first, then a NAME=value line in ./.env (gitignored)."""
    if os.environ.get(name):
        return os.environ[name]
    path = env_file or Path.cwd() / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


class _Throttle:
    """Spaces requests evenly to respect a requests-per-minute limit across threads."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


class OpenAICompatExtractor:
    def __init__(self, provider: str = "gemini", model: str | None = None, api_key: str | None = None,
                 rpm: int | None = None, max_retries: int = 6, timeout_s: float = 120,
                 client: httpx.Client | None = None, sleep=time.sleep) -> None:
        if provider not in PRESETS:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PRESETS)}")
        self.preset = PRESETS[provider]
        self.provider = provider
        self.model = model or self.preset.default_model
        key = api_key or read_key(self.preset.key_env)
        if not key:
            raise RuntimeError(f"set {self.preset.key_env} (environment variable or a line in .env) to use provider {provider!r}")
        self.client = client or httpx.Client(timeout=timeout_s)
        self._headers = {"Authorization": f"Bearer {key}"}
        self.throttle = _Throttle(self.preset.rpm if rpm is None else rpm)
        self.max_retries = max_retries
        self._sleep = sleep

    def _payload(self, document_text: str) -> dict:
        schema = {"name": "invoice", "schema": close_objects(SCHEMA) if self.preset.strict else SCHEMA}
        if self.preset.strict:
            schema["strict"] = True
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_message(document_text)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
        }

    def _post(self, payload: dict) -> httpx.Response:
        url = f"{self.preset.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            self.throttle.wait()
            resp = self.client.post(url, json=payload, headers=self._headers)
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                return resp
            retry_after = resp.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else min(10 * 2 ** attempt, 120)
            except ValueError:
                delay = min(10 * 2 ** attempt, 120)
            self._sleep(delay)
        return resp  # pragma: no cover

    def extract_text(self, document_text: str) -> ExtractionResult:
        result = ExtractionResult(invoice=None, raw=None, model=self.model, provider=self.provider)
        start = time.perf_counter()
        try:
            resp = self._post(self._payload(document_text))
        except httpx.TimeoutException:
            result.error = "request timed out"
            return result
        except httpx.TransportError as e:
            result.error = f"network error: {type(e).__name__}"
            return result
        finally:
            result.latency_s = time.perf_counter() - start

        if resp.status_code != 200:
            result.error = f"{self.provider} error {resp.status_code}: {resp.text[:300]}"
            return result
        body = resp.json()
        usage = body.get("usage") or {}
        result.input_tokens = usage.get("prompt_tokens", 0) or 0
        result.output_tokens = usage.get("completion_tokens", 0) or 0
        result.request_id = resp.headers.get("x-request-id")
        choice = (body.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            result.error = "output truncated (length)"
            return result
        content = (choice.get("message") or {}).get("content")
        if content:
            content = content.strip()
            if content.startswith("```"):  # some models wrap JSON in a code fence
                content = content.strip("`").removeprefix("json").strip()
            try:
                json.loads(content)
            except json.JSONDecodeError:
                result.error = "model returned invalid JSON"
                return result
        return finish(result, content)
