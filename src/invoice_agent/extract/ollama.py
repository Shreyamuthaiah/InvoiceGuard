"""Ollama backend: free, local, no API key. Default model qwen2.5:7b.

Uses Ollama's /api/chat with `format` set to the JSON schema, which constrains decoding
to schema-valid JSON. The schema only constrains output, so the field guide is also put
in the prompt; otherwise the model never sees what each field means.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from invoice_agent.extract.base import SYSTEM_PROMPT, ExtractionResult, finish, user_message
from invoice_agent.extract.models import ExtractedInvoice

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace "$ref": "#/$defs/X" with the definition itself (simpler for grammar engines)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def field_guide(schema: dict[str, Any], indent: str = "") -> str:
    """Compact human-readable list of fields + descriptions for the prompt."""
    lines = []
    for name, spec in schema.get("properties", {}).items():
        desc = spec.get("description", "")
        sub = spec if "properties" in spec else next(
            (s for s in spec.get("anyOf", []) if "properties" in s), None)
        items = spec.get("items") if spec.get("type") == "array" else None
        lines.append(f"{indent}- {name}" + (f": {desc}" if desc else ""))
        if sub:
            lines.append(field_guide(sub, indent + "  "))
        elif items and "properties" in items:
            lines.append(f"{indent}  (each item:)")
            lines.append(field_guide(items, indent + "  "))
    return "\n".join(lines)


def require_all(node: Any) -> Any:
    """Mark every property required (nullable ones still accept null). Without this,
    constrained decoding lets small models skip optional keys entirely."""
    if isinstance(node, dict):
        node = {k: require_all(v) for k, v in node.items()}
        if "properties" in node:
            node["required"] = list(node["properties"])
        return node
    if isinstance(node, list):
        return [require_all(v) for v in node]
    return node


SCHEMA = require_all(inline_refs(ExtractedInvoice.model_json_schema()))
PROMPT = f"{SYSTEM_PROMPT}\n\nReturn one JSON object with these fields:\n{field_guide(SCHEMA)}"


class OllamaExtractor:
    provider = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout_s: float = 900,
                 client: httpx.Client | None = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_s)

    def extract_text(self, document_text: str) -> ExtractionResult:
        result = ExtractionResult(invoice=None, raw=None, model=self.model, provider=self.provider)
        start = time.perf_counter()
        try:
            resp = self.client.post(f"{self.host}/api/chat", json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": user_message(document_text)},
                ],
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
