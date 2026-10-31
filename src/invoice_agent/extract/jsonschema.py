"""Schema + prompt shared by the JSON-schema backends (Ollama, OpenAI-compatible APIs)."""

from __future__ import annotations

from typing import Any

from invoice_agent.extract.base import SYSTEM_PROMPT
from invoice_agent.extract.models import ExtractedInvoice


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


def close_objects(node: Any) -> Any:
    """Add additionalProperties: false to every object (needed by strict-mode APIs)."""
    if isinstance(node, dict):
        node = {k: close_objects(v) for k, v in node.items()}
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [close_objects(v) for v in node]
    return node


SCHEMA = require_all(inline_refs(ExtractedInvoice.model_json_schema()))
PROMPT = f"{SYSTEM_PROMPT}\n\nReturn one JSON object with these fields:\n{field_guide(SCHEMA)}"
