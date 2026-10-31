"""Invoice extraction: document -> typed `Invoice`.

Backends: `OllamaExtractor` (default; free and local) and `ClaudeExtractor` (optional).
"""

from invoice_agent.extract.base import ExtractionResult, Extractor
from invoice_agent.extract.models import ExtractedInvoice
from invoice_agent.extract.normalize import to_invoice
from invoice_agent.extract.ollama import OllamaExtractor
from invoice_agent.extract.pdf_text import has_text_layer, pdf_to_text


def make_extractor(provider: str = "ollama", model: str | None = None, **kwargs) -> Extractor:
    if provider == "ollama":
        return OllamaExtractor(model=model or "qwen2.5:7b", **kwargs)
    if provider == "claude":
        from invoice_agent.extract.llm import DEFAULT_MODEL, ClaudeExtractor

        return ClaudeExtractor(model=model or DEFAULT_MODEL, **kwargs)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = ["ExtractedInvoice", "ExtractionResult", "Extractor", "OllamaExtractor", "has_text_layer",
           "make_extractor", "pdf_to_text", "to_invoice"]
