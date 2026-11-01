"""Invoice extraction: document -> typed `Invoice`.

Backends: free hosted APIs via `OpenAICompatExtractor` (Gemini default, Groq), local
`OllamaExtractor`, and optional `ClaudeExtractor`.
"""

from invoiceguard.extract.base import ExtractionResult, Extractor
from invoiceguard.extract.models import ExtractedInvoice
from invoiceguard.extract.normalize import to_invoice
from invoiceguard.extract.ollama import OllamaExtractor
from invoiceguard.extract.pdf_text import has_text_layer, pdf_to_text

PROVIDERS = ["gemini", "groq", "ollama", "claude"]


def make_extractor(provider: str = "gemini", model: str | None = None, **kwargs) -> Extractor:
    if provider in ("gemini", "groq"):
        from invoiceguard.extract.openai_compat import OpenAICompatExtractor

        return OpenAICompatExtractor(provider, model=model, **kwargs)
    if provider == "ollama":
        return OllamaExtractor(model=model or "qwen2.5:7b", **kwargs)
    if provider == "claude":
        from invoiceguard.extract.llm import DEFAULT_MODEL, ClaudeExtractor

        return ClaudeExtractor(model=model or DEFAULT_MODEL, **kwargs)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = ["ExtractedInvoice", "ExtractionResult", "Extractor", "OllamaExtractor", "PROVIDERS", "has_text_layer",
           "make_extractor", "pdf_to_text", "to_invoice"]
