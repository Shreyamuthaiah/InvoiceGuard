"""Invoice extraction: document -> typed `Invoice`.

Backends: free hosted APIs via `OpenAICompatExtractor` (Gemini default, Groq), local
`OllamaExtractor`, and optional `ClaudeExtractor`.
"""

from invoice_agent.extract.base import ExtractionResult, Extractor
from invoice_agent.extract.models import ExtractedInvoice
from invoice_agent.extract.normalize import to_invoice
from invoice_agent.extract.ollama import OllamaExtractor
from invoice_agent.extract.pdf_text import has_text_layer, pdf_to_text


PROVIDERS = ["gemini", "groq", "ollama", "claude"]


def make_extractor(provider: str = "gemini", model: str | None = None, **kwargs) -> Extractor:
    if provider in ("gemini", "groq"):
        from invoice_agent.extract.openai_compat import OpenAICompatExtractor

        return OpenAICompatExtractor(provider, model=model, **kwargs)
    if provider == "ollama":
        return OllamaExtractor(model=model or "qwen2.5:7b", **kwargs)
    if provider == "claude":
        from invoice_agent.extract.llm import DEFAULT_MODEL, ClaudeExtractor

        return ClaudeExtractor(model=model or DEFAULT_MODEL, **kwargs)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = ["ExtractedInvoice", "ExtractionResult", "Extractor", "OllamaExtractor", "PROVIDERS", "has_text_layer",
           "make_extractor", "pdf_to_text", "to_invoice"]
