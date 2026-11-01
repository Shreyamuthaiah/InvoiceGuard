"""Claude backend (optional; needs ANTHROPIC_API_KEY). The default backend is Ollama."""

from __future__ import annotations

import time

from invoice_agent.extract.base import SYSTEM_PROMPT, ExtractionResult, finish, user_message
from invoice_agent.extract.models import ExtractedInvoice

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"  # extraction is transcription, not reasoning; raise if eval says so


class ClaudeExtractor:
    provider = "claude"

    def __init__(self, client=None, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT) -> None:
        import anthropic  # imported lazily so the free Ollama path doesn't need it

        self._anthropic = anthropic
        if client is None:
            from invoice_agent.extract.openai_compat import read_key

            # Env var / `ant auth login` profile first (SDK default); else a line in ./.env.
            key = read_key("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def extract_text(self, document_text: str) -> ExtractionResult:
        anthropic = self._anthropic
        result = ExtractionResult(invoice=None, raw=None, model=self.model, provider=self.provider)
        start = time.perf_counter()
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message(document_text)}],
                output_format=ExtractedInvoice,
                output_config={"effort": self.effort},
                # Re-run on Anthropic's recommended model if a safety classifier declines.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as e:
            result.error = f"connection error: {e}"
            return result
        except anthropic.RateLimitError as e:
            result.error = f"rate limited: {e.message}"
            return result
        except anthropic.APIStatusError as e:
            result.error = f"API error {e.status_code}: {e.message}"
            return result
        finally:
            result.latency_s = time.perf_counter() - start

        result.request_id = getattr(response, "_request_id", None)
        usage = response.usage
        result.input_tokens = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0) \
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        result.output_tokens = usage.output_tokens or 0
        if response.stop_reason == "refusal":
            result.error = "model refused the request"
            return result
        if response.stop_reason == "max_tokens":
            result.error = "output truncated (max_tokens)"
            return result
        return finish(result, response.parsed_output)
