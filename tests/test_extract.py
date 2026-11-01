from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from invoice_agent.evaluate import compare, run_predictions, score
from invoice_agent.extract import OllamaExtractor, pdf_to_text, to_invoice
from invoice_agent.extract.llm import ClaudeExtractor
from invoice_agent.extract.models import ExtractedAddress, ExtractedBank, ExtractedInvoice, ExtractedLineItem
from invoice_agent.extract.normalize import (
    NormalizationError, date_candidates, parse_money, parse_number, parse_rate, resolve_dates,
)
from invoice_agent.schema import GroundTruth
from invoice_agent.synth import DatasetGenerator, write_dataset
from invoice_agent.synth.render import STYLES, fmt_money, fmt_qty, fmt_rate, fmt_unit, money_style


# ---------- number / date normalisation ----------

@pytest.mark.parametrize("text,expected", [
    ("$1,234.56", "1234.56"), ("1.234,56 €", "1234.56"), ("£0.00", "0.00"), ("2.093,04 €", "2093.04"),
    ("220,32 €", "220.32"), ("$12", "12"), ("-$5.10", "-5.10"), ("(1,000.00)", "-1000.00"),
    ("1,000", "1000"), ("1.000.000,5", "1000000.5"), ("EUR 3 400,00", "3400.00"),
])
def test_parse_number(text, expected):
    assert parse_number(text) == Decimal(expected)


def test_parse_small_unit_prices_keep_precision():
    assert parse_number("$0.0450") == Decimal("0.0450")
    assert parse_number("0,0450 €") == Decimal("0.0450")


def test_parse_money_rounds_and_rejects_garbage():
    assert parse_money("$10.0051") == Decimal("10.01")
    with pytest.raises(NormalizationError):
        parse_number("n/a")


@pytest.mark.parametrize("text,expected", [
    ("8.75%", "0.0875"), ("(19%)", "0.19"), ("Tax (0%)", "0"), ("VAT 20%", "0.2"), ("0.0725", "0.0725"),
])
def test_parse_rate(text, expected):
    assert parse_rate(text) == Decimal(expected)


def test_date_formats():
    assert date_candidates("March 24, 2026") == {"text": date(2026, 3, 24)}
    assert date_candidates("2026-08-09") == {"text": date(2026, 8, 9)}
    assert date_candidates("September 05, 2026") == {"text": date(2026, 9, 5)}
    assert date_candidates("Sept 5, 2026") == {"text": date(2026, 9, 5)}
    assert date_candidates("25/04/2026") == {"dmy": date(2026, 4, 25)}  # only one valid reading
    assert set(date_candidates("03/04/2026")) == {"mdy", "dmy"}


def test_ambiguous_dates_resolved_by_terms_then_country():
    # 03/04 + Net 30: as d/m/y -> 3 Apr -> 3 May = 30 days; as m/d/y -> 4 Mar -> 5 Apr = 32 days.
    inv, due, w = resolve_dates("03/04/2026", "03/05/2026", 30, "United States")
    assert (inv, due) == (date(2026, 4, 3), date(2026, 5, 3)) and not w
    # Terms don't disambiguate -> fall back to country.
    inv, _, w = resolve_dates("03/04/2026", None, None, "Germany")
    assert inv == date(2026, 4, 3) and w
    inv, _, _ = resolve_dates("03/04/2026", None, None, "United States")
    assert inv == date(2026, 3, 4)


# ---------- a "perfect" extraction built from labels in each template's printed format ----------

def printed_extraction(gt: GroundTruth) -> ExtractedInvoice:
    """What a flawless transcriber would return for this document."""
    inv, tpl = gt.invoice, gt.render.template
    ms = money_style(tpl, inv.currency)
    d = lambda x: STYLES[tpl]["date"](x, inv.currency)  # noqa: E731
    m = lambda x: fmt_money(x, inv.currency, ms)  # noqa: E731
    addr = lambda a: ExtractedAddress(**a.model_dump())  # noqa: E731
    return ExtractedInvoice(
        invoice_number=inv.invoice_number, invoice_date=d(inv.invoice_date), due_date=d(inv.due_date),
        po_number=inv.po_number, vendor_name=inv.vendor_name, vendor_address=addr(inv.vendor_address),
        vendor_tax_id=inv.vendor_tax_id, vendor_email=inv.vendor_email, bill_to_name=inv.bill_to_name,
        bill_to_address=addr(inv.bill_to_address), currency=inv.currency,
        line_items=[ExtractedLineItem(description=li.description, quantity=fmt_qty(li.quantity),
                                      unit_price=fmt_unit(li.unit_price, inv.currency, ms), amount=m(li.amount))
                    for li in inv.line_items],
        subtotal=m(inv.subtotal), tax_rate=fmt_rate(inv.tax_rate), tax_amount=m(inv.tax_amount),
        total=m(inv.total), payment_terms=inv.payment_terms,
        remit_to=ExtractedBank(**inv.remit_to.model_dump()),
    )


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    ds = DatasetGenerator(seed=11, n_vendors=12, scan_ratio=0.2).generate(60)
    write_dataset(ds, out)
    return ds, out


def test_normalisation_recovers_labels_for_every_printed_format(small_dataset):
    ds, _ = small_dataset
    for gt in ds.cases:
        pred, _ = to_invoice(printed_extraction(gt))
        result = compare(pred, gt.invoice)
        assert all(result.values()), (gt.doc_id, gt.render.template, [k for k, v in result.items() if not v])


def test_compare_detects_wrong_fields(small_dataset):
    ds, _ = small_dataset
    gold = ds.cases[0].invoice
    bad = gold.model_copy(update={"total": gold.total + 1, "invoice_number": "X"})
    r = compare(bad, gold)
    assert not r["total"] and not r["invoice_number"] and not r["critical_all"] and r["vendor_name"]


def test_pdf_text_has_fields(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    text = pdf_to_text(out / gt.render.file)
    assert gt.invoice.invoice_number in text and gt.invoice.vendor_tax_id in text


# ---------- extractor + eval loop with a fake API client ----------

class FakeClient:
    """Stands in for anthropic.Anthropic; answers from the labels instead of calling the API."""

    def __init__(self, by_invoice_number: dict[str, ExtractedInvoice], stop_reason: str = "end_turn"):
        self.lookup = by_invoice_number
        self.stop_reason = stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        parsed = next(v for k, v in self.lookup.items() if k in text)
        return SimpleNamespace(parsed_output=parsed, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=2000, output_tokens=500), _request_id="req_fake")


def test_extractor_request_shape_and_result(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    client = FakeClient({gt.invoice.invoice_number: printed_extraction(gt)})
    res = ClaudeExtractor(client=client).extract_text(pdf_to_text(out / gt.render.file))
    assert res.error is None and compare(res.invoice, gt.invoice)["critical_all"]
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_format"] is ExtractedInvoice
    assert call["fallbacks"] == "default" and "<document>" in call["messages"][0]["content"]
    assert res.cost_usd == pytest.approx((2000 * 4 + 500 * 20) / 1e6)


def test_extractor_reports_refusal(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    client = FakeClient({gt.invoice.invoice_number: printed_extraction(gt)}, stop_reason="refusal")
    res = ClaudeExtractor(client=client).extract_text(pdf_to_text(out / gt.render.file))
    assert res.invoice is None and "refused" in res.error


def test_eval_loop_end_to_end_and_resume(small_dataset, tmp_path):
    ds, out = small_dataset
    pdf_cases = [g for g in ds.cases if not g.render.scanned]
    # Duplicates share an invoice number with their original, so the lookup is still unambiguous.
    client = FakeClient({g.invoice.invoice_number: printed_extraction(g) for g in pdf_cases})
    extractor = ClaudeExtractor(client=client)
    run_predictions(extractor, out, tmp_path, pdf_cases, workers=2, log=lambda _: None)
    summary = score(tmp_path, pdf_cases)
    assert summary["documents"] == len(pdf_cases)
    assert summary["field_accuracy"]["critical_all"] == 100.0 and not summary["errors"]
    n_calls = len(client.calls)
    run_predictions(extractor, out, tmp_path, pdf_cases, workers=2, log=lambda _: None)
    assert len(client.calls) == n_calls  # cached: nothing re-extracted


# ---------- Ollama backend (mocked HTTP; never touches a real server) ----------

import json as _json

import httpx

from invoice_agent.extract.ollama import SCHEMA, PROMPT


def _ollama_client(answer: dict[str, ExtractedInvoice], seen: list, done_reason: str = "stop", content=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        seen.append(body)
        text = body["messages"][1]["content"]
        out = content if content is not None else next(v for k, v in answer.items() if k in text).model_dump_json()
        return httpx.Response(200, json={"message": {"role": "assistant", "content": out}, "done_reason": done_reason,
                                         "prompt_eval_count": 1500, "eval_count": 700})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_schema_is_self_contained_and_prompt_has_field_guide():
    assert "$ref" not in _json.dumps(SCHEMA) and "$defs" not in SCHEMA
    for field in ("invoice_number", "remit_to", "iban", "unit_price"):
        assert field in PROMPT


def test_ollama_extractor(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    seen = []
    ex = OllamaExtractor(client=_ollama_client({gt.invoice.invoice_number: printed_extraction(gt)}, seen))
    res = ex.extract_text(pdf_to_text(out / gt.render.file))
    assert res.error is None and compare(res.invoice, gt.invoice)["critical_all"]
    assert res.cost_usd == 0.0 and res.input_tokens == 1500
    req = seen[0]
    assert req["model"] == "qwen2.5:7b" and req["format"] == SCHEMA and req["stream"] is False
    assert req["options"]["temperature"] == 0


def test_ollama_reports_bad_output_and_truncation(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    text = pdf_to_text(out / gt.render.file)
    res = OllamaExtractor(client=_ollama_client({}, [], content="not json")).extract_text(text)
    assert res.invoice is None and "invalid JSON" in res.error
    res = OllamaExtractor(client=_ollama_client({}, [], content='{"invoice_number": "1"}')).extract_text(text)
    assert res.invoice is None and "schema validation" in res.error
    res = OllamaExtractor(client=_ollama_client({}, [], done_reason="length", content="{")).extract_text(text)
    assert "truncated" in res.error


def test_ollama_unreachable():
    res = OllamaExtractor(host="http://127.0.0.1:9").extract_text("x")
    assert res.invoice is None and "cannot reach Ollama" in res.error


# ---------- free hosted APIs (OpenAI-compatible; mocked HTTP) ----------

from invoice_agent.extract.openai_compat import OpenAICompatExtractor


def _compat_client(responses: list, seen: list):
    """Serve the queued (status, body, headers) tuples in order."""
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request, _json.loads(request.content)))
        status, body, headers = responses.pop(0)
        return httpx.Response(status, json=body, headers=headers)
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(content: str, finish="stop"):
    return (200, {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                  "usage": {"prompt_tokens": 1800, "completion_tokens": 600}}, {})


@pytest.mark.parametrize("provider,host,model,strict", [
    ("gemini", "generativelanguage.googleapis.com", "gemini-3.5-flash", False),
    ("groq", "api.groq.com", "openai/gpt-oss-120b", True),
])
def test_compat_request_shape(small_dataset, provider, host, model, strict):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    seen = []
    client = _compat_client([_ok(printed_extraction(gt).model_dump_json())], seen)
    ex = OpenAICompatExtractor(provider, api_key="test-key", rpm=0, client=client)
    res = ex.extract_text(pdf_to_text(out / gt.render.file))
    assert res.error is None and compare(res.invoice, gt.invoice)["critical_all"] and res.cost_usd == 0.0
    req, body = seen[0]
    assert req.url.host == host and req.url.path.endswith("/chat/completions")
    assert req.headers["authorization"] == "Bearer test-key"
    assert body["model"] == model and body["response_format"]["type"] == "json_schema"
    sent = body["response_format"]["json_schema"]
    assert sent.get("strict", False) is strict
    assert (sent["schema"].get("additionalProperties") is False) is strict


def test_compat_retries_rate_limits_then_succeeds(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    sleeps, seen = [], []
    client = _compat_client([(429, {"error": "slow down"}, {"retry-after": "7"}),
                             (503, {"error": "busy"}, {}),
                             _ok(printed_extraction(gt).model_dump_json())], seen)
    ex = OpenAICompatExtractor("gemini", api_key="k", rpm=0, client=client, sleep=sleeps.append)
    res = ex.extract_text(pdf_to_text(out / gt.render.file))
    assert res.error is None and len(seen) == 3 and sleeps == [7.0, 20]


def test_compat_gives_up_and_reports(small_dataset):
    client = _compat_client([(429, {"error": "quota"}, {})] * 3, [])
    ex = OpenAICompatExtractor("groq", api_key="k", rpm=0, max_retries=2, client=client, sleep=lambda _: None)
    res = ex.extract_text("x")
    assert res.invoice is None and "429" in res.error


def test_compat_strips_code_fence_and_flags_truncation(small_dataset):
    ds, out = small_dataset
    gt = next(g for g in ds.cases if not g.render.scanned)
    fenced = "```json\n" + printed_extraction(gt).model_dump_json() + "\n```"
    ex = OpenAICompatExtractor("gemini", api_key="k", rpm=0, client=_compat_client([_ok(fenced)], []))
    assert ex.extract_text(pdf_to_text(out / gt.render.file)).error is None
    ex = OpenAICompatExtractor("gemini", api_key="k", rpm=0, client=_compat_client([_ok("{", "length")], []))
    assert "truncated" in ex.extract_text("x").error


def test_compat_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # isolate from a real ./.env
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        OpenAICompatExtractor("gemini")


def test_read_key_from_env_file(tmp_path, monkeypatch):
    from invoice_agent.extract.openai_compat import read_key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY=abc\nGEMINI_API_KEY="xyz"\n', encoding="utf-8")
    assert read_key("GEMINI_API_KEY", env) == "xyz" and read_key("MISSING", env) is None
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert read_key("GEMINI_API_KEY", env) == "from-env"


def test_daily_quota_is_not_retried_and_stops_run(small_dataset, tmp_path):
    from invoice_agent.extract.openai_compat import QUOTA_EXHAUSTED
    ds, out = small_dataset
    pdf_cases = [g for g in ds.cases if not g.render.scanned][:4]
    quota = (429, [{"error": {"code": 429, "message": "per day limit",
                              "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                           "retryDelay": "23559s"}]}}], {})
    sleeps, seen = [], []
    responses = [_ok(printed_extraction(pdf_cases[0]).model_dump_json())] + [quota] * 10
    ex = OpenAICompatExtractor("gemini", api_key="k", rpm=0, client=_compat_client(responses, seen),
                               sleep=sleeps.append)
    finished = run_predictions(ex, out, tmp_path, pdf_cases, workers=1, log=lambda _: None)
    assert finished is False and sleeps == []  # no pointless 6-hour waits
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 1  # only the success was recorded; the rest resume later
    res = ex.extract_text("x")
    assert res.error.startswith(QUOTA_EXHAUSTED) and "6.5h" in res.error
    assert score(tmp_path, pdf_cases)["documents"] == 1
