import base64
import json
from datetime import date
from decimal import Decimal

import httpx
import pymupdf
import pytest

from invoice_agent.extract.base import ExtractionResult
from invoice_agent.extract.document import extract_document, key_fields, validation_issues
from invoice_agent.extract.openai_compat import OpenAICompatExtractor
from invoice_agent.pipeline import InvoicePipeline
from invoice_agent.schema import Anomaly, Decision
from invoice_agent.synth import DatasetGenerator, write_dataset


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("scans")
    ds = DatasetGenerator(seed=17, n_vendors=10, scan_ratio=0.5).generate(60)
    write_dataset(ds, out)
    return ds, out


class Scripted:
    """Returns the queued invoices (or None for a failed parse) in order; records each call."""
    model, provider = "fake", "fake"

    def __init__(self, invoices):
        self.queue = list(invoices)
        self.calls = []

    def _next(self, mode, hint):
        self.calls.append((mode, hint))
        inv = self.queue.pop(0)
        res = ExtractionResult(invoice=inv, raw=None, model="fake", provider="fake", input_tokens=100,
                               output_tokens=50, latency_s=1.0)
        if inv is None:
            res.error = "normalisation failed: unrecognised date 'Jnu 5'"
        return res

    def extract_text(self, text, hint=None):
        assert "<" not in text[:1]  # got document text, not bytes
        return self._next("text", hint)

    def extract_image(self, image, media_type="image/jpeg", hint=None):
        assert image[:2] == b"\xff\xd8" and media_type == "image/jpeg"  # a JPEG
        return self._next("image", hint)


def pick(ds, scanned, anomaly=None):
    return next(g for g in ds.cases if g.render.scanned == scanned
                and (anomaly in g.anomalies if anomaly else not g.anomalies))


def test_pdf_goes_to_text_path_with_one_read(data):
    ds, out = data
    gt = pick(ds, scanned=False)
    ex = Scripted([gt.invoice])
    res = extract_document(ex, out / gt.render.file)
    assert ex.calls == [("text", None)] and res.reads == 1 and res.input_mode == "text"


def test_scan_goes_to_vision_path(data):
    ds, out = data
    gt = pick(ds, scanned=True)
    ex = Scripted([gt.invoice])
    res = extract_document(ex, out / gt.render.file)
    assert ex.calls == [("image", None)] and res.input_mode == "image" and not res.uncertain_fields


def test_image_only_pdf_is_rasterised(data, tmp_path):
    ds, out = data
    gt = pick(ds, scanned=True)
    pdf = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(out / gt.render.file))
    doc.save(pdf)
    ex = Scripted([gt.invoice])
    assert extract_document(ex, pdf).input_mode == "image"


def test_reread_that_agrees_keeps_a_genuine_arithmetic_error(data):
    ds, out = data
    gt = next(g for g in ds.cases if Anomaly.ARITHMETIC_ERROR in g.anomalies)
    ex = Scripted([gt.invoice, gt.invoice.model_copy()])
    res = extract_document(ex, out / gt.render.file)
    assert len(ex.calls) == 2 and ex.calls[1][1] and "Do not" not in ex.calls[1][1]  # hint lists issues
    assert res.reads == 2 and not res.uncertain_fields
    assert res.invoice.arithmetic_issues()  # the printed error survives for the checks to flag
    assert res.input_tokens == 200 and res.latency_s == 2.0
    assert any("confirmed" in w for w in res.warnings)


def test_reread_that_disagrees_is_uncertain_and_keeps_first_read(data):
    ds, out = data
    gt = pick(ds, scanned=True)
    misread = gt.invoice.model_copy(update={"total": gt.invoice.total + Decimal("90.00")})  # OCR slip
    corrected = gt.invoice
    ex = Scripted([misread, corrected])
    res = extract_document(ex, out / gt.render.file)
    assert res.uncertain_fields == ["total"] and res.invoice.total == misread.total


def test_first_read_unparseable_second_valid(data):
    ds, out = data
    gt = pick(ds, scanned=True)
    ex = Scripted([None, gt.invoice])
    res = extract_document(ex, out / gt.render.file)
    assert res.invoice == gt.invoice and not res.uncertain_fields and res.reads == 2


def test_validation_and_key_fields(data):
    ds, _ = data
    inv = pick(ds, scanned=False).invoice
    assert validation_issues(ExtractionResult(invoice=inv, raw=None, model="m", provider="p")) == []
    blank = inv.model_copy(update={"vendor_tax_id": " "})
    assert "vendor tax ID is empty" in validation_issues(ExtractionResult(invoice=blank, raw=None, model="m", provider="p"))
    assert key_fields(inv) == key_fields(inv.model_copy(update={"vendor_address": blank.bill_to_address}))


def test_pipeline_holds_uncertain_extractions(data, tmp_path):
    ds, out = data
    gt = pick(ds, scanned=True)

    def extract(doc_id, path):
        return gt.invoice, None, "fake (image)", ["total"]

    pipe = InvoicePipeline(ds.vendors, ds.purchase_orders, extract, tmp_path, date(2026, 1, 1))
    r = pipe.process(gt.doc_id, out / gt.render.file)
    assert r.decision == Decision.HOLD and r.payment is None
    assert any("two reads disagreed on total" in x for x in r.reasons)


def test_gemini_image_request_shape(data):
    ds, out = data
    gt = pick(ds, scanned=True)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    ex = OpenAICompatExtractor("gemini", api_key="k", rpm=0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    image = (out / gt.render.file).read_bytes()
    ex.extract_image(image, hint="- total does not match")
    parts = seen[0]["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "untrusted" in parts[0]["text"] and "total does not match" in parts[0]["text"]
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and base64.b64decode(url.split(",", 1)[1]) == image
