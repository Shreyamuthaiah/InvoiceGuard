"""Route a document to the text or vision path, then self-check the result.

Self-check policy (conservative on purpose):
- A read "fails validation" if normalisation failed, the arithmetic doesn't add up, or a
  field needed for matching/payment is empty.
- Then the model re-reads, told which fields failed and to report what is printed.
- If both reads agree on every key field, the values are kept: a real invoice can have bad
  arithmetic, and the checks must see it.
- If they disagree, the invoice is marked uncertain and the pipeline holds it for a human.
  The second read never silently replaces the first, because that would let a model
  "correct" a wrong total and approve a bad invoice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from invoice_agent.extract.base import ExtractionResult, Extractor
from invoice_agent.extract.pdf_text import has_text_layer, pdf_to_text
from invoice_agent.schema import Invoice

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def key_fields(inv: Invoice) -> dict[str, object]:
    """The values that decide matching and payment; two reads must agree on all of them."""
    b = inv.remit_to
    return {
        "invoice_number": inv.invoice_number.strip().upper(),
        "invoice_date": inv.invoice_date, "due_date": inv.due_date,
        "po_number": (inv.po_number or "").strip().upper(),
        "vendor_tax_id": "".join(ch for ch in inv.vendor_tax_id if ch.isalnum()).upper(),
        "currency": inv.currency, "subtotal": inv.subtotal, "tax_amount": inv.tax_amount, "total": inv.total,
        "line_amounts": tuple(li.amount for li in inv.line_items),
        "remit_account": (b.account_number, b.routing_number, (b.iban or "").replace(" ", "")),
    }


def validation_issues(res: ExtractionResult) -> list[str]:
    if res.invoice is None:
        return [res.error] if res.error and res.error.startswith("normalisation failed") else []
    inv = res.invoice
    issues = list(inv.arithmetic_issues())
    for name, value in [("invoice number", inv.invoice_number), ("vendor tax ID", inv.vendor_tax_id),
                        ("vendor name", inv.vendor_name)]:
        if not value.strip():
            issues.append(f"{name} is empty")
    b = inv.remit_to
    if not (b.iban or b.account_number):
        issues.append("remit-to bank account is empty")
    return issues


def _read(extractor: Extractor, path: Path, hint: Optional[str]) -> ExtractionResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf" and has_text_layer(path):
        res = extractor.extract_text(pdf_to_text(path), hint=hint)
        res.input_mode = "text"
        return res
    if not hasattr(extractor, "extract_image"):
        raise RuntimeError(f"{type(extractor).__name__} has no vision support for {path.name}")
    if suffix == ".pdf":  # image-only PDF: rasterise page 1
        import pymupdf
        with pymupdf.open(path) as doc:
            data, media_type = doc[0].get_pixmap(dpi=150).tobytes("jpeg"), "image/jpeg"
    elif suffix in IMAGE_TYPES:
        data, media_type = path.read_bytes(), IMAGE_TYPES[suffix]
    else:
        raise RuntimeError(f"unsupported document type {suffix}")
    res = extractor.extract_image(data, media_type, hint=hint)
    res.input_mode = "image"
    return res


def extract_document(extractor: Extractor, path: str | Path, verify: bool = True) -> ExtractionResult:
    path = Path(path)
    first = _read(extractor, path, None)
    issues = validation_issues(first) if verify else []
    if not issues:
        return first

    second = _read(extractor, path, "\n".join(f"- {i}" for i in issues))
    second.reads = 2  # cost and latency cover both reads
    second.input_tokens += first.input_tokens
    second.output_tokens += first.output_tokens
    second.latency_s += first.latency_s

    if first.invoice is None:
        # First read unusable; the second is all we have. Trust it only if it validates.
        if second.invoice is not None and validation_issues(second):
            second.uncertain_fields = ["single usable read that fails validation"]
        second.warnings.append(f"first read failed: {first.error}")
        return second
    if second.invoice is None:
        first.reads = 2
        first.input_tokens, first.output_tokens, first.latency_s = (
            second.input_tokens, second.output_tokens, second.latency_s)
        first.uncertain_fields = ["re-read failed"]
        first.warnings.append(f"re-read failed: {second.error}")
        return first

    a, b = key_fields(first.invoice), key_fields(second.invoice)
    disagree = [k for k in a if a[k] != b[k]]
    second.warnings.append("re-read after validation issues: " + "; ".join(issues))
    if disagree:
        second.uncertain_fields = disagree
        second.warnings.append("reads disagree on: " + ", ".join(disagree))
        second.invoice = first.invoice  # keep the first read; the invoice is held either way
    else:
        second.warnings.append("re-read confirmed the printed values")
    return second
