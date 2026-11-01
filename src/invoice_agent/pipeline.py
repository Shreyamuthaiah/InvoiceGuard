"""End-to-end invoice processing: extract -> check -> decide -> pay, with an audit log.

    # live extraction with a free API (uses quota)
    python -m invoice_agent.pipeline --out runs/pipeline --provider gemini --limit 10
    # reuse cached extractions from an eval run (no API calls)
    python -m invoice_agent.pipeline --out runs/pipeline --from-predictions runs/X/predictions.jsonl
    # demo without any model: use the labels as a perfect extractor
    python -m invoice_agent.pipeline --out runs/pipeline-demo --from-labels

Outputs in --out: audit.jsonl (one record per invoice), review_queue.jsonl (holds and
rejects with reasons), payments.json (payment store), summary.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

from invoice_agent.checks import CheckReport, InvoiceChecker
from invoice_agent.payments import MockPaymentsAPI, Payment, PaymentError, PaymentPolicy
from invoice_agent.schema import Decision, Invoice, PurchaseOrder, Vendor

# (doc_id, path) -> (invoice or None, error or None, extractor label)
ExtractFn = Callable[[str, Path], tuple[Optional[Invoice], Optional[str], str]]


@dataclass
class ProcessResult:
    doc_id: str
    decision: Decision
    reasons: list[str]
    invoice: Optional[Invoice] = None
    report: Optional[CheckReport] = None
    payment: Optional[Payment] = None
    payment_created: bool = False
    extraction_error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _text(path: Path) -> Optional[str]:
    if path.suffix.lower() != ".pdf" or not path.exists():
        return None
    from invoice_agent.extract import pdf_to_text

    return pdf_to_text(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InvoicePipeline:
    def __init__(self, vendors: list[Vendor], purchase_orders: list[PurchaseOrder], extract: ExtractFn,
                 out_dir: Path, today: date, policy: PaymentPolicy | None = None) -> None:
        self.extract = extract
        self.checker = InvoiceChecker(vendors, purchase_orders)
        self.payments = MockPaymentsAPI(out_dir / "payments.json", policy)
        self.out_dir = out_dir
        self.today = today
        out_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = out_dir / "audit.jsonl"
        self.queue_path = out_dir / "review_queue.jsonl"

    def process(self, doc_id: str, path: Path) -> ProcessResult:
        inv, error, extractor = self.extract(doc_id, path)
        if inv is None:
            result = ProcessResult(doc_id, Decision.HOLD, [f"extraction failed: {error}; enter manually"],
                                   extraction_error=error)
        else:
            report = self.checker.check(inv, ref=doc_id, document_text=_text(path))
            reasons = [f.message for f in report.findings if f.anomaly]
            result = ProcessResult(doc_id, report.decision, reasons, invoice=inv, report=report,
                                   warnings=[f.message for f in report.findings if f.severity == "warning"
                                             and not f.anomaly])
            if report.decision == Decision.APPROVE:
                try:
                    result.payment, result.payment_created = self.payments.schedule(
                        inv, report.vendor_match.vendor, report.decision, self.today, source_ref=doc_id)
                except PaymentError as e:  # a payment control tripped: never pay, send to a human
                    result.decision = Decision.HOLD
                    result.reasons.append(f"payment blocked: {e}")
        self._audit(result, path, extractor)
        return result

    def _audit(self, r: ProcessResult, path: Path, extractor: str) -> None:
        inv = r.invoice
        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "doc_id": r.doc_id,
            "file": str(path),
            "file_sha256": _sha256(path) if path.exists() else None,
            "extractor": extractor,
            "extraction_error": r.extraction_error,
            "vendor": inv.vendor_name if inv else None,
            "vendor_id": r.report.vendor_match.vendor.vendor_id if r.report and r.report.vendor_match.vendor else None,
            "vendor_match": r.report.vendor_match.method if r.report else None,
            "invoice_number": inv.invoice_number if inv else None,
            "total": f"{inv.total:.2f}" if inv else None,
            "currency": inv.currency if inv else None,
            "invoice": inv.model_dump(mode="json") if inv else None,
            "findings": [f.model_dump(mode="json") for f in r.report.findings] if r.report else [],
            "decision": r.decision.value,
            "reasons": r.reasons,
            "payment_id": r.payment.payment_id if r.payment else None,
            "payment_status": r.payment.status if r.payment else None,
            "payment_reused": bool(r.payment) and not r.payment_created,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if r.decision != Decision.APPROVE or (r.payment and r.payment.status == "awaiting_approval"):
            item = {"doc_id": r.doc_id, "decision": r.decision.value, "reasons": r.reasons,
                    "warnings": r.warnings, "vendor": record["vendor"], "invoice_number": record["invoice_number"],
                    "total": record["total"], "currency": record["currency"],
                    "awaiting_approval": r.payment.payment_id if r.payment and r.payment.status == "awaiting_approval" else None}
            with self.queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")


def summarize(results: list[ProcessResult], payments: MockPaymentsAPI, today: date) -> str:
    decisions = Counter(r.decision.value for r in results)
    sched, waiting = defaultdict(Decimal), defaultdict(Decimal)
    late = 0
    for r in results:
        if r.payment and r.payment_created:
            (waiting if r.payment.status == "awaiting_approval" else sched)[r.payment.currency] += r.payment.amount
            late += r.payment.late
    reused = sum(1 for r in results if r.payment and not r.payment_created)
    reasons = Counter(f.anomaly.value for r in results if r.report for f in r.report.findings if f.anomaly)
    fmt = lambda d: ", ".join(f"{v:,.2f} {k}" for k, v in sorted(d.items())) or "none"  # noqa: E731
    lines = [f"# Pipeline run ({today.isoformat()})", "",
             f"Invoices processed: {len(results)}  ",
             f"Approved: {decisions['approve']} · Held: {decisions['hold']} · Rejected: {decisions['reject']}  ",
             f"Extraction failures (held for manual entry): {sum(1 for r in results if r.extraction_error)}", "",
             f"Scheduled payments: {fmt(sched)}  ",
             f"Awaiting human approval (over threshold): {fmt(waiting)}  ",
             f"Paid late (due date already passed): {late}  ",
             f"Already scheduled earlier (not paid again): {reused}", "",
             "| Hold/reject reason | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    return "\n".join(lines) + "\n"


# ---------- extraction sources ----------

def labels_extractor(data: Path) -> ExtractFn:
    from invoice_agent.schema import GroundTruth

    def fn(doc_id: str, path: Path):
        gt = GroundTruth.model_validate_json((data / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))
        return gt.invoice, None, "labels"
    return fn


def predictions_extractor(path: Path) -> ExtractFn:
    preds: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        preds[rec["doc_id"]] = rec
    cfg_path = path.parent / "config.json"
    label = json.loads(cfg_path.read_text())["model"] if cfg_path.exists() else "cached"

    def fn(doc_id: str, _path: Path):
        rec = preds.get(doc_id)
        if rec is None:
            return None, "no cached prediction", label
        if not rec.get("invoice"):
            return None, rec.get("error") or "no output", label
        return Invoice.model_validate(rec["invoice"]), None, f"{label} (cached)"
    return fn


def live_extractor(provider: str, model: Optional[str]) -> ExtractFn:
    from invoice_agent.extract import make_extractor, pdf_to_text

    extractor = make_extractor(provider, model)

    def fn(doc_id: str, path: Path):
        res = extractor.extract_text(pdf_to_text(path))
        return res.invoice, res.error, f"{provider}:{extractor.model}"
    return fn


def main(argv: list[str] | None = None) -> None:
    from invoice_agent.checks_eval import load

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic", help="vendors.json, purchase_orders.json, docs/")
    p.add_argument("--out", required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--from-predictions", default=None)
    src.add_argument("--from-labels", action="store_true")
    p.add_argument("--provider", default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--today", default=None, help="YYYY-MM-DD; default: today")
    a = p.parse_args(argv)

    data, out = Path(a.data), Path(a.out)
    today = date.fromisoformat(a.today) if a.today else date.today()
    vendors, pos, cases = load(data)
    docs = [(gt.doc_id, data / gt.render.file) for gt in cases if not gt.render.scanned]  # scans: week 3
    if a.limit:
        docs = docs[: a.limit]
    if a.from_labels:
        extract = labels_extractor(data)
    elif a.from_predictions:
        extract = predictions_extractor(Path(a.from_predictions))
    else:
        extract = live_extractor(a.provider, a.model)

    pipe = InvoicePipeline(vendors, pos, extract, out, today)
    results = []
    for i, (doc_id, path) in enumerate(docs, 1):
        r = pipe.process(doc_id, path)
        results.append(r)
        pay = f" -> {r.payment.payment_id} ({r.payment.status})" if r.payment else ""
        print(f"[{i}/{len(docs)}] {doc_id} {r.decision.value}{pay}", flush=True)
    report = summarize(results, pipe.payments, today)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
