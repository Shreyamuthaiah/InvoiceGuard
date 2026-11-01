"""Extraction eval: compare predicted `Invoice`s to ground-truth labels, field by field.

    python -m invoice_agent.evaluate --data data/synthetic --run runs/opus-low --limit 20

Predictions are cached in <run>/predictions.jsonl; re-running resumes and never re-bills a
document that already has a prediction. Use --report-only to rescore without API calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from invoice_agent.schema import Address, GroundTruth, Invoice

# Fields that must be right for the invoice to be payable without a human fixing it.
CRITICAL_FIELDS = [
    "invoice_number", "invoice_date", "due_date", "po_number", "vendor_name", "vendor_tax_id",
    "currency", "subtotal", "tax_amount", "total", "remit_to.account",
]


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def _addr(a: Address) -> str:
    return _norm(" ".join([a.street, a.city, a.region, a.postal_code, a.country]))


def _compact(s: Optional[str]) -> str:
    return re.sub(r"[\s\-]", "", s or "").upper()


def compare(pred: Invoice, gold: Invoice) -> dict[str, bool]:
    r: dict[str, bool] = {}
    for f in ["invoice_number", "po_number", "vendor_name", "vendor_email", "bill_to_name", "payment_terms"]:
        r[f] = _norm(getattr(pred, f)) == _norm(getattr(gold, f))
    r["vendor_tax_id"] = _compact(pred.vendor_tax_id) == _compact(gold.vendor_tax_id)
    for f in ["invoice_date", "due_date", "subtotal", "tax_rate", "tax_amount", "total"]:
        r[f] = getattr(pred, f) == getattr(gold, f)
    r["currency"] = pred.currency.upper() == gold.currency.upper()
    r["vendor_address"] = _addr(pred.vendor_address) == _addr(gold.vendor_address)
    r["bill_to_address"] = _addr(pred.bill_to_address) == _addr(gold.bill_to_address)
    pb, gb = pred.remit_to, gold.remit_to
    r["remit_to.bank_name"] = _norm(pb.bank_name) == _norm(gb.bank_name)
    r["remit_to.account"] = all(_compact(getattr(pb, k)) == _compact(getattr(gb, k))
                                for k in ("account_number", "routing_number", "iban"))
    r["remit_to.swift"] = _compact(pb.swift) == _compact(gb.swift)
    r["line_items.count"] = len(pred.line_items) == len(gold.line_items)
    r["line_items.exact"] = r["line_items.count"] and all(
        _norm(p.description) == _norm(g.description) and p.quantity == g.quantity
        and p.unit_price == g.unit_price and p.amount == g.amount
        for p, g in zip(pred.line_items, gold.line_items)
    )
    r["critical_all"] = all(r[f] for f in CRITICAL_FIELDS)
    return r


# ---------- running ----------

def _load_cases(data: Path, include_scans: bool) -> list[GroundTruth]:
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines() if line.strip()]
    cases = []
    for row in rows:
        if row["scanned"] and not include_scans:
            continue
        cases.append(GroundTruth.model_validate_json((data / row["label"]).read_text(encoding="utf-8")))
    return cases


def _predict_one(extractor, data: Path, gt: GroundTruth) -> dict:
    from invoice_agent.extract import pdf_to_text

    try:
        res = extractor.extract_text(pdf_to_text(data / gt.render.file))
    except Exception as e:  # keep the run going; record the failure
        return {"doc_id": gt.doc_id, "error": f"{type(e).__name__}: {e}"}
    return {
        "doc_id": gt.doc_id,
        "invoice": res.invoice.model_dump(mode="json") if res.invoice else None,
        "raw": res.raw.model_dump(mode="json") if res.raw else None,
        "warnings": res.warnings,
        "error": res.error,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cost_usd": res.cost_usd,
        "latency_s": round(res.latency_s, 2),
        "model": res.model,
        "request_id": res.request_id,
    }


def run_predictions(extractor, data: Path, run_dir: Path, cases: list[GroundTruth], workers: int,
                    log: Callable[[str], None] = print) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("invoice") is not None or rec.get("raw") is not None:
                done.add(rec["doc_id"])  # failed calls (no output) are retried
    todo = [gt for gt in cases if gt.doc_id not in done]
    log(f"{len(done)} cached, {len(todo)} to extract")
    with pred_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_predict_one, extractor, data, gt): gt for gt in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if _rate_limited(rec):
                # Daily quota: stop now. Nothing is recorded, so a later run picks these up.
                for other in futures:
                    other.cancel()
                log(f"STOPPED: {rec['error']}. Re-run the same command later to resume.")
                return False
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = "ERR " + rec["error"] if rec.get("error") else "ok"
            log(f"[{i}/{len(todo)}] {rec['doc_id']} {status}")
    return True


def _rate_limited(rec: dict) -> bool:
    """Quota / rate-limit failures say nothing about extraction quality: not scored, retried later."""
    err = rec.get("error") or ""
    return rec.get("invoice") is None and (err.startswith("quota exhausted") or " error 429" in err)


def score(run_dir: Path, cases: list[GroundTruth]) -> dict:
    latest: dict[str, dict] = {}
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if _rate_limited(rec):
            continue  # not attempted, as far as quality is concerned
        latest[rec["doc_id"]] = rec  # last write wins
    field_hits: dict[str, list[bool]] = defaultdict(list)
    by_template: dict[str, list[bool]] = defaultdict(list)
    by_currency: dict[str, list[bool]] = defaultdict(list)
    failures, cost, latency, tokens_in, tokens_out = [], 0.0, [], 0, 0
    errors = []
    for gt in cases:
        rec = latest.get(gt.doc_id)
        if rec is None:
            continue
        cost += rec.get("cost_usd") or 0
        tokens_in += rec.get("input_tokens") or 0
        tokens_out += rec.get("output_tokens") or 0
        if rec.get("latency_s"):
            latency.append(rec["latency_s"])
        if not rec.get("invoice"):
            errors.append({"doc_id": gt.doc_id, "error": rec.get("error")})
            result = {k: False for k in compare(gt.invoice, gt.invoice)}  # every field counts as wrong
        else:
            result = compare(Invoice.model_validate(rec["invoice"]), gt.invoice)
            wrong = [k for k, ok in result.items() if not ok and k != "critical_all"]
            if wrong:
                failures.append({"doc_id": gt.doc_id, "template": gt.render.template, "wrong": wrong})
        for k, ok in result.items():
            field_hits[k].append(ok)
        by_template[gt.render.template].append(result.get("critical_all", False))
        by_currency[gt.invoice.currency].append(result.get("critical_all", False))
    n = len([gt for gt in cases if gt.doc_id in latest])
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    latency.sort()
    return {
        "documents": n,
        "errors": errors,
        "field_accuracy": {k: pct(v) for k, v in sorted(field_hits.items())},
        "critical_all_by_template": {k: pct(v) for k, v in sorted(by_template.items())},
        "critical_all_by_currency": {k: pct(v) for k, v in sorted(by_currency.items())},
        "cost_usd_total": round(cost, 4),
        "cost_usd_per_doc": round(cost / n, 4) if n else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_p50_s": latency[len(latency) // 2] if latency else None,
        "latency_p95_s": latency[int(len(latency) * 0.95) - 1] if latency else None,
        "failures": failures,
    }


def render_report(summary: dict, title: str) -> str:
    lines = [f"# Extraction eval — {title}", "", f"Documents scored: {summary['documents']}  ",
             f"Errors (no output): {len(summary['errors'])}  ",
             f"Cost: ${summary['cost_usd_total']} total, ${summary['cost_usd_per_doc']}/doc  ",
             f"Latency p50 / p95: {summary['latency_p50_s']}s / {summary['latency_p95_s']}s", "",
             "| Field | Accuracy % |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["field_accuracy"].items()]
    lines += ["", "| Template | All critical fields correct % |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["critical_all_by_template"].items()]
    lines += ["", "| Currency | All critical fields correct % |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["critical_all_by_currency"].items()]
    if summary["failures"]:
        lines += ["", "## Documents with wrong fields", ""]
        lines += [f"- {f['doc_id']} ({f['template']}): {', '.join(f['wrong'])}" for f in summary["failures"]]
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e['doc_id']}: {e['error']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--run", required=True, help="run directory for cached predictions + report")
    p.add_argument("--limit", type=int, default=None, help="only the first N documents")
    p.add_argument("--provider", choices=["gemini", "groq", "ollama", "claude"], default="gemini")
    p.add_argument("--model", default=None, help="default depends on provider (see extract/)")
    p.add_argument("--effort", default=None, help="claude only")
    p.add_argument("--workers", type=int, default=None, help="default: 1 for ollama, 2 for free APIs, 4 for claude")
    p.add_argument("--report-only", action="store_true", help="rescore cached predictions; no API calls")
    a = p.parse_args(argv)

    data, run_dir = Path(a.data), Path(a.run)
    cases = _load_cases(data, include_scans=False)  # week 2: digital PDFs only
    if a.limit:
        cases = cases[: a.limit]
    if not a.report_only:
        from invoice_agent.extract import make_extractor

        kwargs = {"effort": a.effort} if (a.provider == "claude" and a.effort) else {}
        extractor = make_extractor(a.provider, a.model, **kwargs)
        config = {"provider": a.provider, "model": extractor.model, "effort": getattr(extractor, "effort", None)}
        cfg = run_dir / "config.json"
        if cfg.exists():
            prev = json.loads(cfg.read_text())
            if prev != config:
                sys.exit(f"{run_dir} was run with {prev}; use a new --run directory for a different config")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(config))
        workers = a.workers or {"ollama": 1, "claude": 4}.get(a.provider, 2)
        run_predictions(extractor, data, run_dir, cases, workers, log=lambda m: print(m, flush=True))

    summary = score(run_dir, cases)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = render_report(summary, run_dir.name)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
