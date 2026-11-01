"""Score the checks against ground-truth anomaly labels and expected decisions.

    python -m invoice_agent.checks_eval --data data/synthetic                  # on label invoices
    python -m invoice_agent.checks_eval --predictions runs/X/predictions.jsonl  # end to end

With --predictions, each document's *extracted* invoice is checked instead of the label,
so extraction errors flow through into decisions. Documents without a prediction are skipped.
Invoices are processed in document order, as an AP inbox would receive them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from invoice_agent.checks import InvoiceChecker
from invoice_agent.schema import Anomaly, Decision, GroundTruth, Invoice, PurchaseOrder, Vendor


def load(data: Path) -> tuple[list[Vendor], list[PurchaseOrder], list[GroundTruth]]:
    vendors = [Vendor.model_validate(v) for v in json.loads((data / "vendors.json").read_text(encoding="utf-8"))]
    pos = [PurchaseOrder.model_validate(p)
           for p in json.loads((data / "purchase_orders.json").read_text(encoding="utf-8"))]
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines() if line.strip()]
    cases = [GroundTruth.model_validate_json((data / r["label"]).read_text(encoding="utf-8")) for r in rows]
    return vendors, pos, cases


def load_predictions(path: Path) -> dict[str, Invoice]:
    out: dict[str, Invoice] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("invoice"):
            out[rec["doc_id"]] = Invoice.model_validate(rec["invoice"])
    return out


def load_texts(data: Path, cases: list[GroundTruth]) -> dict[str, str]:
    """PDF text layer per document, for the suspicious-instructions check."""
    from invoice_agent.extract import pdf_to_text

    return {gt.doc_id: pdf_to_text(data / gt.render.file) for gt in cases
            if gt.render.format == "pdf" and (data / gt.render.file).exists()}


def evaluate(vendors, pos, cases, predictions: dict[str, Invoice] | None = None,
             texts: dict[str, str] | None = None) -> dict:
    checker = InvoiceChecker(vendors, pos)
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion: Counter = Counter()
    mistakes = []
    n = 0
    for gt in cases:
        if predictions is not None and gt.doc_id not in predictions:
            continue
        inv = predictions[gt.doc_id] if predictions is not None else gt.invoice
        report = checker.check(inv, ref=gt.doc_id, document_text=(texts or {}).get(gt.doc_id))
        n += 1
        got, want = set(report.anomalies), set(gt.anomalies)
        for a in Anomaly:
            tp[a] += a in got and a in want
            fp[a] += a in got and a not in want
            fn[a] += a not in got and a in want
        confusion[(gt.expected_decision.value, report.decision.value)] += 1
        if got != want or report.decision != gt.expected_decision:
            mistakes.append({"doc_id": gt.doc_id, "expected": sorted(a.value for a in want),
                             "got": sorted(a.value for a in got), "decision": report.decision.value,
                             "expected_decision": gt.expected_decision.value,
                             "findings": [f.message for f in report.findings if f.severity != "info"]})
    per_anomaly = {}
    for a in Anomaly:
        p = tp[a] / (tp[a] + fp[a]) if tp[a] + fp[a] else None
        r = tp[a] / (tp[a] + fn[a]) if tp[a] + fn[a] else None
        per_anomaly[a.value] = {"precision": p, "recall": r, "support": tp[a] + fn[a]}
    correct = sum(v for (e, g), v in confusion.items() if e == g)
    # The costly error: paying an invoice that should have been held or rejected.
    false_approve = sum(v for (e, g), v in confusion.items() if g == Decision.APPROVE.value and e != g)
    should_not_pay = sum(v for (e, _), v in confusion.items() if e != Decision.APPROVE.value)
    return {
        "documents": n,
        "decision_accuracy": correct / n if n else None,
        "false_approve_rate": false_approve / should_not_pay if should_not_pay else None,
        "false_approves": false_approve,
        "per_anomaly": per_anomaly,
        "confusion": {f"{e}->{g}": v for (e, g), v in sorted(confusion.items())},
        "mistakes": mistakes,
    }


def render(summary: dict, title: str) -> str:
    f = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
    lines = [f"# Checks eval — {title}", "",
             f"Documents: {summary['documents']}  ",
             f"Decision accuracy: {f(summary['decision_accuracy'])}%  ",
             f"False-approve rate: {f(summary['false_approve_rate'])}% "
             f"({summary['false_approves']} invoices paid that should not have been)", "",
             "| Anomaly | Precision % | Recall % | Support |", "|---|---|---|---|"]
    lines += [f"| {k} | {f(v['precision'])} | {f(v['recall'])} | {v['support']} |"
              for k, v in summary["per_anomaly"].items()]
    lines += ["", "| Expected -> got | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["confusion"].items()]
    if summary["mistakes"]:
        lines += ["", "## Mistakes", ""]
        for m in summary["mistakes"]:
            lines.append(f"- {m['doc_id']}: expected {m['expected']} ({m['expected_decision']}), "
                         f"got {m['got']} ({m['decision']})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--predictions", default=None, help="predictions.jsonl from an extraction run")
    p.add_argument("--out", default=None, help="write report markdown here")
    p.add_argument("--no-text", action="store_true", help="skip reading PDF text (no instruction check)")
    a = p.parse_args(argv)
    vendors, pos, cases = load(Path(a.data))
    preds = load_predictions(Path(a.predictions)) if a.predictions else None
    texts = None if a.no_text else load_texts(Path(a.data), cases)
    summary = evaluate(vendors, pos, cases, preds, texts)
    title = f"extracted ({Path(a.predictions).parent.name})" if a.predictions else "ground-truth invoices"
    report = render(summary, title)
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
