"""Writes a generated `Dataset` to disk: documents, labels, vendor master, POs, manifest."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from invoice_agent.synth.generator import Dataset
from invoice_agent.synth.render import degrade_to_scan, render_pdf_bytes


def _dump(models) -> str:
    return json.dumps([m.model_dump(mode="json") for m in models], indent=2, ensure_ascii=False)


def write_dataset(ds: Dataset, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    docs, labels = out / "docs", out / "labels"
    docs.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    # Remove stale generated files only (never anything else in out_dir).
    for prefix in ("doc_", "adv_"):
        for p in list(docs.glob(f"{prefix}*.pdf")) + list(docs.glob(f"{prefix}*.jpg"))                 + list(labels.glob(f"{prefix}*.json")):
            p.unlink()

    (out / "vendors.json").write_text(_dump(ds.vendors), encoding="utf-8")
    (out / "purchase_orders.json").write_text(_dump(ds.purchase_orders), encoding="utf-8")

    manifest = []
    for gt in ds.cases:
        pdf = render_pdf_bytes(gt.invoice, gt.render.template, gt.render.style, gt.render.injections)
        if gt.render.scanned:
            # String seeds are hashed deterministically, so scans reproduce across runs.
            data, params = degrade_to_scan(pdf, random.Random(f"{ds.seed}:{gt.doc_id}"))
            gt.render.degradations = params
        else:
            data = pdf
        (out / gt.render.file).write_bytes(data)
        (labels / f"{gt.doc_id}.json").write_text(gt.model_dump_json(indent=2), encoding="utf-8")
        manifest.append({
            "doc_id": gt.doc_id,
            "file": gt.render.file,
            "label": f"labels/{gt.doc_id}.json",
            "template": gt.render.template,
            "scanned": gt.render.scanned,
            "currency": gt.invoice.currency,
            "anomalies": [a.value for a in gt.anomalies],
            "expected_decision": gt.expected_decision.value,
        })
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")

    stats = {
        "documents": len(ds.cases),
        "vendors": len(ds.vendors),
        "purchase_orders": len(ds.purchase_orders),
        "scanned": sum(gt.render.scanned for gt in ds.cases),
        "decisions": dict(Counter(gt.expected_decision.value for gt in ds.cases)),
        "anomalies": dict(Counter(a.value for gt in ds.cases for a in gt.anomalies)),
        "templates": dict(Counter(gt.render.template for gt in ds.cases)),
        "currencies": dict(Counter(gt.invoice.currency for gt in ds.cases)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
