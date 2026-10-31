# Invoice-to-Payment Agent — Build Plan

Goal: an agent that takes a vendor invoice (digital PDF or scanned image), extracts it into a
typed schema, validates it against the vendor master and purchase orders, flags fraud/duplicate
risk, and either schedules payment, holds it for a human, or rejects it — with an audit trail and
a measurable eval on a labelled dataset.

Every week ends with something runnable plus tests, and the eval numbers are tracked in
`EVAL.md` from week 2 onward so progress is visible.

---

## Week 1 — Foundation & synthetic dataset  *(current)*

- [x] Python 3.10 venv: pymupdf, pydantic, faker, reportlab, pillow, pytest
- [x] `git init`, `requirements.txt`, `pyproject.toml`
- [x] Pydantic domain schema: `Vendor`, `BankAccount`, `PurchaseOrder`, `Invoice`, `LineItem`,
      `GroundTruth`, `Anomaly`, `Decision`
- [x] Synthetic generator: vendor master, POs, invoices across 3 layouts, 3 currencies,
      US/EU number + date formats, ~30% "scanned" (rasterised, rotated, noisy JPEG)
- [x] Ground-truth labels per document: printed field values (extraction target), true
      vendor id, injected anomalies, expected decision
- [x] Tests: schema validity, arithmetic consistency, PDF text round-trip, determinism

**Exit:** `python -m invoice_agent.synth --n 300 --seed 42` produces a reproducible dataset.

## Week 2 — Extraction v1 (digital PDFs)

- PyMuPDF text + layout extraction → LLM structured output into the `Invoice` schema
  (default: free local model via Ollama, `qwen2.5:7b`; Claude backend optional)
- Normalisers: dates (US vs EU ambiguity), money (`1.234,56 €`), quantities
- Eval harness: per-field exact-match / tolerance accuracy, per-template breakdown
- Baseline numbers into `EVAL.md`

## Week 3 — Extraction v2 (scanned docs) + self-checks

- Image path for scanned docs (vision model), compare against text path on PDFs
- Post-extraction validation: arithmetic checks, required fields, low-confidence fields
- One targeted re-extraction pass when validation fails (no blind retries)
- Track cost + latency per document

## Week 4 — Validation & matching tools

- Vendor resolution: exact tax-id match → fuzzy name/address match → "unknown vendor"
- PO matching (2-way: invoice vs PO, configurable tolerance)
- Duplicate detection (vendor + invoice number, plus near-duplicate amount/date)
- Bank-detail change detection against vendor master (top fraud signal)
- All checks are deterministic Python; each returns structured findings

## Week 5 — Agent orchestration

- Tool-use loop: `extract_invoice`, `lookup_vendor`, `match_po`, `check_duplicates`,
  `check_bank_details`, `propose_decision`
- Decision policy: approve / hold (with reasons) / reject; the LLM explains, rules decide
- Human-review queue for holds; structured audit log per document
- Prompt-injection defence: invoice text is data, never instructions (add adversarial docs)

## Week 6 — Payment step

- Mock payments API (schedule by due date, idempotency keys, approval thresholds)
- Payment only after all checks pass, never to unverified bank details
- End-to-end run: folder of invoices → decisions → scheduled payments + audit log

## Week 7 — End-to-end eval & hardening

- Decision accuracy, per-anomaly recall, false-approve rate (the metric that matters)
- Adversarial set: injected instructions, look-alike vendors, split invoices
- Error analysis → fixes → re-run; cost/latency budget per invoice

## Week 8 — Demo & write-up

- Small UI (upload → extracted fields → findings → decision) or CLI demo + recording
- README with architecture diagram, eval table, design trade-offs
- Portfolio write-up / blog post
