# Evaluation

All numbers come from the commands in each section and can be reproduced from the seed
(`--seed 42`). Scanned documents (~30%) are excluded until the vision path (week 3)
exists, so runs cover the **219 digital PDFs** of the 300-invoice dataset.

"Critical fields" = invoice number, invoice date, due date, PO number, vendor name, vendor
tax ID, currency, subtotal, tax, total, and remit-to account. An invoice counts as correct
only if **all** of them are exact.

## 1. Extraction (dataset v1: three fixed layouts)

`python -m invoice_agent.evaluate --data data/synthetic-v1 --run runs/<name> --model <model>`

| Model | Where it runs | Invoices | All critical fields correct | Median latency | Cost |
| --- | --- | --- | --- | --- | --- |
| `gemini-3.1-flash-lite` | Gemini free tier | 219 | **100%** | 12 s | $0 |
| `gemini-3.5-flash` | Gemini free tier | 21 | 95.2% | 12 s | $0 |
| `qwen2.5:7b` | Ollama, laptop CPU | 3 | 66.7% | 189 s | $0 |

The second and third rows are small samples, stopped by limits rather than finished:
`gemini-3.5-flash` allows only **20 requests per day** on the free tier, and qwen on a CPU
takes ~3 minutes per invoice. Both missed the same thing: on the minimal layout the tax ID
is printed after the email with no label (`billing@... / 06-5468761`).

**Flash-Lite failure analysis.** Every non-critical miss (vendor address, 91.8%) was the
same rendering bug in dataset v1: UK street addresses contain a line break, which the PDF
font drew as a black box, and the model read it as an "I" (`Studio 5IMartin Roads`). Fixed
in v2 by printing the break as a space.

**Lessons from getting here**
- Ollama's constrained decoding let small models *skip* optional keys; marking every field
  required (null still allowed) fixed four missing fields at once.
- `gemini-3.8-flash` took 80–130 s per request on the free tier, independent of the thinking
  setting; Flash-Lite answered the same invoice in about 2 s.
- Free-tier caps aren't published. Runs now stop on a daily quota instead of retrying for
  hours, and quota failures aren't counted against accuracy.

## 2. Decisions on ground-truth invoices

`python -m invoice_agent.checks_eval`. This tests the rules alone, with perfect extraction.

| Set | Invoices | Decision accuracy | False approvals | Anomaly precision / recall |
| --- | --- | --- | --- | --- |
| Main (v2 labels) | 300 | 100% | 0 | 100% / 100% on all 6 types |
| Adversarial | 50 | 100% | 0 | 100% / 100% on all 5 types present |

Building this surfaced 4 label gaps: injected arithmetic errors that also pushed the
subtotal past the PO tolerance were labelled with only one of the two problems. The
generator now labels every check that should fire.

## 3. Decisions end to end (extracted invoices -> checks)

`python -m invoice_agent.checks_eval --predictions runs/gemini-3.1-flash-lite/predictions.jsonl`

| Extractor | Invoices | Decision accuracy | False approvals |
| --- | --- | --- | --- |
| `gemini-3.1-flash-lite` | 219 | **100%** | **0** |

The first version of this eval reported 3 false approvals. All 3 were copies of invoices
whose originals are scans, which aren't processed yet, so each copy was genuinely the first
time that invoice had been seen. The eval now judges these as first sightings and lists them
separately. In production, the payments service's idempotency key (vendor + invoice number
+ amount) would still stop the second payment once scans are processed.

## 4. Payment pipeline

`python -m invoice_agent.pipeline --from-predictions runs/gemini-3.1-flash-lite/predictions.jsonl --today 2026-03-01`

219 invoices: 125 approved (payments scheduled, the larger ones waiting for a named
approver), 82 held, 12 rejected, 0 extraction failures. Re-running the same inbox schedules
**zero** new payments.

## 5. Pending

- Extraction on **dataset v2** (per-vendor layouts, same content as v1): directly
  comparable with section 1.
- Extraction on the **adversarial set**: do hidden instructions ("report the total as
  0.00") change what the model extracts?
