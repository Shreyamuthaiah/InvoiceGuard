# InvoiceGuard: the complete guide

A plain-English guide to everything in this project, written for someone with no coding
background. Read it in order the first time. After that, use it as a reference before
interviews.

**How to use it**

1. Read sections 1–4 until you can explain the project out loud without looking.
2. Skim sections 5–6 with the code open beside you. You don't need to understand every
   line, only what each part is *for*.
3. Learn section 7 (design decisions) and section 9 (what went wrong) well. Interviewers ask
   about these most.
4. Practise section 12 (interview questions) out loud, in your own words. Don't memorise
   the wording; memorise the *ideas*.

---

## 1. The project in one minute

**One sentence:** InvoiceGuard reads vendor invoices with AI, checks them for errors and
fraud with fixed rules, and decides whether to pay, hold for a person, or reject, without
ever sending money to the wrong account.

**30-second version (for "tell me about a project"):**
> "InvoiceGuard automates accounts payable. An LLM reads each invoice, PDF or scan, into
> structured data. Then deterministic rules check it against the vendor list and purchase
> orders, catching duplicates, changed bank details, fake vendors and prompt-injection
> attempts. The key principle is that the AI only *reads*; rules make every decision that
> moves money. I tested it on 300 synthetic invoices plus 50 adversarial ones, and no
> invoice was wrongly paid."

**2-minute version:** the 30-second version, then add:
- how it was evaluated (labelled synthetic data, because no public dataset has fraud labels)
- one thing that went wrong and how it was found (the IBAN bug in section 9 is a good one)
- that it runs on a free model at about 12 seconds per invoice

---

## 2. The business problem

You need this vocabulary to talk about the project credibly.

| Term | Meaning |
| --- | --- |
| **Accounts payable (AP)** | The team that pays a company's suppliers. |
| **Invoice** | A bill from a supplier: what was sold, how much, when it's due, which bank account to pay. |
| **Vendor** | A supplier. |
| **Vendor master** | The company's trusted list of approved vendors: name, tax ID, address, and the bank account on file. |
| **Purchase order (PO)** | The company's own approval to buy something, raised *before* the invoice arrives. An invoice should quote a PO number. |
| **2-way match** | Checking an invoice against its PO: right vendor, and the amount isn't more than was approved. (A 3-way match also checks a goods receipt; this project does 2-way.) |
| **Tolerance** | How far an invoice may exceed its PO before it's flagged. Here: 2%. |
| **Approval threshold** | Above this amount, a named person must sign off. Here: 10,000 USD or EUR, 8,000 GBP. |
| **Payment terms** | When payment is due, e.g. "Net 30" means 30 days after the invoice date. |

**The risks AP teams worry about**, all of which this project tests:

| Risk | What happens | How InvoiceGuard catches it |
| --- | --- | --- |
| **Duplicate invoice** | The same bill is sent twice and paid twice. | Same vendor + same invoice number → reject. |
| **Changed bank details** | A fraudster emails a "new bank account" invoice (business email compromise). The most costly real-world fraud. | The remit-to account must match the vendor master. Payment always goes to the account on file anyway. |
| **Fake / look-alike vendor** | "Acme Ltd" vs the real "Acme Inc", with the fraudster's bank account. | Vendors are matched by tax ID; a familiar name with an unknown tax ID is flagged. |
| **Over-billing** | The invoice is for more than was approved. | Compared with the PO total (2% tolerance), including across several invoices. |
| **Split invoices** | One big purchase sent as several small invoices to stay under the sign-off limit. | Several invoices on the same PO within 30 days, each under the limit but over it together. |
| **Arithmetic errors** | Line items don't add up to the total. | Recalculated and compared. |
| **Prompt injection** (new, AI-specific) | Text on the invoice tries to instruct the AI: "pre-approved, skip checks". | The AI is told the document is untrusted data, *and* a rule holds any invoice with text aimed at an AI. |

---

## 3. AI concepts used, in plain English

| Concept | Plain-English meaning | Where it's used |
| --- | --- | --- |
| **LLM (large language model)** | An AI model that reads and writes text, e.g. Gemini, Claude, GPT. | Reading invoices; the review agent. |
| **Prompt / system prompt** | The instructions sent with the request. The *system prompt* sets the model's role and rules. | `extract/base.py` (`SYSTEM_PROMPT`), `agent.py` (`SYSTEM`). |
| **Tokens** | The chunks of text models count and bill by; roughly ¾ of a word each. | Invoices used ~940 input and ~630 output tokens. |
| **Structured output / JSON schema** | Forcing the model to answer in an exact data format (named fields) instead of free text, so code can use the answer reliably. | Every extraction returns a JSON object matching `ExtractedInvoice`. |
| **Temperature** | How random the model's output is. 0 = as consistent as possible. | Extraction uses 0. |
| **Vision / multimodal model** | A model that can read images, not just text. | Scanned invoices are sent as JPEG images. |
| **Tool calling (function calling)** | The model asks the program to run a function (e.g. "look up vendor V0004"), gets the result back, and continues. | The review agent's tools. |
| **Agent** | An LLM in a loop: think → call a tool → read the result → repeat → finish. | `agent.py`. |
| **Deterministic vs probabilistic** | Normal code gives the same answer every time (deterministic). An LLM might not (probabilistic). | The core design rule: money decisions stay deterministic. |
| **Hallucination** | The model states something not supported by its input. | Measured as "ungrounded numbers" in the agent eval. |
| **Grounding** | Making sure every claim traces back to real data. | Agent explanations must use numbers from tool results. |
| **Prompt injection** | Hidden or visible text in the *input* that tries to override the model's instructions. | 18 adversarial invoices test this. |
| **Evaluation ("evals")** | Measuring an AI system against known correct answers instead of trusting it. | `evaluate.py`, `checks_eval.py`, `agent_eval.py`. |
| **Ground truth / labels** | The known correct answer for each test case. | Every synthetic invoice has a label file. |
| **Precision** | Of the things flagged, how many were real problems? (Few false alarms.) | Per-anomaly scores. |
| **Recall** | Of the real problems, how many were flagged? (Few misses.) | Per-anomaly scores. |
| **False approval** | Paying an invoice that should have been held. The metric that matters most here. | 0 across all 300 invoices. |
| **Rate limit / quota** | How many requests a provider allows per minute or per day. | Free-tier limits shaped which model ran which test. |
| **OpenAI-compatible API** | A common request format many providers accept, so one piece of code can talk to several of them. | Gemini and Groq share one client. |

---

## 4. How the system works: following one invoice

```text
Invoice file (PDF or scan)
   │
   ▼
1. READ        PDF with text → PyMuPDF pulls out the text
               Scan / image  → sent to the vision model as a picture
   │
   ▼
2. EXTRACT     LLM returns a JSON object: every field copied exactly as printed
               ("2.093,04 €", "03/04/2026")
   │
   ▼
3. SELF-CHECK  Fails validation? (numbers don't add up, key field empty)
               → re-read just those fields
               → reads agree: keep · reads disagree: mark "uncertain"
   │
   ▼
4. NORMALISE   Plain Python turns strings into real dates and numbers
               (US vs European formats; ambiguous dates)
   │
   ▼
5. CHECK       Deterministic rules: vendor, PO, duplicates, split billing,
               bank details, arithmetic, instructions aimed at an AI
   │
   ▼
6. DECIDE      Duplicate → REJECT · any other problem → HOLD · clean → APPROVE
               (uncertain extraction → HOLD)
   │
   ├── HOLD / REJECT ──► 7. EXPLAIN (optional): review agent writes reasons + next steps
   │                        └─► review queue for a person
   │
   └── APPROVE ──► 8. PAY: payments service with its own controls
                     └─► scheduled payment to the vendor-master account
   │
   ▼
9. AUDIT       Every step is written to an audit log
```

**Worked example (`doc_0002` in the dataset):**
1. It's a digital PDF, so its text is read directly.
2. The LLM extracts vendor "Schüler Hethur Stiftung & Co. KG", tax ID DE592806894, the
   totals, and the bank account printed on it.
3. Validation passes, so there's no re-read.
4. The amounts in "2.093,04 €" format become the number 2093.04.
5. The checks: the vendor matches by tax ID ✓; the PO exists and the amount is fine ✓; but
   **the printed bank account differs from the one in the vendor master** ✗.
6. Decision: **HOLD**, reason "remit-to bank account differs from the vendor master; verify
   by phone".
7. The agent explains this to the reviewer and suggests calling the vendor on the number on
   file.
8. No payment is made.

---

## 5. File-by-file tour

Everything lives in `src/invoiceguard/`. Suggested reading order: `schema.py` →
`checks.py` → `payments.py` → `pipeline.py` → `extract/` → `agent.py` → the eval files.

| File | What it's for | Key parts to recognise |
| --- | --- | --- |
| `schema.py` | Defines the "shape" of every piece of data: Invoice, Vendor, PurchaseOrder, BankAccount, the list of anomalies, and the decision rule. | `class Invoice`, `Anomaly`, `expected_decision()`, `same_destination()` (bank comparison) |
| `synth/generator.py` | Creates the fake but realistic dataset: vendors, POs, invoices, injected problems, labels. | `DatasetGenerator.generate()` |
| `synth/render.py` | Draws invoices as PDFs in 3 base layouts with per-vendor styles, and degrades some into "scans". | `render_pdf_bytes()`, `make_style()`, `degrade_to_scan()` |
| `synth/adversarial.py` | Creates the 50 attack invoices. | `AdversarialGenerator`, `lookalike_name()` |
| `extract/pdf_text.py` | Pulls text out of PDFs. | `pdf_to_text()` |
| `extract/models.py` | The JSON schema the LLM must fill: every field as a string, exactly as printed. | `ExtractedInvoice` |
| `extract/base.py` | Shared prompts and the result type. | `SYSTEM_PROMPT`, `ExtractionResult` |
| `extract/openai_compat.py` | Talks to Gemini (default) or Groq, with throttling, retries, and quota handling. | `OpenAICompatExtractor`, `extract_text()`, `extract_image()`, `chat()` |
| `extract/ollama.py`, `extract/llm.py` | Alternative backends: a local model (Ollama) and Claude. | — |
| `extract/document.py` | Picks text vs image reading, and runs the self-check (second read). | `extract_document()`, `validation_issues()` |
| `extract/normalize.py` | Turns printed strings into real numbers and dates. | `parse_number()`, `resolve_dates()`, `to_invoice()` |
| `checks.py` | **The rules.** Vendor matching, PO checks, duplicates, split billing, bank, arithmetic, instruction detection. | `InvoiceChecker.check()`, `VendorMaster.resolve()`, `INSTRUCTION_PATTERNS` |
| `payments.py` | Mock payments service with its own safety controls. | `MockPaymentsAPI.schedule()`, `approve()` |
| `pipeline.py` | Connects everything: extract → check → decide → explain → pay → audit. | `InvoicePipeline.process()` |
| `agent.py` | The review agent: tool-calling loop that explains held invoices. | `ReviewAgent.explain()`, `TOOLS` |
| `review.py` | Command-line tool for the human reviewer: list, show, approve, cancel. | `ReviewDesk` |
| `ui.py` | Small local web page to try the system. | `Session`, `PAGE` |
| `evaluate.py` | Scores extraction field by field. | `compare()`, `score()` |
| `checks_eval.py` | Scores decisions: precision, recall, false approvals. | `evaluate()` |
| `agent_eval.py` | Scores the agent's explanations. | `score_review()`, `classify_numbers()` |
| `tests/` | 107 automatic tests that check everything still works. AI calls are faked, so tests are free and fast. | — |

**Tools and libraries, and why each was chosen**

| Tool | Why |
| --- | --- |
| **Python** | The standard language for AI engineering. |
| **Pydantic** | Defines data shapes and validates them, and turns them into JSON schemas for the LLM. |
| **PyMuPDF** | Reads PDF text and turns pages into images. |
| **ReportLab, Faker, Pillow** | Generate the synthetic invoices: drawing PDFs, fake names and addresses, image degradation. |
| **httpx** | Makes web requests to the AI APIs. |
| **Gemini (free tier)** | Free, fast, supports JSON schemas, vision and tool calling. |
| **pytest / ruff / mypy** | Tests / style and bug checks / type checks. |
| **GitHub Actions** | Runs all checks automatically on every push (the green badge). |

---

## 6. The dataset

**Why synthetic?** No public dataset has invoices labelled with fraud and control outcomes,
and real invoices contain private data. Generating them gives perfect labels and full
control over which problems appear.

| | |
| --- | --- |
| Invoices | 300 (219 digital PDFs, 81 scans), 25 vendors, 236 POs |
| Currencies | USD, EUR, GBP, with US and European number and date formats |
| Layouts | 3 base layouts, each vendor with its own style (font, colours, column order, label wording) |
| Problems injected | 22 duplicates, 31 PO mismatches, 42 missing POs, 20 unknown vendors, 25 changed bank accounts, 13 arithmetic errors |
| Adversarial set | 50 invoices: 18 prompt injections (6 visible, 12 hidden in white 1pt text), 6 look-alike vendors, 6 split purchases (20 invoices), 6 clean controls |
| Labels | For each invoice: the printed values, the true vendor, the problems, the expected decision |

Everything is generated from a **seed** (`--seed 42`), so anyone can recreate exactly the
same dataset.

---

## 7. Design decisions (and why)

For each: the decision, why, and what the alternative was. Interviewers love "why" questions.

1. **"The LLM reads, rules decide."** LLMs are probabilistic and can be manipulated; payment
   decisions need to be consistent and auditable. *Alternative:* let the LLM decide
   directly. Simpler, but impossible to fully trust or audit.

2. **Verbatim extraction, then Python normalisation.** The model copies `2.093,04 €` exactly;
   code converts it. This keeps number and date handling testable, and stops the model from
   "fixing" a wrong total the checks need to see. *Alternative:* ask the model for clean
   numbers. Easier, but it hides errors and mixes up formats.

3. **Pay the vendor-master account, never the invoice's.** Even if every check failed, money
   can't be redirected by changing the invoice. *Alternative:* pay what's printed. That's
   exactly what fraudsters exploit.

4. **Second read never overwrites the first.** If reads disagree, a person checks. A model
   must not be able to "correct" a wrong total into an approval.

5. **Two layers against prompt injection.** The prompt fences the document as untrusted data,
   *and* a rule holds instruction-like text regardless of what the model did. Safe even if a
   future model is fooled.

6. **The agent explains but can't decide.** It has read-only tools, no decision tool, never
   sees raw document text, and sees bank accounts only as the last 4 digits.

7. **Measure everything, prioritise false approvals.** Holding a good invoice costs a minute;
   paying a fraudulent one costs the money. The system is tuned to fail toward holding.

8. **Free models by default, providers swappable.** One OpenAI-compatible client covers Gemini
   and Groq; Ollama runs locally; Claude is optional. *Trade-off:* free tiers have strict daily
   limits, so runs cache results and resume.

9. **Store the model's raw output.** When the normalisation code improves, old results can be
   rescored without new API calls. This is how the IBAN fix was measured for free.

10. **Labels list every check that should fire.** If one invoice trips two controls, both
    are labelled, so precision and recall stay honest.

---

## 8. Results, and what they mean

| What | Result | What it means |
| --- | --- | --- |
| Digital PDFs (219) | 100% of fields exact | The model read every field correctly. |
| Scans (81) | 98.8% with every critical field exact | 80 of 81; one tax ID missed on a very low-quality image. |
| All 300 end to end | 100% correct decisions, **0 false approvals** | Including scans, every decision matched the label. |
| Prompt injections (18) | 0 extracted values changed | The model ignored every instruction, and the rule held all 18. |
| Adversarial checks (50) | Every attack caught, 0 false alarms | Look-alikes, splits and injections all held; clean controls approved. |
| Review agent (50 reviews) | 100% explained every finding; 0 invented issues; 0 ungrounded numbers | Explanations were complete and based on real data. |
| Speed / cost | ~12 s per invoice, $0 | Free tier of Gemini Flash-Lite. |

Full detail: [EVAL.md](../EVAL.md).

---

## 9. What went wrong, and what it taught

These stories show judgement, which is what interviewers look for. Know them well.

1. **The "misread" bank accounts that weren't.** Scans first scored 95.1%. Three bank
   accounts looked misread, but the IBANs were read perfectly: the model had *also* copied the
   IBAN into the account-number field, and the comparison treated the extra field as a
   changed account. *Fix:* compare accounts by IBAN in the deterministic code. *Lesson:*
   investigate before blaming the model.

2. **Duplicates that looked like false approvals.** 3 duplicates were approved in a PDF-only
   run because their originals were scans, which hadn't been processed yet. *Lesson:* check
   what your evaluation actually covers. With scans included, all 22 duplicates were caught.

3. **The scorer was wrong, twice.** The grounding check flagged `10,000.00` against `10000`,
   then read `PO-2600104` as a negative number. *Lesson:* when a metric looks bad, test the
   metric.

4. **Grounded but wrong.** The agent compared tax-inclusive totals with a pre-tax PO total
   and claimed over-billing. Every number was real, so the grounding metric passed it.
   *Fix:* label every amount "before tax" or "including tax". *Lesson:* automated metrics
   miss reasoning errors; spot-check by hand.

5. **Labels were incomplete.** 4 invoices had an arithmetic error that also broke the PO
   tolerance, but only one problem was labelled. *Lesson:* label everything that should fire.

6. **Small local models skipped fields.** With Ollama, the model left out optional fields
   entirely. *Fix:* mark every field required (empty allowed).

7. **A rendering bug in the data.** UK addresses contained a line break that printed as a
   black box, which the model read as an "I". *Lesson:* bugs in your test data look like
   model errors.

8. **Free tiers have limits.** One model took 80–130 s per request, one allowed 20 requests
   a day, another stopped after about 500. *Fix:* cache results, stop cleanly on quota,
   resume later.

---

## 10. Honest limitations

Saying these yourself shows maturity.

- **Synthetic data.** Real invoices are messier: handwriting, multi-page, other languages.
- **2-way match only;** no goods-receipt (3-way) matching.
- **18 injections is a smoke test**, not proof of robustness.
- **Mock payments,** not a real bank or accounting system.
- **Free-tier models,** chosen for cost; production would need a paid tier with service guarantees.
- **One reasoning error found by hand** in the agent's explanations suggests explanations
  still need human spot checks.

---

## 11. Running it yourself

From the project folder, in PowerShell:

```powershell
# one-time setup
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# create the datasets
python -m invoiceguard.synth --n 300 --seed 42
python -m invoiceguard.synth --adversarial --out data/adversarial

# run the tests (free, no API key)
pytest

# the demo web page (no API key needed in "Labels" mode)
python -m invoiceguard.ui --today 2026-03-01
```

To use the real model, get a free key at <https://aistudio.google.com/apikey> and put
`GEMINI_API_KEY=your-key` in a file named `.env` in the project folder. Never share it or
commit it.

**A good live demo (about 3 minutes):** in the UI, open `doc_0005` (approved; point out
the payment goes to the vendor-master account), then `doc_0002` (changed bank account →
hold), then switch to the adversarial dataset and open `adv_0006` (prompt injection →
hold; read the quoted text aloud).

---

## 12. Interview questions you should be ready for

Practise these out loud. Answer in your own words; the notes give the key points.

### About the project
**Q: Walk me through the project.** → Use the 2-minute version from section 1.

**Q: Why did you build this?** → AP is high-volume, document-heavy, and has real fraud risk,
so it shows both how useful and how dangerous LLMs are. It forced safety thinking, not just
"make the demo work".

**Q: What was your role?** → Be honest: *"I designed and directed it using Claude Code. I
decided the problem, the safety principle, the trade-offs (free models, what to test), and
reviewed the results. The AI wrote the code. Since then I've been learning the codebase
in depth, and I can walk you through any part."* Then prove it by explaining a part well.

### Design
**Q: Why not let the LLM make the decision?** → Decision 1 in section 7. LLMs are
probabilistic and manipulable; money decisions need consistency and an audit trail.

**Q: Why extract strings instead of numbers?** → Decision 2. Testable formats, and the model
can't hide errors.

**Q: How do you stop paying a fraudster?** → Bank-change check, look-alike vendor check, and
payment always goes to the vendor-master account even if checks fail.

**Q: What happens when the AI is unsure?** → Validation failure triggers a second read;
disagreement means hold for a person; a second read never overwrites the first.

### AI-specific
**Q: What is prompt injection and how did you handle it?** → Text in the input that tries to
override instructions. Two layers: the prompt treats the document as untrusted data, and a
rule holds instruction-like text regardless. Tested on 18 invoices; 0 values changed and
all 18 held.

**Q: What's structured output and why use it?** → Forcing JSON matching a schema so code can
use the answer reliably, instead of parsing free text.

**Q: How does your agent work?** → Tool-calling loop: the model calls read-only tools (vendor
record, PO, prior invoices, bank comparison), reads results, repeats, then submits a review
in a fixed format. It can't change the decision and never sees raw document text.

**Q: How did you choose the model?** → Measured: a fast free model (Flash-Lite) matched
accuracy needs at ~12 s per invoice; a bigger model was slower and had a 20-a-day limit; a
local 7B model took ~3 minutes per invoice.

**Q: How would you reduce cost or latency at scale?** → Batch processing, caching, smaller
models for easy invoices, only re-reading on validation failures, running the agent only on
held invoices (already done).

### Evaluation
**Q: How do you know it works?** → Labelled synthetic dataset; field-level extraction
accuracy; decision precision/recall; false-approval rate; agent coverage and grounding.
Everything reproducible from a seed.

**Q: What's precision vs recall here?** → Precision: of invoices flagged, how many really had
the problem. Recall: of real problems, how many were flagged. For fraud, recall and false
approvals matter most.

**Q: Why synthetic data? Isn't that a weakness?** → It gives perfect labels and controlled
attacks; no public dataset has fraud labels. Yes, the limitation is real; next step would be
testing on real (anonymised) invoices.

**Q: Tell me about a bug you found.** → The IBAN story (section 9, #1). Shows investigation
before blaming the model, and fixing at the right layer.

### Engineering
**Q: How is it tested?** → 107 automated tests, AI calls faked so tests are free; lint (ruff)
and type checks (mypy); CI on GitHub runs everything on every push.

**Q: What would you improve?** → Section 10: real data, 3-way matching, bigger red-team set,
real payment integration, monitoring in production.

**Q: How would you deploy this?** → As a service that receives invoices (email or upload),
a queue to process them, a database for the vendor master and audit log, the review UI for
the AP team, and monitoring of extraction accuracy and hold rates over time.

---

## 13. What to learn next (for AI engineer roles)

Interviews for AI engineer roles usually include live Python coding, so the plan is to make
the skills your own:

1. **Python basics** (4–6 weeks): Harvard's free CS50P course. Then read `schema.py` and
   `checks.py`; you'll recognise most of it.
2. **LLM apps** (3–4 weeks): call an API yourself, get structured output, write a tool-calling
   loop. Compare with `extract/openai_compat.py` and `agent.py`.
3. **Evaluation** (2–3 weeks): build a small eval set for something of your own. Compare with
   `evaluate.py`.
4. **Your own build** (3–4 weeks): a smaller project written by you, with AI as a tutor rather
   than the author, e.g. a document Q&A tool (RAG).
5. **Make this project yours:** add a rule by hand (e.g. flag invoices over a set amount) and
   commit it. Each hand-made change is something you can truthfully say you built.

---

*All vendors, people and bank details in this project are fictional.*
