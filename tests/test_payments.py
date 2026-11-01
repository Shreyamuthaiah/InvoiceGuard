import json
from datetime import date, timedelta
from decimal import Decimal

import pytest

from invoiceguard.payments import MockPaymentsAPI, PaymentError, PaymentPolicy
from invoiceguard.pipeline import InvoicePipeline, labels_extractor
from invoiceguard.schema import Decision
from invoiceguard.synth import DatasetGenerator, write_dataset


@pytest.fixture(scope="module")
def ds():
    return DatasetGenerator(seed=21, n_vendors=10).generate(80)


def clean(ds):
    gt = next(g for g in ds.cases if not g.anomalies)
    vendor = next(v for v in ds.vendors if v.vendor_id == gt.vendor_id)
    return gt.invoice, vendor


def test_schedules_before_due_date_to_vendor_master_account(ds):
    inv, vendor = clean(ds)
    api = MockPaymentsAPI()
    today = inv.invoice_date
    pay, created = api.schedule(inv, vendor, Decision.APPROVE, today)
    assert created and pay.pay_on == max(today, inv.due_date - timedelta(days=2)) and not pay.late
    assert pay.destination == vendor.bank_account and pay.amount == inv.total


def test_never_pays_to_bank_details_from_the_invoice(ds):
    inv, vendor = clean(ds)
    tampered = inv.model_copy(update={"remit_to": inv.remit_to.model_copy(
        update={"account_number": "999999999", "iban": "XX00 FRAUD"})})
    pay, _ = MockPaymentsAPI().schedule(tampered, vendor, Decision.APPROVE, inv.invoice_date)
    assert pay.destination == vendor.bank_account


def test_overdue_invoice_paid_today_and_flagged_late(ds):
    inv, vendor = clean(ds)
    today = inv.due_date + timedelta(days=10)
    pay, _ = MockPaymentsAPI().schedule(inv, vendor, Decision.APPROVE, today)
    assert pay.pay_on == today and pay.late


@pytest.mark.parametrize("decision", [Decision.HOLD, Decision.REJECT])
def test_refuses_unapproved_decisions(ds, decision):
    inv, vendor = clean(ds)
    with pytest.raises(PaymentError):
        MockPaymentsAPI().schedule(inv, vendor, decision, inv.invoice_date)


def test_idempotent_and_blocks_same_invoice_with_new_amount(ds):
    inv, vendor = clean(ds)
    api = MockPaymentsAPI()
    first, _ = api.schedule(inv, vendor, Decision.APPROVE, inv.invoice_date)
    again, created = api.schedule(inv.model_copy(update={"invoice_number": inv.invoice_number.lower()}),
                                  vendor, Decision.APPROVE, inv.invoice_date)
    assert not created and again.payment_id == first.payment_id and len(api.payments) == 1
    with pytest.raises(PaymentError, match="already has payment"):
        api.schedule(inv.model_copy(update={"total": inv.total + 1}), vendor, Decision.APPROVE, inv.invoice_date)


def test_threshold_requires_named_approver(ds):
    inv, vendor = clean(ds)
    policy = PaymentPolicy(approval_threshold={vendor.currency: inv.total - Decimal("0.01")})
    api = MockPaymentsAPI(policy=policy)
    pay, _ = api.schedule(inv, vendor, Decision.APPROVE, inv.invoice_date)
    assert pay.status == "awaiting_approval"
    with pytest.raises(PaymentError):
        api.approve(pay.payment_id, "  ")
    assert api.approve(pay.payment_id, "AP manager").status == "scheduled"
    with pytest.raises(PaymentError):
        api.approve(pay.payment_id, "AP manager")


def test_state_persists_across_instances(ds, tmp_path):
    inv, vendor = clean(ds)
    store = tmp_path / "payments.json"
    MockPaymentsAPI(store).schedule(inv, vendor, Decision.APPROVE, inv.invoice_date)
    _, created = MockPaymentsAPI(store).schedule(inv, vendor, Decision.APPROVE, inv.invoice_date)
    assert not created


def test_pipeline_end_to_end_on_labels(ds, tmp_path):
    data = tmp_path / "data"
    write_dataset(ds, data)
    docs = [(g.doc_id, data / g.render.file) for g in ds.cases if not g.render.scanned]
    expected = {g.doc_id: g.expected_decision for g in ds.cases}

    def run(out):
        pipe = InvoicePipeline(ds.vendors, ds.purchase_orders, labels_extractor(data), out, date(2026, 1, 1))
        return [pipe.process(d, p) for d, p in docs]

    out = tmp_path / "run"
    results = run(out)
    assert all(r.decision == expected[r.doc_id] for r in results)
    approved = [r for r in results if r.decision == Decision.APPROVE]
    assert approved and all(r.payment and r.payment_created for r in approved)
    assert all(r.payment is None for r in results if r.decision != Decision.APPROVE)
    audit = [json.loads(line) for line in (out / "audit.jsonl").read_text().splitlines()]
    assert len(audit) == len(docs) and all(a["file_sha256"] for a in audit)
    queue = [json.loads(line) for line in (out / "review_queue.jsonl").read_text().splitlines()]
    assert {q["doc_id"] for q in queue} >= {r.doc_id for r in results if r.decision != Decision.APPROVE}
    # Re-processing the same inbox schedules nothing new.
    again = run(out)
    assert not any(r.payment_created for r in again)
    assert len(json.loads((out / "payments.json").read_text())) == len(approved)


def test_pipeline_holds_when_extraction_fails(ds, tmp_path):
    def broken(doc_id, path):
        return None, "model returned invalid JSON", "test"
    pipe = InvoicePipeline(ds.vendors, ds.purchase_orders, broken, tmp_path, date(2026, 1, 1))
    r = pipe.process("doc_x", tmp_path / "missing.pdf")
    assert r.decision == Decision.HOLD and "enter manually" in r.reasons[0] and r.payment is None
