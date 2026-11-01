from decimal import Decimal

import pytest

from invoice_agent.checks import InvoiceChecker, name_similarity, norm_id, norm_name
from invoice_agent.checks_eval import evaluate
from invoice_agent.schema import Anomaly, Decision
from invoice_agent.synth import DatasetGenerator


@pytest.fixture(scope="module")
def ds():
    return DatasetGenerator(seed=5, n_vendors=12).generate(120)


def clean_case(ds):
    return next(g for g in ds.cases if not g.anomalies)


def fresh(ds):
    return InvoiceChecker(ds.vendors, ds.purchase_orders)


def test_checks_reproduce_every_label(ds):
    summary = evaluate(ds.vendors, ds.purchase_orders, ds.cases)
    assert summary["decision_accuracy"] == 1.0 and summary["false_approves"] == 0
    assert summary["mistakes"] == []


def test_clean_invoice_is_approved(ds):
    report = fresh(ds).check(clean_case(ds).invoice)
    assert report.anomalies == [] and report.decision == Decision.APPROVE
    assert report.vendor_match.method == "tax_id"


def test_name_normalisation():
    assert norm_name("Müller GmbH & Co. KG") == norm_name("Müller")
    assert name_similarity("Gonzalez-Curtis", "Gonzalez Curtis LLC") > 0.95


def test_tax_id_formatting_noise_still_matches(ds):
    inv = clean_case(ds).invoice
    noisy = inv.model_copy(update={"vendor_tax_id": " " + inv.vendor_tax_id.replace("-", " ") + " "})
    assert fresh(ds).check(noisy).vendor_match.method == "tax_id"


def test_name_only_match_when_tax_id_missing(ds):
    inv = clean_case(ds).invoice
    report = fresh(ds).check(inv.model_copy(update={"vendor_tax_id": ""}))
    assert report.vendor_match.method == "name" and Anomaly.UNKNOWN_VENDOR not in report.anomalies


def test_lookalike_vendor_with_other_tax_id_is_unknown(ds):
    inv = clean_case(ds).invoice
    fake = inv.model_copy(update={"vendor_name": inv.vendor_name + " Ltd", "vendor_tax_id": "99-9999999"})
    report = fresh(ds).check(fake)
    assert Anomaly.UNKNOWN_VENDOR in report.anomalies and report.decision == Decision.HOLD
    assert any("look-alike" in f.message for f in report.findings)


def test_po_not_found_and_wrong_vendor(ds):
    checker = fresh(ds)
    inv = clean_case(ds).invoice
    no_such_po = inv.model_copy(update={"po_number": "PO-0000000"})
    assert Anomaly.MISSING_PO in checker.check(no_such_po, record=False).anomalies
    vendor_id = checker.master.by_tax_id[norm_id(inv.vendor_tax_id)].vendor_id
    other_po = next(p for p in ds.purchase_orders if p.vendor_id != vendor_id)
    report = checker.check(inv.model_copy(update={"po_number": other_po.po_number}), record=False)
    assert Anomaly.MISSING_PO in report.anomalies
    assert any("belongs to" in f.message for f in report.findings)


def test_po_tolerance_boundary(ds):
    checker = fresh(ds)
    inv = clean_case(ds).invoice
    po = checker.pos[inv.po_number.upper()]
    within = inv.model_copy(update={"subtotal": (po.total * Decimal("1.02")).quantize(Decimal("0.01"))})
    over = inv.model_copy(update={"subtotal": po.total * Decimal("1.03")})
    assert Anomaly.PO_AMOUNT_MISMATCH not in checker.check(within, record=False).anomalies
    assert Anomaly.PO_AMOUNT_MISMATCH in checker.check(over, record=False).anomalies


def test_duplicate_and_near_duplicate(ds):
    checker = fresh(ds)
    inv = clean_case(ds).invoice
    assert checker.check(inv, ref="first").decision == Decision.APPROVE
    again = checker.check(inv.model_copy(update={"invoice_number": " " + inv.invoice_number.lower() + " "}))
    assert again.decision == Decision.REJECT  # case/whitespace differences don't hide a duplicate
    renumbered = checker.check(inv.model_copy(update={"invoice_number": inv.invoice_number + "-B"}))
    assert Anomaly.DUPLICATE not in renumbered.anomalies
    assert any(f.check == "duplicate" and f.severity == "warning" for f in renumbered.findings)


def test_changed_bank_details_held(ds):
    inv = clean_case(ds).invoice
    bank = inv.remit_to.model_copy(update={"account_number": "000111222", "iban": None})
    if inv.remit_to.iban:
        bank = inv.remit_to.model_copy(update={"iban": "DE00000000000000000000"})
    report = fresh(ds).check(inv.model_copy(update={"remit_to": bank}))
    assert report.anomalies == [Anomaly.BANK_DETAILS_CHANGED] and report.decision == Decision.HOLD


def test_extraction_errors_flow_into_decisions(ds):
    """End-to-end mode: a misread total on a clean invoice should not be silently approved."""
    preds = {g.doc_id: g.invoice for g in ds.cases}
    target = clean_case(ds)
    preds[target.doc_id] = target.invoice.model_copy(update={"total": target.invoice.total + 100})
    summary = evaluate(ds.vendors, ds.purchase_orders, ds.cases, preds)
    assert summary["false_approves"] == 0
    assert any(m["doc_id"] == target.doc_id for m in summary["mistakes"])


def test_duplicate_of_unprocessed_original_is_a_first_sighting(ds):
    dup = next(g for g in ds.cases if Anomaly.DUPLICATE in g.anomalies)
    preds = {g.doc_id: g.invoice for g in ds.cases if g.doc_id != dup.duplicate_of}
    summary = evaluate(ds.vendors, ds.purchase_orders, ds.cases, preds)
    assert dup.doc_id in summary["duplicates_of_unprocessed"]
    assert summary["mistakes"] == [] and summary["false_approves"] == 0
