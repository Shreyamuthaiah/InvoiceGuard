import json
import re

import pymupdf
import pytest
from PIL import Image

from invoiceguard.schema import PO_TOLERANCE, Anomaly, Decision, GroundTruth, Vendor
from invoiceguard.synth import DatasetGenerator, write_dataset
from invoiceguard.synth.render import fmt_money, resolve_style

N = 80


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    ds = DatasetGenerator(seed=7, n_vendors=10).generate(N)
    stats = write_dataset(ds, out)
    return ds, out, stats


def _digits(s: str) -> str:
    return re.sub(r"[\s.,]", "", s)


def test_files_and_labels_exist(dataset):
    ds, out, stats = dataset
    assert stats["documents"] == N
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == N
    for row in manifest:
        assert (out / row["file"]).exists()
        gt = GroundTruth.model_validate_json((out / row["label"]).read_text(encoding="utf-8"))
        assert gt.doc_id == row["doc_id"]
    vendors = [Vendor.model_validate(v) for v in json.loads((out / "vendors.json").read_text(encoding="utf-8"))]
    assert len(vendors) == 10


def test_every_anomaly_type_present(dataset):
    ds, _, _ = dataset
    seen = {a for gt in ds.cases for a in gt.anomalies}
    adversarial_only = {Anomaly.SUSPICIOUS_INSTRUCTIONS, Anomaly.SPLIT_INVOICE}
    assert seen == set(Anomaly) - adversarial_only
    assert any(not gt.anomalies for gt in ds.cases)


def test_arithmetic_consistent_unless_flagged(dataset):
    ds, _, _ = dataset
    for gt in ds.cases:
        issues = gt.invoice.arithmetic_issues()
        if Anomaly.ARITHMETIC_ERROR in gt.anomalies:
            assert issues, gt.doc_id
        else:
            assert not issues, (gt.doc_id, issues)


def test_labels_agree_with_vendor_master_and_pos(dataset):
    ds, _, _ = dataset
    vendors = {v.vendor_id: v for v in ds.vendors}
    pos = {p.po_number: p for p in ds.purchase_orders}
    by_id = {gt.doc_id: gt for gt in ds.cases}
    for gt in ds.cases:
        inv, a = gt.invoice, gt.anomalies
        if Anomaly.UNKNOWN_VENDOR in a:
            assert gt.vendor_id is None
            assert inv.vendor_name not in {v.name for v in ds.vendors}
            continue
        v = vendors[gt.vendor_id]
        assert inv.vendor_name == v.name and inv.vendor_tax_id == v.tax_id
        assert inv.remit_to.same_destination(v.bank_account) == (Anomaly.BANK_DETAILS_CHANGED not in a)
        if Anomaly.MISSING_PO in a:
            assert inv.po_number is None
        else:
            po = pos[inv.po_number]
            assert po.vendor_id == v.vendor_id
            over = inv.subtotal > po.total * (1 + PO_TOLERANCE)
            if Anomaly.PO_AMOUNT_MISMATCH in a:
                assert over
            elif Anomaly.ARITHMETIC_ERROR not in a:
                assert inv.subtotal == po.total
        if Anomaly.DUPLICATE in a:
            orig = by_id[gt.duplicate_of]
            assert orig.invoice.invoice_number == inv.invoice_number and not orig.anomalies


def test_invoice_numbers_unique_except_duplicates(dataset):
    ds, _, _ = dataset
    keys = [(gt.vendor_id, gt.invoice.invoice_number) for gt in ds.cases if Anomaly.DUPLICATE not in gt.anomalies]
    assert len(keys) == len(set(keys))


def test_decisions_follow_policy(dataset):
    ds, _, _ = dataset
    for gt in ds.cases:
        if Anomaly.DUPLICATE in gt.anomalies:
            assert gt.expected_decision == Decision.REJECT
        elif gt.anomalies:
            assert gt.expected_decision == Decision.HOLD
        else:
            assert gt.expected_decision == Decision.APPROVE


def test_pdf_text_contains_key_fields(dataset):
    ds, out, _ = dataset
    pdfs = [gt for gt in ds.cases if not gt.render.scanned]
    assert pdfs
    for gt in pdfs:
        with pymupdf.open(out / gt.render.file) as doc:
            text = doc[0].get_text()
        inv = gt.invoice
        assert inv.invoice_number in text
        assert inv.vendor_tax_id in text
        st = resolve_style(gt.render.template, inv.currency, gt.render.style)
        total = fmt_money(inv.total, inv.currency, st.money)
        assert _digits(total) in _digits(text), (gt.doc_id, total)


def test_scans_are_images(dataset):
    ds, out, _ = dataset
    scans = [gt for gt in ds.cases if gt.render.scanned]
    assert scans
    for gt in scans:
        with Image.open(out / gt.render.file) as img:
            assert img.format == "JPEG" and img.width > 800
        assert set(gt.render.degradations) >= {"dpi", "rotation_deg", "jpeg_quality"}


def test_deterministic(tmp_path):
    a = DatasetGenerator(seed=3, n_vendors=5).generate(25)
    b = DatasetGenerator(seed=3, n_vendors=5).generate(25)
    assert [g.model_dump() for g in a.cases] == [g.model_dump() for g in b.cases]
    write_dataset(a, tmp_path / "a")
    write_dataset(b, tmp_path / "b")
    for gt in a.cases:
        assert (tmp_path / "a" / gt.render.file).read_bytes() == (tmp_path / "b" / gt.render.file).read_bytes()


def test_layout_variety_changes_look_not_content():
    plain = DatasetGenerator(seed=9, n_vendors=8, layout_variety=False).generate(40)
    varied = DatasetGenerator(seed=9, n_vendors=8).generate(40)
    assert [g.invoice for g in plain.cases] == [g.invoice for g in varied.cases]
    assert [g.anomalies for g in plain.cases] == [g.anomalies for g in varied.cases]
    assert all(g.render.style is None for g in plain.cases)
    styles = {g.render.style.model_dump_json() for g in varied.cases}
    assert len(styles) >= 8  # every vendor looks different
    by_vendor = {}
    for g in varied.cases:
        if g.vendor_id:
            by_vendor.setdefault(g.vendor_id, set()).add(g.render.style.model_dump_json())
    assert all(len(s) == 1 for s in by_vendor.values())  # ...but consistent per vendor
