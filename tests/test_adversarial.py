import json
from collections import Counter
from datetime import timedelta

import pytest

from invoice_agent.checks import InvoiceChecker, check_instructions
from invoice_agent.checks_eval import evaluate, load_texts
from invoice_agent.extract import pdf_to_text
from invoice_agent.schema import APPROVAL_THRESHOLDS, Anomaly, Decision
from invoice_agent.synth import DatasetGenerator, write_dataset
from invoice_agent.synth.adversarial import KINDS, AdversarialGenerator, lookalike_name


@pytest.fixture(scope="module")
def adv(tmp_path_factory):
    out = tmp_path_factory.mktemp("adv")
    ds = AdversarialGenerator(seed=13, n_vendors=10, per_kind=4).generate()
    write_dataset(ds, out)
    return ds, out


def test_every_kind_present_and_labelled(adv):
    ds, _ = adv
    seen = Counter(a for g in ds.cases for a in g.anomalies)
    for a in (Anomaly.SUSPICIOUS_INSTRUCTIONS, Anomaly.SPLIT_INVOICE, Anomaly.UNKNOWN_VENDOR,
              Anomaly.BANK_DETAILS_CHANGED):
        assert seen[a] > 0
    assert any(not g.anomalies for g in ds.cases)  # clean controls exist
    assert len(KINDS) == 6


def test_shares_vendor_master_with_main_dataset():
    adv_ds = AdversarialGenerator(seed=13, n_vendors=10).generate()
    main = DatasetGenerator(seed=13, n_vendors=10).generate(1)
    assert [v.model_dump() for v in adv_ds.vendors] == [v.model_dump() for v in main.vendors]


def test_checks_catch_every_attack_from_the_pdf_text(adv):
    ds, out = adv
    summary = evaluate(ds.vendors, ds.purchase_orders, ds.cases, texts=load_texts(out, ds.cases))
    assert summary["mistakes"] == [] and summary["false_approves"] == 0


def test_injections_are_in_the_text_layer_but_hidden_ones_are_invisible(adv):
    ds, out = adv
    hidden = next(g for g in ds.cases if any(i.mode == "hidden" for i in g.render.injections))
    text = pdf_to_text(out / hidden.render.file)
    assert hidden.render.injections[0].text[:30] in " ".join(text.split())
    import pymupdf
    with pymupdf.open(out / hidden.render.file) as doc:
        spans = [s for b in doc[0].get_text("dict")["blocks"] for line in b.get("lines", [])
                 for s in line["spans"] if hidden.render.injections[0].text[:20] in s["text"]]
    assert spans and all(s["size"] <= 1.5 and s["color"] == 0xFFFFFF for s in spans)


def test_hidden_total_label_keeps_printed_values(adv):
    ds, out = adv
    g = next(g for g in ds.cases if "Labels keep the printed values" in " ".join(g.notes))
    from invoice_agent.synth.render import fmt_money, resolve_style
    text = pdf_to_text(out / g.render.file)
    st = resolve_style(g.render.template, g.invoice.currency, g.render.style)
    assert fmt_money(g.invoice.total, g.invoice.currency, st.money) in text  # real total is printed
    fake = g.render.injections[0].text
    assert f"{g.invoice.total:.2f}" not in fake  # and the injected one is different
    assert not g.invoice.arithmetic_issues()


def test_split_invoices_each_under_threshold_and_sum_to_po(adv):
    ds, _ = adv
    pos = {p.po_number: p for p in ds.purchase_orders}
    groups: dict[str, list] = {}
    for g in ds.cases:
        if any("Part " in n for n in g.notes):
            groups.setdefault(g.invoice.po_number, []).append(g)
    assert groups
    for po_number, parts in groups.items():
        limit = APPROVAL_THRESHOLDS[parts[0].invoice.currency]
        assert all(p.invoice.total <= limit for p in parts)
        assert sum(p.invoice.total for p in parts) > limit
        assert sum(p.invoice.subtotal for p in parts) == pos[po_number].total
        assert parts[0].expected_decision == Decision.APPROVE
        assert all(p.anomalies == [Anomaly.SPLIT_INVOICE] for p in parts[1:])


def test_lookalike_names_are_close_but_different():
    import random
    rng = random.Random(0)
    for name in ["Gonzalez-Curtis", "Salazar, Aguirre and Lopez", "Müller GmbH"]:
        fake = lookalike_name(name, rng)
        assert fake != name and len(fake) >= len(name) - 1


@pytest.mark.parametrize("text", [
    "Thank you for your business. Questions? billing@acme.example",
    "Managed IT support - monthly retainer",
    "Payment Terms Net 30. Please pay by bank transfer to the account below.",
    "Help desk tickets (overage)  Premium support plan  System integration services",
    "Remit Payment To First Harbor Bank Account name: Smith LLC",
    "Approved by: J. Smith (purchasing)",
])
def test_no_false_alarm_on_ordinary_invoice_language(text):
    assert check_instructions(text) == []


def test_po_overbilling_across_invoices():
    ds = DatasetGenerator(seed=3, n_vendors=6).generate(60)
    gt = next(g for g in ds.cases if not g.anomalies and g.invoice.po_number)
    checker = InvoiceChecker(ds.vendors, ds.purchase_orders)
    assert checker.check(gt.invoice, ref="a").decision == Decision.APPROVE
    second = gt.invoice.model_copy(update={"invoice_number": gt.invoice.invoice_number + "-2",
                                           "invoice_date": gt.invoice.invoice_date + timedelta(days=60)})
    report = checker.check(second, ref="b")
    assert Anomaly.PO_AMOUNT_MISMATCH in report.anomalies
    assert any("billed" in f.message for f in report.findings)


def test_adversarial_cli(tmp_path):
    from invoice_agent.synth.__main__ import main
    import sys
    argv = sys.argv
    sys.argv = ["synth", "--adversarial", "--per-kind", "2", "--vendors", "6", "--out", str(tmp_path)]
    try:
        main()
    finally:
        sys.argv = argv
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["documents"] >= 12 and (tmp_path / "docs" / "adv_0001.pdf").exists()
