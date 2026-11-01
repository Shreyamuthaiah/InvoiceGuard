import json
import threading
import urllib.error
import urllib.request
from datetime import date
from decimal import Decimal
from http.server import HTTPServer

import pytest

from invoice_agent.payments import PaymentError, PaymentPolicy
from invoice_agent.pipeline import InvoicePipeline, labels_extractor
from invoice_agent.review import ReviewDesk, main as review_main
from invoice_agent.synth import DatasetGenerator, write_dataset
from invoice_agent.ui import Session, make_handler


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("rv")
    ds = DatasetGenerator(seed=31, n_vendors=8).generate(50)
    write_dataset(ds, root / "synthetic")
    return ds, root


@pytest.fixture()
def run_dir(data, tmp_path):
    ds, root = data
    # Low threshold so some payments need sign-off.
    pipe = InvoicePipeline(ds.vendors, ds.purchase_orders, labels_extractor(root / "synthetic"), tmp_path,
                           date(2026, 1, 1), PaymentPolicy(approval_threshold={c: Decimal("2000")
                                                                               for c in ("USD", "EUR", "GBP")}))
    for g in ds.cases:
        if not g.render.scanned:
            pipe.process(g.doc_id, root / "synthetic" / g.render.file)
    return tmp_path


def test_queue_approve_cancel_are_audited(run_dir):
    desk = ReviewDesk(run_dir)
    q = desk.queue()
    assert q["awaiting_approval"] and q["held_or_rejected"]
    pid = q["awaiting_approval"][0]["payment_id"]
    with pytest.raises(PaymentError):
        desk.approve(pid, " ")
    assert desk.approve(pid, "Dana").status == "scheduled"
    assert pid not in {p["payment_id"] for p in ReviewDesk(run_dir).queue()["awaiting_approval"]}  # persisted
    with pytest.raises(PaymentError):
        desk.cancel(pid, "Dana", "")
    desk.cancel(pid, "Dana", "vendor asked to re-issue")
    actions = [json.loads(l) for l in (run_dir / "audit.jsonl").read_text().splitlines()]
    assert [a["action"] for a in actions if "action" in a] == ["approve_payment", "cancel_payment"]
    assert actions[-1]["by"] == "Dana" and actions[-1]["reason"] == "vendor asked to re-issue"


def test_show_has_full_invoice(run_dir):
    desk = ReviewDesk(run_dir)
    held = desk.queue()["held_or_rejected"][0]
    rec = desk.show(held["doc_id"])
    assert rec["invoice"]["invoice_number"] and rec["findings"]


def test_review_cli_exit_codes(run_dir, capsys):
    assert review_main(["--run", str(run_dir), "list"]) == 0
    assert "Awaiting approval" in capsys.readouterr().out
    assert review_main(["--run", str(run_dir), "approve", "PAY-99999", "--as", "Dana"]) == 1
    assert review_main(["--run", str(run_dir / "nope"), "list"]) == 1


@pytest.fixture()
def server(data, tmp_path):
    _, root = data
    session = Session([root / "synthetic"], tmp_path / "ui", "gemini", None, date(2026, 1, 1))
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(session))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _call(url, data=None, method="GET"):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_ui_serves_page_and_processes_in_order(server, data):
    ds, _ = data
    status, html = _call(server + "/")
    assert status == 200 and b"Invoice Agent" in html and b"textContent" in html
    docs = json.loads(_call(server + "/api/docs")[1])["synthetic"]["docs"]
    assert len(docs) == len(ds.cases)
    dup = next(g for g in ds.cases if g.duplicate_of and not g.render.scanned)
    for doc_id in (dup.duplicate_of, dup.doc_id):
        status, body = _call(f"{server}/api/process?dataset=synthetic&doc_id={doc_id}&source=labels", b"", "POST")
        assert status == 200
    result = json.loads(body)
    assert result["decision"] == "reject" and result["expected"]["decision"] == "reject"
    assert result["preview"].startswith("data:image/")
    stats = json.loads(_call(server + "/api/stats")[1])
    assert stats["processed"] == 2 and stats["reject"] == 1


def test_ui_rejects_bad_input(server):
    status, body = _call(f"{server}/api/process?dataset=synthetic&doc_id=../../secret&source=labels", b"", "POST")
    assert status == 400 and b"unknown document" in body
    status, body = _call(f"{server}/api/upload?name=evil.pdf", b"<script>alert(1)</script>", "POST")
    assert status == 400 and b"only PDF" in body
    assert _call(server + "/nope")[0] == 404
