import json

import pytest

from invoice_agent.agent import ReviewAgent
from invoice_agent.agent_eval import score_review, summarize, ungrounded_numbers
from invoice_agent.checks import InvoiceChecker
from invoice_agent.schema import Anomaly, Decision
from invoice_agent.synth import DatasetGenerator


@pytest.fixture(scope="module")
def ds():
    return DatasetGenerator(seed=23, n_vendors=8).generate(150)


def bank_case(ds):
    checker = InvoiceChecker(ds.vendors, ds.purchase_orders)
    gt = next(g for g in ds.cases if g.anomalies == [Anomaly.BANK_DETAILS_CHANGED])
    return checker, gt, checker.check(gt.invoice, ref=gt.doc_id)


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)},
            "extra_content": {"google": {"thought_signature": "sig-" + cid}}}


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.payloads = []

    def chat(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))  # snapshot
        msg = self.turns.pop(0)
        if isinstance(msg, Exception):
            raise msg
        return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 500, "completion_tokens": 80}}


GOOD_REVIEW = {"summary": "Held: the remit-to account differs from the vendor master.",
               "issues": [{"check": "bank", "explanation": "Invoice account differs from the account on file.",
                           "next_step": "Call the vendor on the number in the vendor master before paying."}],
               "recommended_action": "hold"}


def test_tool_loop_masks_accounts_and_echoes_messages(ds):
    checker, gt, report = bank_case(ds)
    turn1 = {"role": "assistant", "content": None, "tool_calls": [
        call("get_vendor_record", {"vendor_id": gt.vendor_id}, "c1"), call("compare_bank_details", {}, "c2")]}
    turn2 = {"role": "assistant", "content": None, "tool_calls": [call("submit_review", GOOD_REVIEW, "c3")]}
    llm = ScriptedLLM([turn1, turn2])
    r = ReviewAgent(llm, checker).explain(gt.doc_id, gt.invoice, report, report.decision)
    assert r.review and r.review.recommended_action == "hold" and r.error is None
    assert r.tool_calls == ["get_vendor_record", "compare_bank_details", "submit_review"]
    second = llm.payloads[1]["messages"]
    assert second[2] == turn1  # assistant message (with thought signatures) sent back unchanged
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    everything = json.dumps(llm.payloads)
    vendor = next(v for v in ds.vendors if v.vendor_id == gt.vendor_id)
    for acct in [gt.invoice.remit_to.iban or gt.invoice.remit_to.account_number,
                 vendor.bank_account.iban or vendor.bank_account.account_number]:
        assert acct.replace(" ", "") not in everything.replace(" ", "")  # only last 4 digits
        assert acct.replace(" ", "")[-4:] in everything
    assert json.loads(tool_msgs[1]["content"])["same_account"] is False
    assert r.input_tokens == 1000


def test_invalid_review_is_rejected_then_fixed(ds):
    checker, gt, report = bank_case(ds)
    bad = {**GOOD_REVIEW, "recommended_action": "pay immediately"}
    llm = ScriptedLLM([{"role": "assistant", "tool_calls": [call("submit_review", bad, "a")]},
                       {"role": "assistant", "tool_calls": [call("submit_review", GOOD_REVIEW, "b")]}])
    r = ReviewAgent(llm, checker).explain(gt.doc_id, gt.invoice, report, report.decision)
    assert r.review and r.steps == 2
    assert "invalid review" in llm.payloads[1]["messages"][-1]["content"]


def test_gives_up_after_max_steps_and_records_api_errors(ds):
    checker, gt, report = bank_case(ds)
    chatty = ScriptedLLM([{"role": "assistant", "content": "Thinking..."}] * 3)
    r = ReviewAgent(chatty, checker, max_steps=3).explain(gt.doc_id, gt.invoice, report, report.decision)
    assert r.review is None and "no review submitted" in r.error
    assert chatty.payloads[1]["messages"][-1]["content"].startswith("Call submit_review")
    broken = ScriptedLLM([RuntimeError("quota exhausted: resets in ~5h")])
    r = ReviewAgent(broken, checker).explain(gt.doc_id, gt.invoice, report, report.decision)
    assert r.review is None and "quota exhausted" in r.error


def test_tools_answer_from_master_data(ds):
    checker = InvoiceChecker(ds.vendors, ds.purchase_orders)
    gt = next(g for g in ds.cases if Anomaly.UNKNOWN_VENDOR in g.anomalies)
    report = checker.check(gt.invoice, ref=gt.doc_id)
    agent = ReviewAgent(ScriptedLLM([]), checker)
    sim = agent._tool("find_similar_vendors", {"name": gt.invoice.vendor_name}, gt.invoice, report)
    assert len(sim["matches"]) == 3 and all("tax_id" in m for m in sim["matches"])
    assert agent._tool("get_purchase_order", {"po_number": "PO-0"}, gt.invoice, report) == {"error": "PO not found"}
    po = ds.purchase_orders[0]
    assert agent._tool("get_purchase_order", {"po_number": po.po_number.lower()}, gt.invoice, report)["total"] == f"{po.total:.2f}"
    assert agent._tool("compare_bank_details", {}, gt.invoice, report)["vendor_master"] is None


def test_scoring_and_grounding():
    rec = {"doc_id": "d", "decision": "hold", "context": '{"total": "1234.50", "po": "PO-2600104"}',
           "expected_checks": ["bank"], "finding_checks": ["bank", "duplicate"], "tool_calls": [], "latency_s": 1,
           "input_tokens": 1, "output_tokens": 1,
           "review": {**GOOD_REVIEW, "summary": "Total 1,234.50 on PO-2600104 but the PO said 9,999.00."}}
    s = score_review(rec)
    assert s["covered"] and s["agrees"] and s["invented"] == [] and s["ungrounded"] == ["9,999.00"]
    assert ungrounded_numbers("totals 1234.50", rec["context"]) == []
    summary = summarize([rec])
    assert summary["coverage_pct"] == 100.0 and summary["reviews_with_ungrounded_numbers"] == 1


def test_pipeline_explains_only_held_and_rejected(ds, tmp_path):
    from datetime import date
    from invoice_agent.pipeline import InvoicePipeline

    class AlwaysSubmits(ScriptedLLM):
        def chat(self, payload):
            self.payloads.append(payload)
            return {"choices": [{"message": {"role": "assistant",
                                             "tool_calls": [call("submit_review", GOOD_REVIEW)]}}]}

    llm = AlwaysSubmits([])
    gts = ds.cases[:25]
    pipe = InvoicePipeline(ds.vendors, ds.purchase_orders, lambda d, p: (next(g for g in gts if g.doc_id == d).invoice,
                           None, "labels"), tmp_path, date(2026, 1, 1), explain_llm=llm)
    results = [pipe.process(g.doc_id, tmp_path / "none.pdf") for g in gts]
    not_approved = [r for r in results if r.decision != Decision.APPROVE]
    assert not_approved and len(llm.payloads) == len(not_approved)
    assert all(r.explanation and r.explanation["summary"] for r in not_approved)
    assert all(r.explanation is None for r in results if r.decision == Decision.APPROVE)
    audit = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert sum(1 for a in audit if a["explanation"]) == len(not_approved)
