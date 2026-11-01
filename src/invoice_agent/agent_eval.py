"""Evaluate the review agent's explanations on held/rejected invoices.

    python -m invoice_agent.agent_eval --data data/synthetic --run runs/agent-main --limit 30
    python -m invoice_agent.agent_eval --data data/adversarial --run runs/agent-adv

Invoices go through the checks in document order (so duplicate/split history builds up);
each held or rejected one is explained by the agent. Metrics:
- submitted:        the agent produced a valid review
- coverage:         every check that raised an anomaly is explained
- invented:         issues citing a check that raised no finding at all
- agreement:        the agent's recommended action equals the rules' decision
- ungrounded:       numbers in the explanation that never appeared in the case or tool results
Results are cached in <run>/reviews.jsonl; re-runs skip documents already explained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from invoice_agent.agent import ReviewAgent
from invoice_agent.checks import InvoiceChecker
from invoice_agent.checks_eval import load, load_texts
from invoice_agent.schema import Decision

_NUM = re.compile(r"\d[\d,]*\.\d+|\d{3,}")


def ungrounded_numbers(text: str, context: str) -> list[str]:
    ctx = context.replace(",", "")
    return [n for n in _NUM.findall(text) if n.replace(",", "") not in ctx]


def score_review(rec: dict) -> dict:
    review = rec.get("review")
    if not review:
        return {"submitted": False}
    expected, allowed = set(rec["expected_checks"]), set(rec["finding_checks"])
    cited = {i["check"] for i in review["issues"]}
    text = " ".join([review["summary"]] + [i["explanation"] + " " + i["next_step"] for i in review["issues"]])
    return {
        "submitted": True,
        "covered": expected <= cited,
        "missing": sorted(expected - cited),
        "invented": sorted(cited - allowed),
        "agrees": review["recommended_action"] == rec["decision"],
        "ungrounded": ungrounded_numbers(text, rec["context"]),
    }


def run(data: Path, run_dir: Path, llm, limit: int | None, log=print) -> list[dict]:
    vendors, pos, cases = load(data)
    texts = load_texts(data, cases)
    checker = InvoiceChecker(vendors, pos)
    agent = ReviewAgent(llm, checker)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "reviews.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("review"):
                done[rec["doc_id"]] = rec
    explained = 0
    with path.open("a", encoding="utf-8") as f:
        for gt in cases:  # all invoices go through the checks so history builds up
            report = checker.check(gt.invoice, ref=gt.doc_id, document_text=texts.get(gt.doc_id))
            if report.decision == Decision.APPROVE:
                continue
            if limit is not None and explained >= limit:
                continue
            explained += 1
            if gt.doc_id in done:
                continue
            reasons = [x.message for x in report.findings if x.anomaly]
            r = agent.explain(gt.doc_id, gt.invoice, report, report.decision, reasons)
            if r.error and "quota exhausted" in r.error:
                log(f"STOPPED: {r.error}. Re-run later to resume.")
                break
            rec = {**r.to_dict(), "context": r.context,
                   "expected_checks": sorted({x.check for x in report.findings if x.anomaly}),
                   "finding_checks": sorted({x.check for x in report.findings})}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            log(f"[{explained}] {gt.doc_id} {report.decision.value} "
                f"{'ok' if r.review else 'ERR ' + str(r.error)} ({len(r.tool_calls)} tool calls)")
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        latest[rec["doc_id"]] = rec
    return list(latest.values())


def summarize(records: list[dict]) -> dict:
    scored = [(r, score_review(r)) for r in records]
    sub = [s for _, s in scored if s["submitted"]]
    n = len(scored)
    pct = lambda k: round(100 * sum(1 for s in sub if s[k]) / len(sub), 1) if sub else None  # noqa: E731
    return {
        "reviews": n,
        "submitted_pct": round(100 * len(sub) / n, 1) if n else None,
        "coverage_pct": pct("covered"),
        "agreement_pct": pct("agrees"),
        "reviews_with_invented_issues": sum(1 for s in sub if s["invented"]),
        "reviews_with_ungrounded_numbers": sum(1 for s in sub if s["ungrounded"]),
        "avg_tool_calls": round(sum(len(r["tool_calls"]) for r, _ in scored) / n, 1) if n else None,
        "avg_latency_s": round(sum(r["latency_s"] for r, _ in scored) / n, 1) if n else None,
        "tokens_in": sum(r["input_tokens"] for r, _ in scored),
        "tokens_out": sum(r["output_tokens"] for r, _ in scored),
        "problems": [{"doc_id": r["doc_id"], **{k: v for k, v in s.items() if k != "submitted" and v}}
                     for r, s in scored if not s["submitted"] or s["missing"] or s["invented"]
                     or not s["agrees"] or s["ungrounded"]],
    }


def render(summary: dict, title: str, sample: list[dict]) -> str:
    lines = [f"# Agent eval: {title}", "",
             f"Reviews: {summary['reviews']} · submitted {summary['submitted_pct']}% · "
             f"covered every finding {summary['coverage_pct']}% · agreed with rules {summary['agreement_pct']}%  ",
             f"Reviews with invented issues: {summary['reviews_with_invented_issues']} · "
             f"with ungrounded numbers: {summary['reviews_with_ungrounded_numbers']}  ",
             f"Avg tool calls: {summary['avg_tool_calls']} · avg latency {summary['avg_latency_s']}s · "
             f"tokens {summary['tokens_in']} in / {summary['tokens_out']} out", ""]
    if summary["problems"]:
        lines += ["## Problems", ""] + [f"- {json.dumps(p)}" for p in summary["problems"]] + [""]
    lines += ["## Sample explanations", ""]
    for r in sample:
        rv = r.get("review")
        if not rv:
            continue
        lines += [f"### {r['doc_id']} ({r['decision']})", "", rv["summary"], ""]
        lines += [f"- **{i['check']}**: {i['explanation']} *Next:* {i['next_step']}" for i in rv["issues"]] + [""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--run", required=True)
    p.add_argument("--limit", type=int, default=None, help="explain at most N held/rejected invoices")
    p.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    p.add_argument("--model", default=None)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)
    run_dir = Path(a.run)
    if a.report_only:
        records = [json.loads(line) for line in (run_dir / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
        records = list({r["doc_id"]: r for r in records}.values())
    else:
        from invoice_agent.extract.openai_compat import OpenAICompatExtractor
        llm = OpenAICompatExtractor(a.provider, model=a.model)
        (run_dir).mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({"provider": a.provider, "model": llm.model}))
        records = run(Path(a.data), run_dir, llm, a.limit, log=lambda m: print(m, flush=True))
    summary = summarize(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = render(summary, run_dir.name, records[:5])
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
