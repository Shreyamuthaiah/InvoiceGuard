"""The human side of the pipeline: work the review queue and sign off payments.

    python -m invoiceguard.review --run runs/pipe list
    python -m invoiceguard.review --run runs/pipe show doc_0042
    python -m invoiceguard.review --run runs/pipe payments
    python -m invoiceguard.review --run runs/pipe approve PAY-00007 --as "Dana (AP manager)"
    python -m invoiceguard.review --run runs/pipe cancel PAY-00007 --as "Dana" --reason "vendor called"

Every approve/cancel is appended to audit.jsonl with who did it and when.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from invoiceguard.payments import MockPaymentsAPI, Payment, PaymentError


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ReviewDesk:
    def __init__(self, run_dir: Path) -> None:
        if not (run_dir / "audit.jsonl").exists():
            raise FileNotFoundError(f"{run_dir} has no audit.jsonl; run the pipeline first")
        self.run_dir = run_dir
        self.payments = MockPaymentsAPI(run_dir / "payments.json")

    def latest_audit(self) -> dict[str, dict]:
        """Most recent pipeline record per document (re-runs append)."""
        out: dict[str, dict] = {}
        for rec in _jsonl(self.run_dir / "audit.jsonl"):
            if rec.get("action", "process") == "process":
                out[rec["doc_id"]] = rec
        return out

    def queue(self) -> dict[str, list[dict]]:
        audit = self.latest_audit()
        holds = [r for r in audit.values() if r["decision"] in ("hold", "reject")]
        waiting = [p for p in self.payments.payments.values() if p.status == "awaiting_approval"]
        return {"awaiting_approval": [p.model_dump(mode="json") for p in waiting], "held_or_rejected": holds}

    def show(self, doc_id: str) -> dict:
        audit = self.latest_audit()
        if doc_id not in audit:
            raise KeyError(f"no record for {doc_id}")
        return audit[doc_id]

    def _log(self, action: str, payment: Payment, who: str, reason: str = "") -> None:
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action,
               "doc_id": payment.source_ref, "payment_id": payment.payment_id, "by": who, "reason": reason,
               "amount": f"{payment.amount:.2f}", "currency": payment.currency}
        with (self.run_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def approve(self, payment_id: str, who: str) -> Payment:
        p = self.payments.approve(payment_id, who)
        self._log("approve_payment", p, who)
        return p

    def cancel(self, payment_id: str, who: str, reason: str) -> Payment:
        if not who.strip() or not reason.strip():
            raise PaymentError("cancelling needs --as and --reason")
        p = self.payments.cancel(payment_id)
        self._log("cancel_payment", p, who.strip(), reason.strip())
        return p


def _print_queue(q: dict) -> None:
    waiting, holds = q["awaiting_approval"], q["held_or_rejected"]
    print(f"Awaiting approval ({len(waiting)}):")
    for p in waiting:
        print(f"  {p['payment_id']}  {p['source_ref']:<9} {p['vendor_name'][:28]:<28} "
              f"{float(p['amount']):>12,.2f} {p['currency']}  pay on {p['pay_on']}")
    print(f"\nHeld or rejected ({len(holds)}):")
    for r in sorted(holds, key=lambda r: r["doc_id"]):
        reason = r["reasons"][0] if r["reasons"] else ""
        print(f"  {r['doc_id']:<9} {r['decision']:<7} {str(r.get('vendor'))[:26]:<26} {reason[:70]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="pipeline output directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="review queue: payments awaiting approval, held and rejected invoices")
    s = sub.add_parser("show", help="full record for one document")
    s.add_argument("doc_id")
    sub.add_parser("payments", help="all payments")
    a_ = sub.add_parser("approve", help="sign off a payment over the threshold")
    a_.add_argument("payment_id")
    a_.add_argument("--as", dest="who", required=True, help="approver name")
    c = sub.add_parser("cancel", help="cancel a scheduled payment")
    c.add_argument("payment_id")
    c.add_argument("--as", dest="who", required=True)
    c.add_argument("--reason", required=True)
    a = p.parse_args(argv)

    try:
        desk = ReviewDesk(Path(a.run))
        if a.cmd == "list":
            _print_queue(desk.queue())
        elif a.cmd == "show":
            print(json.dumps(desk.show(a.doc_id), indent=2))
        elif a.cmd == "payments":
            for pay in desk.payments.payments.values():
                late = " LATE" if pay.late else ""
                who = f" (approved by {pay.approved_by})" if pay.approved_by else ""
                print(f"{pay.payment_id}  {pay.status:<17} {pay.vendor_name[:28]:<28} "
                      f"{pay.amount:>12,.2f} {pay.currency}  {pay.pay_on}{late}{who}")
        elif a.cmd == "approve":
            pay = desk.approve(a.payment_id, a.who)
            print(f"{pay.payment_id} approved by {pay.approved_by}; scheduled for {pay.pay_on}")
        elif a.cmd == "cancel":
            pay = desk.cancel(a.payment_id, a.who, a.reason)
            print(f"{pay.payment_id} cancelled")
    except (PaymentError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
