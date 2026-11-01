"""Mock payments service with the controls a real AP payment run needs.

Controls (enforced here, independently of the checks that came before):
- Only invoices whose decision is APPROVE can be scheduled.
- Money goes to the bank account in the vendor master, never to one printed on the invoice.
- Idempotent: the same invoice (vendor + invoice number + amount) maps to one payment.
- Amounts above a per-currency threshold wait for a named human approver.
- Payment date: `lead_days` before the due date, never in the past.
State is persisted to a JSON file so re-running a batch cannot double-pay.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from invoice_agent.checks import norm_id
from invoice_agent.schema import BankAccount, Decision, Invoice, Vendor


class PaymentError(Exception):
    pass


class PaymentPolicy(BaseModel):
    approval_threshold: dict[str, Decimal] = Field(
        default_factory=lambda: {"USD": Decimal("10000"), "EUR": Decimal("10000"), "GBP": Decimal("8000")})
    lead_days: int = 2


class Payment(BaseModel):
    payment_id: str
    idempotency_key: str
    vendor_id: str
    vendor_name: str
    invoice_number: str
    amount: Decimal
    currency: str
    pay_on: date
    late: bool
    destination: BankAccount
    status: Literal["scheduled", "awaiting_approval", "cancelled"]
    approved_by: Optional[str] = None
    created_at: str
    source_ref: str = ""


def idempotency_key(vendor_id: str, invoice_number: str, amount: Decimal, currency: str) -> str:
    raw = f"{vendor_id}|{norm_id(invoice_number)}|{amount:.2f}|{currency.upper()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class MockPaymentsAPI:
    def __init__(self, store: Path | None = None, policy: PaymentPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or PaymentPolicy()
        self.payments: dict[str, Payment] = {}
        if store and store.exists():
            for p in json.loads(store.read_text(encoding="utf-8")):
                pay = Payment.model_validate(p)
                self.payments[pay.idempotency_key] = pay

    def _save(self) -> None:
        if self.store:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            data = [p.model_dump(mode="json") for p in self.payments.values()]
            self.store.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def schedule(self, inv: Invoice, vendor: Vendor, decision: Decision, today: date,
                 source_ref: str = "") -> tuple[Payment, bool]:
        """Returns (payment, created). created=False means an existing payment was returned."""
        if decision != Decision.APPROVE:
            raise PaymentError(f"refusing to pay: decision is {decision.value}")
        if inv.currency != vendor.currency:
            raise PaymentError(f"invoice currency {inv.currency} differs from vendor currency {vendor.currency}")
        if inv.total <= 0:
            raise PaymentError(f"refusing to pay non-positive amount {inv.total}")

        key = idempotency_key(vendor.vendor_id, inv.invoice_number, inv.total, inv.currency)
        if key in self.payments:
            return self.payments[key], False
        for p in self.payments.values():  # same invoice, different amount: never silently pay twice
            if p.vendor_id == vendor.vendor_id and norm_id(p.invoice_number) == norm_id(inv.invoice_number) \
                    and p.status != "cancelled":
                raise PaymentError(f"invoice {inv.invoice_number} already has payment {p.payment_id} "
                                   f"for {p.amount} {p.currency}")

        pay_on = inv.due_date - timedelta(days=self.policy.lead_days)
        late = inv.due_date < today
        pay_on = max(pay_on, today)
        threshold = self.policy.approval_threshold.get(inv.currency)
        needs_approval = threshold is None or inv.total > threshold
        payment = Payment(
            payment_id=f"PAY-{len(self.payments) + 1:05d}",
            idempotency_key=key,
            vendor_id=vendor.vendor_id,
            vendor_name=vendor.name,
            invoice_number=inv.invoice_number,
            amount=inv.total,
            currency=inv.currency,
            pay_on=pay_on,
            late=late,
            destination=vendor.bank_account,  # vendor master, never the invoice's remit-to
            status="awaiting_approval" if needs_approval else "scheduled",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_ref=source_ref,
        )
        self.payments[key] = payment
        self._save()
        return payment, True

    def approve(self, payment_id: str, approver: str) -> Payment:
        if not approver.strip():
            raise PaymentError("an approver name is required")
        p = self._get(payment_id)
        if p.status != "awaiting_approval":
            raise PaymentError(f"{payment_id} is {p.status}, not awaiting approval")
        p.status, p.approved_by = "scheduled", approver.strip()
        self._save()
        return p

    def cancel(self, payment_id: str) -> Payment:
        p = self._get(payment_id)
        p.status = "cancelled"
        self._save()
        return p

    def _get(self, payment_id: str) -> Payment:
        for p in self.payments.values():
            if p.payment_id == payment_id:
                return p
        raise PaymentError(f"unknown payment {payment_id}")
