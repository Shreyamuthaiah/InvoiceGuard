"""Deterministic validation & matching checks run on an extracted `Invoice`.

Each check returns structured `Finding`s. Findings that map to an `Anomaly` drive the
decision (via `expected_decision`); the rest are informational warnings for a reviewer.
No LLM is involved: these rules are the part of the system that must be auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Literal, Optional

from pydantic import BaseModel

from invoice_agent.schema import (
    PO_TOLERANCE, Anomaly, Decision, Invoice, PurchaseOrder, Vendor, expected_decision,
)

NAME_MATCH_THRESHOLD = 0.92  # similarity needed to accept a vendor by name alone
LOOKALIKE_THRESHOLD = 0.80  # similar enough to warn about a possible impersonation
NEAR_DUPLICATE_DAYS = 7

_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|plc|gmbh|ag|kg|co|corp|corporation|company|group|and|und|stiftung|ohg|eg|"
    r"partners?|sons|limited|e\.?\s?v)\b\.?"
)


def norm_name(name: str) -> str:
    s = name.casefold().replace("&", " and ")
    s = _SUFFIXES.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_id(value: Optional[str]) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", value or "").upper()


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_name(a), norm_name(b)).ratio()


class Finding(BaseModel):
    check: str
    severity: Literal["info", "warning", "error"]
    message: str
    anomaly: Optional[Anomaly] = None


@dataclass
class VendorMatch:
    vendor: Optional[Vendor]
    method: Optional[Literal["tax_id", "name"]]
    score: float = 0.0


@dataclass
class CheckReport:
    vendor_match: VendorMatch
    findings: list[Finding]

    @property
    def anomalies(self) -> list[Anomaly]:
        seen: list[Anomaly] = []
        for f in self.findings:
            if f.anomaly and f.anomaly not in seen:
                seen.append(f.anomaly)
        return seen

    @property
    def decision(self) -> Decision:
        return expected_decision(self.anomalies)


# ---------- vendor master ----------

class VendorMaster:
    def __init__(self, vendors: list[Vendor]) -> None:
        self.vendors = vendors
        self.by_tax_id = {norm_id(v.tax_id): v for v in vendors}

    def resolve(self, inv: Invoice) -> tuple[VendorMatch, list[Finding]]:
        findings: list[Finding] = []
        tax = norm_id(inv.vendor_tax_id)
        by_tax = self.by_tax_id.get(tax) if tax else None
        best = max(self.vendors, key=lambda v: name_similarity(inv.vendor_name, v.name), default=None)
        best_score = name_similarity(inv.vendor_name, best.name) if best else 0.0

        if by_tax:
            score = name_similarity(inv.vendor_name, by_tax.name)
            if score < LOOKALIKE_THRESHOLD:
                findings.append(Finding(
                    check="vendor", severity="warning",
                    message=f"tax ID matches {by_tax.vendor_id} ({by_tax.name}) but the name on the "
                            f"invoice is {inv.vendor_name!r}"))
            return VendorMatch(by_tax, "tax_id", 1.0), findings

        # No tax-ID match. Accept by name only when the invoice shows no usable tax ID;
        # a *different* tax ID under a familiar name is a classic impersonation pattern.
        if best and best_score >= NAME_MATCH_THRESHOLD and not tax:
            findings.append(Finding(check="vendor", severity="info",
                                    message=f"matched {best.vendor_id} by name only (no tax ID on invoice)"))
            return VendorMatch(best, "name", best_score), findings

        if best and best_score >= LOOKALIKE_THRESHOLD:
            findings.append(Finding(
                check="vendor", severity="warning",
                message=f"name resembles {best.vendor_id} ({best.name}, similarity {best_score:.2f}) "
                        f"but tax ID {inv.vendor_tax_id!r} is not on file: possible look-alike vendor"))
        findings.append(Finding(check="vendor", severity="error", anomaly=Anomaly.UNKNOWN_VENDOR,
                                message=f"vendor {inv.vendor_name!r} is not in the vendor master"))
        return VendorMatch(None, None, best_score), findings


# ---------- individual checks ----------

def check_po(inv: Invoice, vendor: Optional[Vendor], pos: dict[str, PurchaseOrder]) -> list[Finding]:
    if not inv.po_number:
        return [Finding(check="po", severity="error", anomaly=Anomaly.MISSING_PO,
                        message="invoice does not reference a purchase order")]
    po = pos.get(inv.po_number.strip().upper())
    if po is None:
        return [Finding(check="po", severity="error", anomaly=Anomaly.MISSING_PO,
                        message=f"PO {inv.po_number} does not exist")]
    out: list[Finding] = []
    if vendor and po.vendor_id != vendor.vendor_id:
        out.append(Finding(check="po", severity="error", anomaly=Anomaly.MISSING_PO,
                           message=f"PO {po.po_number} belongs to {po.vendor_id}, not {vendor.vendor_id}"))
    if po.status != "open":
        out.append(Finding(check="po", severity="warning", message=f"PO {po.po_number} is {po.status}"))
    if po.currency != inv.currency:
        out.append(Finding(check="po", severity="error", anomaly=Anomaly.PO_AMOUNT_MISMATCH,
                           message=f"currency {inv.currency} differs from PO currency {po.currency}"))
    limit = po.total * (1 + PO_TOLERANCE)
    if inv.subtotal > limit:
        over = (inv.subtotal / po.total - 1) * 100 if po.total else Decimal("100")
        out.append(Finding(check="po", severity="error", anomaly=Anomaly.PO_AMOUNT_MISMATCH,
                           message=f"subtotal {inv.subtotal} exceeds PO total {po.total} by {over:.1f}% "
                                   f"(tolerance {PO_TOLERANCE * 100:.0f}%)"))
    elif inv.subtotal < po.total:
        out.append(Finding(check="po", severity="info",
                           message=f"subtotal {inv.subtotal} below PO total {po.total} (partial billing?)"))
    return out


def check_bank(inv: Invoice, vendor: Optional[Vendor]) -> list[Finding]:
    if vendor is None:
        return []
    if not inv.remit_to.same_destination(vendor.bank_account):
        return [Finding(check="bank", severity="error", anomaly=Anomaly.BANK_DETAILS_CHANGED,
                        message="remit-to bank account differs from the vendor master; verify by phone "
                                "using contact details on file before paying")]
    return []


def check_arithmetic(inv: Invoice) -> list[Finding]:
    return [Finding(check="arithmetic", severity="error", anomaly=Anomaly.ARITHMETIC_ERROR, message=m)
            for m in inv.arithmetic_issues()]


@dataclass
class _Seen:
    vendor_key: str
    invoice_number: str
    total: Decimal
    invoice_date: date
    ref: str


@dataclass
class InvoiceLedger:
    """Invoices already received, for duplicate detection."""
    entries: list[_Seen] = field(default_factory=list)

    @staticmethod
    def vendor_key(inv: Invoice, vendor: Optional[Vendor]) -> str:
        return vendor.vendor_id if vendor else f"name:{norm_name(inv.vendor_name)}"

    def check(self, inv: Invoice, vendor: Optional[Vendor]) -> list[Finding]:
        key, number = self.vendor_key(inv, vendor), norm_id(inv.invoice_number)
        out: list[Finding] = []
        for e in self.entries:
            if e.vendor_key != key:
                continue
            if e.invoice_number == number:
                return [Finding(check="duplicate", severity="error", anomaly=Anomaly.DUPLICATE,
                                message=f"invoice {inv.invoice_number} from this vendor was already "
                                        f"received ({e.ref})")]
            if e.total == inv.total and abs((e.invoice_date - inv.invoice_date).days) <= NEAR_DUPLICATE_DAYS:
                out.append(Finding(check="duplicate", severity="warning",
                                   message=f"same vendor and total {inv.total} within {NEAR_DUPLICATE_DAYS} "
                                           f"days as {e.ref} under a different number"))
        return out

    def add(self, inv: Invoice, vendor: Optional[Vendor], ref: str) -> None:
        self.entries.append(_Seen(self.vendor_key(inv, vendor), norm_id(inv.invoice_number),
                                  inv.total, inv.invoice_date, ref))


# ---------- orchestration ----------

class InvoiceChecker:
    def __init__(self, vendors: list[Vendor], purchase_orders: list[PurchaseOrder]) -> None:
        self.master = VendorMaster(vendors)
        self.pos = {po.po_number.upper(): po for po in purchase_orders}
        self.ledger = InvoiceLedger()

    def check(self, inv: Invoice, ref: str = "", record: bool = True) -> CheckReport:
        match, findings = self.master.resolve(inv)
        vendor = match.vendor
        findings += self.ledger.check(inv, vendor)
        findings += check_po(inv, vendor, self.pos)
        findings += check_bank(inv, vendor)
        findings += check_arithmetic(inv)
        report = CheckReport(match, findings)
        if record and Anomaly.DUPLICATE not in report.anomalies:
            self.ledger.add(inv, vendor, ref)
        return report
