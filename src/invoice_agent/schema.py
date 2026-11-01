"""Domain schema shared by the synthetic generator, the extractor and the agent.

`Invoice` describes exactly what is *printed* on a document — it is the extraction target.
Errors on the document (wrong totals, changed bank details, ...) are captured as
`Anomaly` labels in `GroundTruth`, not "corrected" in the invoice fields.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

CENT = Decimal("0.01")
# Invoice subtotal may exceed the PO total by at most this fraction before it is a mismatch.
PO_TOLERANCE = Decimal("0.02")
# Invoices with a total above this need a named human approver before payment.
APPROVAL_THRESHOLDS = {"USD": Decimal("10000"), "EUR": Decimal("10000"), "GBP": Decimal("8000")}


def money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


class Address(BaseModel):
    street: str
    city: str
    region: str
    postal_code: str
    country: str

    def lines(self) -> list[str]:
        return [self.street, f"{self.city}, {self.region} {self.postal_code}".strip(), self.country]


class BankAccount(BaseModel):
    bank_name: str
    account_name: str
    account_number: Optional[str] = None  # US accounts
    routing_number: Optional[str] = None  # US ABA routing
    iban: Optional[str] = None  # EU/UK accounts
    swift: Optional[str] = None

    def same_destination(self, other: "BankAccount") -> bool:
        """True if both describe the same destination account (ignores display names)."""
        return (self.account_number, self.routing_number, self.iban) == (
            other.account_number,
            other.routing_number,
            other.iban,
        )


class Vendor(BaseModel):
    vendor_id: str
    name: str
    category: str
    address: Address
    tax_id: str
    email: str
    currency: str
    payment_terms_days: int
    bank_account: BankAccount


class LineItem(BaseModel):
    description: str
    quantity: Decimal
    unit_price: Decimal
    amount: Decimal


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_id: str
    issue_date: date
    currency: str
    lines: list[LineItem]
    total: Decimal = Field(description="Pre-tax total; compared against invoice subtotal.")
    status: Literal["open", "closed"] = "open"


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date
    due_date: date
    po_number: Optional[str] = None

    vendor_name: str
    vendor_address: Address
    vendor_tax_id: str
    vendor_email: str

    bill_to_name: str
    bill_to_address: Address

    currency: str
    line_items: list[LineItem]
    subtotal: Decimal
    tax_rate: Decimal = Field(description="Fraction, e.g. 0.0725 for 7.25%.")
    tax_amount: Decimal
    total: Decimal
    payment_terms: str
    remit_to: BankAccount

    def arithmetic_issues(self) -> list[str]:
        issues: list[str] = []
        for i, li in enumerate(self.line_items, 1):
            if money(li.quantity * li.unit_price) != li.amount:
                issues.append(f"line {i}: qty x unit price != amount")
        line_sum = sum((li.amount for li in self.line_items), Decimal("0"))
        if line_sum != self.subtotal:
            issues.append(f"line sum {line_sum} != subtotal {self.subtotal}")
        if money(self.subtotal * self.tax_rate) != self.tax_amount:
            issues.append(f"tax {self.tax_amount} != subtotal x rate")
        if self.subtotal + self.tax_amount != self.total:
            issues.append(f"subtotal + tax != total {self.total}")
        return issues


class Anomaly(str, Enum):
    DUPLICATE = "duplicate"
    PO_AMOUNT_MISMATCH = "po_amount_mismatch"
    MISSING_PO = "missing_po"
    UNKNOWN_VENDOR = "unknown_vendor"
    BANK_DETAILS_CHANGED = "bank_details_changed"
    ARITHMETIC_ERROR = "arithmetic_error"
    SUSPICIOUS_INSTRUCTIONS = "suspicious_instructions"  # text addressed to an automated system
    SPLIT_INVOICE = "split_invoice"  # one purchase billed in pieces to dodge the approval threshold


class Decision(str, Enum):
    APPROVE = "approve"
    HOLD = "hold"
    REJECT = "reject"


def expected_decision(anomalies: list[Anomaly]) -> Decision:
    """Reference policy: duplicates are rejected, any other finding goes to a human."""
    if Anomaly.DUPLICATE in anomalies:
        return Decision.REJECT
    if anomalies:
        return Decision.HOLD
    return Decision.APPROVE


class Style(BaseModel):
    """A vendor's visual variation of a base layout (see synth/render.py)."""
    template: str
    font: str = "Helvetica"
    accent: str = "#1F4E5F"
    columns: list[str] = Field(default_factory=lambda: ["desc", "qty", "unit", "amount"])
    labels: dict[str, str]
    date_fmt: str
    money: str = "us"  # "us" -> $1,234.56 ; "eu" -> 1.234,56 EUR-style
    upper_labels: bool = False
    vendor_right: bool = False
    bank_beside_totals: bool = False


class Injection(BaseModel):
    """Text printed on the document that is not invoice data (adversarial cases)."""
    text: str
    mode: Literal["note", "hidden"]  # visible notes block | white 1pt text (invisible on paper)


class RenderInfo(BaseModel):
    template: str
    file: str  # relative to the dataset root
    format: Literal["pdf", "jpg"]
    scanned: bool
    degradations: dict[str, float] = Field(default_factory=dict)
    style: Optional[Style] = None  # None = the template's fixed v1 look
    injections: list[Injection] = Field(default_factory=list)


class GroundTruth(BaseModel):
    doc_id: str
    invoice: Invoice
    vendor_id: Optional[str] = Field(description="True vendor-master id; None if unknown vendor.")
    anomalies: list[Anomaly]
    expected_decision: Decision
    duplicate_of: Optional[str] = None
    render: RenderInfo
    notes: list[str] = Field(default_factory=list)
