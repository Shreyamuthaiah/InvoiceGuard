"""LLM-facing extraction schema.

The model copies dates, amounts and rates *verbatim as printed* (e.g. "2.093,04 €",
"03/04/2026", "(8.75%)"). Deterministic Python in `normalize.py` turns them into typed
values. This keeps locale handling testable and stops the model from "fixing" arithmetic
it should be reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedAddress(BaseModel):
    street: str | None = None
    city: str | None = None
    region: str | None = Field(None, description="State / county / region, if printed.")
    postal_code: str | None = None
    country: str | None = None


class ExtractedLineItem(BaseModel):
    description: str
    quantity: str = Field(description="Quantity exactly as printed.")
    unit_price: str = Field(description="Unit price exactly as printed, including currency symbol.")
    amount: str = Field(description="Line total exactly as printed, including currency symbol.")


class ExtractedBank(BaseModel):
    bank_name: str | None = None
    account_name: str | None = None
    account_number: str | None = None
    routing_number: str | None = Field(None, description="US ABA routing number.")
    iban: str | None = None
    swift: str | None = Field(None, description="SWIFT / BIC code.")


class ExtractedInvoice(BaseModel):
    invoice_number: str
    invoice_date: str = Field(description="Invoice/issue date exactly as printed.")
    due_date: str | None = Field(None, description="Due date exactly as printed.")
    po_number: str | None = Field(None, description="Purchase order / customer reference, if printed.")
    vendor_name: str = Field(description="The company issuing the invoice (the seller).")
    vendor_address: ExtractedAddress
    vendor_tax_id: str | None = None
    vendor_email: str | None = None
    bill_to_name: str | None = Field(None, description="The customer being billed.")
    bill_to_address: ExtractedAddress
    currency: str = Field(description="ISO 4217 code implied by the symbols used, e.g. USD, EUR, GBP.")
    line_items: list[ExtractedLineItem]
    subtotal: str = Field(description="Pre-tax subtotal exactly as printed.")
    tax_rate: str | None = Field(None, description="Tax/VAT rate exactly as printed, e.g. '8.75%'.")
    tax_amount: str = Field(description="Tax/VAT amount exactly as printed.")
    total: str = Field(description="Total / amount due exactly as printed.")
    payment_terms: str | None = Field(None, description="e.g. 'Net 30'.")
    remit_to: ExtractedBank = Field(description="Bank details the vendor asks to be paid to.")
