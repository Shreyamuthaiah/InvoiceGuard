"""LLM-facing extraction schema.

The model copies dates, amounts and rates *verbatim as printed* (e.g. "2.093,04 €",
"03/04/2026", "(8.75%)"). Deterministic Python in `normalize.py` turns them into typed
values. This keeps locale handling testable and stops the model from "fixing" arithmetic
it should be reporting.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = Field(None, description="State / county / region, if printed.")
    postal_code: Optional[str] = None
    country: Optional[str] = None


class ExtractedLineItem(BaseModel):
    description: str
    quantity: str = Field(description="Quantity exactly as printed.")
    unit_price: str = Field(description="Unit price exactly as printed, including currency symbol.")
    amount: str = Field(description="Line total exactly as printed, including currency symbol.")


class ExtractedBank(BaseModel):
    bank_name: Optional[str] = None
    account_name: Optional[str] = None
    account_number: Optional[str] = None
    routing_number: Optional[str] = Field(None, description="US ABA routing number.")
    iban: Optional[str] = None
    swift: Optional[str] = Field(None, description="SWIFT / BIC code.")


class ExtractedInvoice(BaseModel):
    invoice_number: str
    invoice_date: str = Field(description="Invoice/issue date exactly as printed.")
    due_date: Optional[str] = Field(None, description="Due date exactly as printed.")
    po_number: Optional[str] = Field(None, description="Purchase order / customer reference, if printed.")
    vendor_name: str = Field(description="The company issuing the invoice (the seller).")
    vendor_address: ExtractedAddress
    vendor_tax_id: Optional[str] = None
    vendor_email: Optional[str] = None
    bill_to_name: Optional[str] = Field(None, description="The customer being billed.")
    bill_to_address: ExtractedAddress
    currency: str = Field(description="ISO 4217 code implied by the symbols used, e.g. USD, EUR, GBP.")
    line_items: list[ExtractedLineItem]
    subtotal: str = Field(description="Pre-tax subtotal exactly as printed.")
    tax_rate: Optional[str] = Field(None, description="Tax/VAT rate exactly as printed, e.g. '8.75%'.")
    tax_amount: str = Field(description="Tax/VAT amount exactly as printed.")
    total: str = Field(description="Total / amount due exactly as printed.")
    payment_terms: Optional[str] = Field(None, description="e.g. 'Net 30'.")
    remit_to: ExtractedBank = Field(description="Bank details the vendor asks to be paid to.")
