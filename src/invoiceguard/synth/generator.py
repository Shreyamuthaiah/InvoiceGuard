"""Generates vendors, purchase orders and labelled invoice cases (no rendering here)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

from faker import Faker

from invoiceguard.schema import (
    PO_TOLERANCE,
    Address,
    Anomaly,
    BankAccount,
    GroundTruth,
    Invoice,
    LineItem,
    PurchaseOrder,
    RenderInfo,
    Style,
    Vendor,
    expected_decision,
    money,
)
from invoiceguard.synth import catalog
from invoiceguard.synth.render import make_style

# Probability of each case type; "clean" means no anomaly.
DEFAULT_WEIGHTS: dict[str, float] = {
    "clean": 0.55,
    Anomaly.DUPLICATE.value: 0.08,
    Anomaly.PO_AMOUNT_MISMATCH.value: 0.08,
    Anomaly.MISSING_PO.value: 0.08,
    Anomaly.UNKNOWN_VENDOR.value: 0.06,
    Anomaly.BANK_DETAILS_CHANGED.value: 0.08,
    Anomaly.ARITHMETIC_ERROR.value: 0.07,
}

START_DATE = date(2026, 1, 5)
DATE_SPAN_DAYS = 280


@dataclass
class _VendorState:
    vendor: Vendor
    template: str
    style: Style | None
    pattern: str
    prefix: str
    next_n: int


@dataclass
class Dataset:
    seed: int
    vendors: list[Vendor]
    purchase_orders: list[PurchaseOrder]
    cases: list[GroundTruth] = field(default_factory=list)


class DatasetGenerator:
    def __init__(
        self,
        seed: int = 42,
        n_vendors: int = 25,
        scan_ratio: float = 0.3,
        weights: dict[str, float] | None = None,
        layout_variety: bool = True,
    ) -> None:
        self.seed = seed
        self.n_vendors = n_vendors
        self.scan_ratio = scan_ratio
        self.layout_variety = layout_variety
        self.weights = weights or DEFAULT_WEIGHTS
        self.rng = random.Random(seed)
        self.fakers: dict[str, Faker] = {}
        for i, (cur, cfg) in enumerate(catalog.CURRENCIES.items()):
            f = Faker(cfg["locale"])
            f.seed_instance(seed * 1000 + i)
            self.fakers[cur] = f
        self._used_names: set[str] = set()
        self._po_seq = 100
        self._vendor_seq = 0

    # ---------- vendors ----------

    def _address(self, cur: str) -> Address:
        f = self.fakers[cur]
        if cur == "USD":
            region = f.state_abbr()
        else:
            try:
                region = f.administrative_unit()
            except AttributeError:
                region = ""
        return Address(
            street=f.street_address(),
            city=f.city(),
            region=region,
            postal_code=f.postcode(),
            country=str(catalog.CURRENCIES[cur]["country"]),
        )

    def _bank(self, cur: str, account_name: str) -> BankAccount:
        f = self.fakers[cur]
        bank_name = self.rng.choice(catalog.CURRENCIES[cur]["banks"])
        if cur == "USD":
            return BankAccount(
                bank_name=bank_name,
                account_name=account_name,
                account_number=f.numerify("#" * self.rng.choice([9, 10, 12])),
                routing_number=f.aba(),
            )
        return BankAccount(bank_name=bank_name, account_name=account_name, iban=f.iban(), swift=f.swift())

    def _tax_id(self, cur: str) -> str:
        f = self.fakers[cur]
        if cur == "USD":
            return f.numerify("##-#######")  # EIN format
        if cur == "EUR":
            return "DE" + f.numerify("#########")
        return "GB" + f.numerify("### #### ##")

    def _company_name(self, cur: str) -> str:
        f = self.fakers[cur]
        for _ in range(50):
            name = f.company()
            if name not in self._used_names:
                self._used_names.add(name)
                return name
        raise RuntimeError("could not generate a unique company name")

    def _make_vendor(self, register: bool = True) -> _VendorState:
        cur = self.rng.choices(list(catalog.CURRENCY_WEIGHTS), weights=list(catalog.CURRENCY_WEIGHTS.values()))[0]
        name = self._company_name(cur)
        f = self.fakers[cur]
        domain = "".join(ch for ch in name.lower() if ch.isascii() and ch.isalnum())[:18] + ".example"
        if register:
            self._vendor_seq += 1
            vendor_id = f"V{self._vendor_seq:04d}"
        else:
            vendor_id = f"UNREGISTERED-{f.numerify('####')}"
        vendor = Vendor(
            vendor_id=vendor_id,
            name=name,
            category=self.rng.choice(list(catalog.CATALOG)),
            address=self._address(cur),
            tax_id=self._tax_id(cur),
            email=f"{self.rng.choice(['billing', 'accounts', 'ar', 'invoices'])}@{domain}",
            currency=cur,
            payment_terms_days=self.rng.choice([15, 30, 30, 45, 60]),
            bank_account=self._bank(cur, name),
        )
        initials = "".join(w[0] for w in name.replace("-", " ").split() if w[0].isalpha()).upper()[:3] or "INV"
        # Same main-RNG call order as dataset v1, so invoice content is unchanged.
        template = self.rng.choice(catalog.TEMPLATES)
        pattern = self.rng.choice(catalog.INVOICE_NUMBER_PATTERNS)
        next_n = self.rng.randint(100, 9000)
        return _VendorState(vendor=vendor, template=template, style=self._style(template, cur, vendor_id),
                            pattern=pattern, prefix=initials, next_n=next_n)

    def _style(self, template: str, currency: str, key: str) -> Style | None:
        """Visual style from its own RNG: changing styles never changes invoice content."""
        if not self.layout_variety:
            return None
        return make_style(random.Random(f"{self.seed}:style:{key}"), template, currency)

    # ---------- documents ----------

    def _line_items(self, category: str) -> list[LineItem]:
        options = catalog.CATALOG[category]
        picks = self.rng.sample(options, k=self.rng.randint(1, min(len(options), 5)))
        items = []
        for desc, (pmin, pmax), (qmin, qmax) in picks:
            raw = self.rng.uniform(pmin, pmax)
            unit = Decimal(str(round(raw, 4 if pmax < 1 else 2)))
            if "hours" in desc and qmax > 1:
                qty = Decimal(self.rng.randint(qmin * 2, qmax * 2)) / 2
            else:
                qty = Decimal(self.rng.randint(qmin, qmax))
            items.append(LineItem(description=desc, quantity=qty, unit_price=unit, amount=money(qty * unit)))
        return items

    def _invoice_number(self, vs: _VendorState, inv_date: date) -> str:
        n = vs.next_n
        vs.next_n += self.rng.randint(1, 40)  # vendors invoice other customers in between
        return vs.pattern.format(year=inv_date.year, yy=f"{inv_date.year % 100:02d}", n=n, prefix=vs.prefix)

    def _new_po_number(self) -> str:
        self._po_seq += self.rng.randint(1, 7)
        return f"PO-26{self._po_seq:05d}"

    def _build_invoice(
        self, vs: _VendorState, items: list[LineItem], inv_date: date, po_number: str | None,
        remit_to: BankAccount,
    ) -> Invoice:
        v = vs.vendor
        tax_rate = Decimal(self.rng.choice(catalog.CURRENCIES[v.currency]["tax_rates"]))
        subtotal = sum((li.amount for li in items), Decimal("0"))
        tax = money(subtotal * tax_rate)
        return Invoice(
            invoice_number=self._invoice_number(vs, inv_date),
            invoice_date=inv_date,
            due_date=inv_date + timedelta(days=v.payment_terms_days),
            po_number=po_number,
            vendor_name=v.name,
            vendor_address=v.address,
            vendor_tax_id=v.tax_id,
            vendor_email=v.email,
            bill_to_name=catalog.BUYER_NAME,
            bill_to_address=Address(**catalog.BUYER_ADDRESS),
            currency=v.currency,
            line_items=items,
            subtotal=subtotal,
            tax_rate=tax_rate,
            tax_amount=tax,
            total=subtotal + tax,
            payment_terms=f"Net {v.payment_terms_days}",
            remit_to=remit_to,
        )

    def _render_info(self, doc_id: str, template: str, style: Style | None = None) -> RenderInfo:
        scanned = self.rng.random() < self.scan_ratio
        fmt: Literal["pdf", "jpg"] = "jpg" if scanned else "pdf"
        return RenderInfo(template=template, file=f"docs/{doc_id}.{fmt}", format=fmt, scanned=scanned,
                          style=style)

    def generate(self, n: int) -> Dataset:
        states = [self._make_vendor() for _ in range(self.n_vendors)]
        ds = Dataset(seed=self.seed, vendors=[s.vendor for s in states], purchase_orders=[])
        clean_cases: list[GroundTruth] = []
        kinds, kind_weights = list(self.weights), list(self.weights.values())

        for i in range(1, n + 1):
            doc_id = f"doc_{i:04d}"
            kind = self.rng.choices(kinds, weights=kind_weights)[0]
            if kind == Anomaly.DUPLICATE.value and not clean_cases:
                kind = "clean"

            if kind == Anomaly.DUPLICATE.value:
                orig = self.rng.choice(clean_cases)
                dup_anomalies = [Anomaly.DUPLICATE]
                ds.cases.append(GroundTruth(
                    doc_id=doc_id,
                    invoice=orig.invoice.model_copy(deep=True),
                    vendor_id=orig.vendor_id,
                    anomalies=dup_anomalies,
                    expected_decision=expected_decision(dup_anomalies),
                    duplicate_of=orig.doc_id,
                    render=self._render_info(doc_id, orig.render.template, orig.render.style),
                    notes=[f"Re-sent copy of {orig.doc_id} (same vendor + invoice number)."],
                ))
                continue

            notes: list[str] = []
            anomalies: list[Anomaly] = []
            if kind == Anomaly.UNKNOWN_VENDOR.value:
                vs = self._make_vendor(register=False)
                anomalies = [Anomaly.UNKNOWN_VENDOR, Anomaly.MISSING_PO]
                notes.append("Vendor not in vendor master; no PO could have been raised.")
            else:
                vs = self.rng.choice(states)

            v = vs.vendor
            inv_date = START_DATE + timedelta(days=self.rng.randint(0, DATE_SPAN_DAYS))
            items = self._line_items(v.category)

            po_number = None
            if kind not in (Anomaly.UNKNOWN_VENDOR.value, Anomaly.MISSING_PO.value):
                po_lines = [li.model_copy() for li in items]
                po = PurchaseOrder(
                    po_number=self._new_po_number(),
                    vendor_id=v.vendor_id,
                    issue_date=inv_date - timedelta(days=self.rng.randint(3, 45)),
                    currency=v.currency,
                    lines=po_lines,
                    total=sum((li.amount for li in po_lines), Decimal("0")),
                )
                ds.purchase_orders.append(po)
                po_number = po.po_number
            if kind == Anomaly.MISSING_PO.value:
                anomalies = [Anomaly.MISSING_PO]
                notes.append("Invoice references no PO.")

            if kind == Anomaly.PO_AMOUNT_MISMATCH.value:
                anomalies = [Anomaly.PO_AMOUNT_MISMATCH]
                po_total = ds.purchase_orders[-1].total
                idx = max(range(len(items)), key=lambda k: items[k].amount)
                base = items[idx]
                factor = Decimal(str(round(self.rng.uniform(1.08, 1.30), 3)))
                while True:
                    scaled = base.unit_price * factor
                    unit = money(scaled) if base.unit_price >= 1 else scaled.quantize(Decimal("0.0001"))
                    new = base.model_copy(update={"unit_price": unit, "amount": money(base.quantity * unit)})
                    new_sub = sum((li.amount for li in items), Decimal("0")) - base.amount + new.amount
                    if new_sub > po_total * (1 + PO_TOLERANCE + Decimal("0.01")):
                        break
                    factor += Decimal("0.1")
                items[idx] = new
                notes.append(
                    f"Line {idx + 1} unit price {base.unit_price} -> {unit} vs PO; "
                    f"subtotal {new_sub} vs PO total {po_total}."
                )

            remit_to = v.bank_account
            if kind == Anomaly.BANK_DETAILS_CHANGED.value:
                anomalies = [Anomaly.BANK_DETAILS_CHANGED]
                remit_to = self._bank(v.currency, v.name)
                notes.append("Remit-to account differs from vendor master (possible BEC fraud).")

            inv = self._build_invoice(vs, items, inv_date, po_number, remit_to)

            if kind == Anomaly.ARITHMETIC_ERROR.value:
                anomalies = [Anomaly.ARITHMETIC_ERROR]
                delta = money(self.rng.uniform(5, 250)) * self.rng.choice([-1, 1])
                if inv.subtotal + delta <= 0:
                    delta = abs(delta)
                if self.rng.random() < 0.5:
                    sub = inv.subtotal + delta
                    tax = money(sub * inv.tax_rate)
                    inv = inv.model_copy(update={"subtotal": sub, "tax_amount": tax, "total": sub + tax})
                    notes.append(f"Printed subtotal off by {delta} from the sum of line items.")
                    # Labels list every check that should fire: an inflated subtotal can also
                    # break the PO tolerance.
                    po_total = ds.purchase_orders[-1].total
                    if po_number and sub > po_total * (1 + PO_TOLERANCE):
                        anomalies.append(Anomaly.PO_AMOUNT_MISMATCH)
                        notes.append(f"Inflated subtotal {sub} also exceeds PO total {po_total} beyond tolerance.")
                else:
                    inv = inv.model_copy(update={"total": inv.total + delta})
                    notes.append(f"Printed total off by {delta} from subtotal + tax.")

            unknown = anomalies == [Anomaly.UNKNOWN_VENDOR, Anomaly.MISSING_PO]
            template = self.rng.choice(catalog.TEMPLATES) if unknown else vs.template
            gt = GroundTruth(
                doc_id=doc_id,
                invoice=inv,
                vendor_id=v.vendor_id if kind != Anomaly.UNKNOWN_VENDOR.value else None,
                anomalies=anomalies,
                expected_decision=expected_decision(anomalies),
                render=self._render_info(
                    doc_id, template,
                    vs.style if template == vs.template else self._style(template, v.currency, doc_id)),
                notes=notes,
            )
            ds.cases.append(gt)
            if not anomalies:
                clean_cases.append(gt)
        return ds
