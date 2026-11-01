"""Deterministic normalisation of verbatim strings into the typed `Invoice` schema."""

from __future__ import annotations

import contextlib
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from invoice_agent.extract.models import ExtractedAddress, ExtractedBank, ExtractedInvoice
from invoice_agent.schema import Address, BankAccount, Invoice, LineItem, money


class NormalizationError(ValueError):
    pass


# ---------- numbers ----------

_NUM_CHARS = re.compile(r"[^0-9,.\-()]")


def parse_number(text: str) -> Decimal:
    """Parse '1,234.56', '1.234,56 €', '$0.0450', '(12.00)', '9.5' etc.

    Separator rule: if both ',' and '.' appear, the right-most one is the decimal mark.
    If only one kind appears, it is a decimal mark unless it is followed by exactly three
    digits (then it's a thousands separator), or it appears more than once.
    """
    if text is None:
        raise NormalizationError("missing number")
    raw = text.strip()
    neg = raw.startswith("-") or (raw.startswith("(") and raw.endswith(")"))
    s = _NUM_CHARS.sub("", raw).replace("(", "").replace(")", "").replace("-", "")
    if not s or not re.search(r"\d", s):
        raise NormalizationError(f"no digits in {text!r}")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:
            s = s.replace(sep, "")
        else:
            s = s.replace(sep, ".")
    try:
        value = Decimal(s)
    except InvalidOperation as e:
        raise NormalizationError(f"bad number {text!r}") from e
    return -value if neg else value


def parse_money(text: str) -> Decimal:
    return money(parse_number(text))


def parse_rate(text: str | None) -> Decimal | None:
    if not text:
        return None
    value = abs(parse_number(text.replace("(", " ").replace(")", " ")))  # "(19%)" is not negative
    if "%" in text or value > 1:
        value = value / 100
    return value.normalize() if value != 0 else Decimal("0")


def parse_terms_days(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d{1,3})", text)
    return int(m.group(1)) if m else None


# ---------- dates ----------

_TEXT_FORMATS = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%d.%m.%Y"]
_NUMERIC = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")


def date_candidates(text: str) -> dict[str, date]:
    """Possible readings of a printed date, keyed by convention ('mdy', 'dmy' or 'text')."""
    t = re.sub(r"\bSept\b", "Sep", text.strip())
    for fmt in _TEXT_FORMATS:
        try:
            return {"text": datetime.strptime(t, fmt).date()}
        except ValueError:
            pass
    m = _NUMERIC.match(t)
    if not m:
        raise NormalizationError(f"unrecognised date {text!r}")
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    out = {}
    for conv, (mo, d) in {"mdy": (a, b), "dmy": (b, a)}.items():
        with contextlib.suppress(ValueError):
            out[conv] = date(y, mo, d)
    if not out:
        raise NormalizationError(f"impossible date {text!r}")
    return out


def resolve_dates(
    invoice_date: str, due_date: str | None, terms_days: int | None, country: str | None
) -> tuple[date, date | None, list[str]]:
    """Pick one convention for both dates on a document.

    Order of evidence: a convention that is the only valid reading > the convention whose
    due - issue gap equals the payment terms > the vendor country (US -> m/d/y, else d/m/y).
    """
    warnings: list[str] = []
    inv_c = date_candidates(invoice_date)
    due_c = date_candidates(due_date) if due_date else {}
    convs: list[str] = [c for c in ("text", "mdy", "dmy")
                        if c in inv_c and (not due_c or c in due_c or "text" in due_c)]
    if not convs:
        convs = [str(k) for k in inv_c]

    def pick(conv: str) -> tuple[date, date | None]:
        due = None
        if due_c:
            due = due_c.get(conv) or due_c.get("text") or next(iter(due_c.values()))
        return inv_c[conv], due

    if len(convs) == 1:
        return (*pick(convs[0]), warnings)
    if terms_days is not None and due_c:
        def gap(conv: str) -> int | None:
            issued, due = pick(conv)
            return (due - issued).days if due else None
        matching = [c for c in convs if gap(c) == terms_days]
        if len(matching) == 1:
            return (*pick(matching[0]), warnings)
    us = (country or "").strip().lower() in {"united states", "usa", "us", "u.s.", "united states of america"}
    conv = "mdy" if us else "dmy"
    warnings.append(f"ambiguous date {invoice_date!r}: assumed {conv} from country {country!r}")
    return (*pick(conv), warnings)


# ---------- assembly ----------

def _s(x: str | None) -> str:
    return (x or "").strip()


def _address(a: ExtractedAddress) -> Address:
    return Address(street=_s(a.street), city=_s(a.city), region=_s(a.region),
                   postal_code=_s(a.postal_code), country=_s(a.country))


def _bank(b: ExtractedBank) -> BankAccount:
    clean = lambda x: re.sub(r"\s+", "", x) if x else None  # noqa: E731
    return BankAccount(
        bank_name=_s(b.bank_name), account_name=_s(b.account_name),
        account_number=clean(b.account_number), routing_number=clean(b.routing_number),
        iban=clean(b.iban), swift=clean(b.swift),
    )


def to_invoice(x: ExtractedInvoice) -> tuple[Invoice, list[str]]:
    warnings: list[str] = []
    terms = parse_terms_days(x.payment_terms)
    inv_date, due, w = resolve_dates(x.invoice_date, x.due_date, terms, x.vendor_address.country)
    warnings += w
    if due is None:
        if terms is None:
            raise NormalizationError("no due date and no payment terms")
        from datetime import timedelta
        due = inv_date + timedelta(days=terms)
        warnings.append("due date derived from payment terms")

    items = []
    for li in x.line_items:
        unit = parse_number(li.unit_price)
        unit = money(unit) if abs(unit) >= 1 else unit
        items.append(LineItem(description=li.description.strip(), quantity=parse_number(li.quantity),
                              unit_price=unit, amount=parse_money(li.amount)))

    subtotal, tax, total = parse_money(x.subtotal), parse_money(x.tax_amount), parse_money(x.total)
    rate = parse_rate(x.tax_rate)
    if rate is None:
        rate = (tax / subtotal).quantize(Decimal("0.0001")) if subtotal else Decimal("0")
        warnings.append("tax rate not printed; inferred from tax / subtotal")

    inv = Invoice(
        invoice_number=x.invoice_number.strip(),
        invoice_date=inv_date,
        due_date=due,
        po_number=_s(x.po_number) or None,
        vendor_name=x.vendor_name.strip(),
        vendor_address=_address(x.vendor_address),
        vendor_tax_id=_s(x.vendor_tax_id),
        vendor_email=_s(x.vendor_email),
        bill_to_name=_s(x.bill_to_name),
        bill_to_address=_address(x.bill_to_address),
        currency=x.currency.strip().upper(),
        line_items=items,
        subtotal=subtotal,
        tax_rate=rate,
        tax_amount=tax,
        total=total,
        payment_terms=_s(x.payment_terms),
        remit_to=_bank(x.remit_to),
    )
    return inv, warnings
