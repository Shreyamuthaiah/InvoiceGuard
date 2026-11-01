"""Renders an `Invoice` to PDF and degrades it into a 'scan'.

Three base layouts (classic / modern / minimal). Each vendor gets a `Style` on top of its
base layout — font, accent colour, column order, label wording, date and number format,
block placement — so documents vary the way real vendors' invoices do. A `RenderInfo`
without a style renders with the original fixed look of its template (dataset v1).
"""

from __future__ import annotations

import io
import random
from datetime import date
from decimal import Decimal
from typing import Optional

import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from invoice_agent.schema import Injection, Invoice, Style

W, H = letter
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}
FONTS = {  # family -> (regular, bold, italic)
    "Helvetica": ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique"),
    "Times": ("Times-Roman", "Times-Bold", "Times-Italic"),
    "Courier": ("Courier", "Courier-Bold", "Courier-Oblique"),
}
COLUMN_KEYS = ["desc", "qty", "unit", "amount"]


# ---------- formatting ----------

def fmt_money(amount: Decimal, currency: str, style: str = "us") -> str:
    s = f"{amount:,.2f}"
    neg = s.startswith("-")
    s = s.lstrip("-")
    if style == "eu":
        s = s.translate(str.maketrans(",.", ".,"))
        out = f"{s} {SYMBOLS[currency]}"
    else:
        out = f"{SYMBOLS[currency]}{s}"
    return f"-{out}" if neg else out


def fmt_unit(price: Decimal, currency: str, style: str) -> str:
    if price < 1:
        s = f"{price:.4f}"
        if style == "eu":
            return f"{s.replace('.', ',')} {SYMBOLS[currency]}"
        return f"{SYMBOLS[currency]}{s}"
    return fmt_money(price, currency, style)


def fmt_qty(q: Decimal) -> str:
    return str(int(q)) if q == q.to_integral_value() else f"{q:.1f}"


def fmt_rate(rate: Decimal) -> str:
    return f"{rate * 100:.2f}".rstrip("0").rstrip(".") + "%"


def fmt_date(style: Style, d: date) -> str:
    return d.strftime(style.date_fmt)


# ---------- styles ----------

_V1_LABELS = {
    "classic": {"number": "Invoice Number:", "date": "Invoice Date:", "due": "Due Date:", "po": "PO Number:",
                "terms": "Terms:", "subtotal": "Subtotal", "total": "Total Due", "bill_to": "BILL TO",
                "remit": "REMIT PAYMENT TO", "desc": "Description", "qty": "Qty", "unit": "Unit Price",
                "amount": "Amount", "title": "INVOICE"},
    "modern": {"number": "Invoice #", "date": "Issued", "due": "Due", "po": "Purchase Order",
               "terms": "Payment Terms", "subtotal": "Sub-total", "total": "Amount Due", "bill_to": "Billed to",
               "remit": "Payment details", "desc": "ITEM", "qty": "QTY", "unit": "RATE", "amount": "TOTAL",
               "title": "Invoice"},
    "minimal": {"number": "Inv No.", "date": "Date", "due": "Payment due", "po": "Your ref", "terms": "Terms",
                "subtotal": "Net", "total": "TOTAL", "bill_to": "To:", "remit": "Bank:", "desc": "", "qty": "",
                "unit": "", "amount": "", "title": ""},
}


def default_style(template: str, currency: str) -> Style:
    """The fixed per-template look used by dataset v1."""
    eu = currency == "EUR" and template != "classic"
    date_fmt = {"classic": "%B %d, %Y", "modern": "%Y-%m-%d",
                "minimal": "%m/%d/%Y" if currency == "USD" else "%d/%m/%Y"}[template]
    return Style(template=template, font="Courier" if template == "minimal" else "Helvetica",
                 labels=dict(_V1_LABELS[template]), date_fmt=date_fmt, money="eu" if eu else "us")


SYNONYMS = {
    "number": ["Invoice Number", "Invoice No.", "Invoice #", "Inv No.", "Document No.", "Bill Number"],
    "date": ["Invoice Date", "Date", "Issued", "Date of Issue", "Billing Date"],
    "due": ["Due Date", "Due", "Payment Due", "Pay By", "Due On"],
    "po": ["PO Number", "Purchase Order", "Your Ref", "PO #", "Customer PO", "Order Ref"],
    "terms": ["Terms", "Payment Terms", "Credit Terms"],
    "subtotal": ["Subtotal", "Sub-total", "Net", "Net Amount", "Total before tax"],
    "total": ["Total Due", "Amount Due", "TOTAL", "Balance Due", "Total Payable", "Please Pay"],
    "bill_to": ["Bill To", "Billed to", "To:", "Customer", "Invoice To", "Sold To"],
    "remit": ["Remit Payment To", "Payment details", "Bank:", "Pay to", "Bank Details", "Wire Instructions"],
    "desc": ["Description", "Item", "Service", "Details"],
    "qty": ["Qty", "Quantity", "Units", "Hrs/Qty"],
    "unit": ["Unit Price", "Rate", "Price", "Unit Cost"],
    "amount": ["Amount", "Total", "Line Total", "Ext. Price"],
    "title": ["INVOICE", "Invoice", "TAX INVOICE", "Bill"],
}
ACCENTS = ["#1F4E5F", "#7A1F2B", "#2E4A1F", "#3B2F6B", "#8A5A00", "#222222", "#0B5C8C", "#5C3317"]
COLUMN_ORDERS = [["desc", "qty", "unit", "amount"], ["qty", "desc", "unit", "amount"],
                 ["desc", "unit", "qty", "amount"]]


def make_style(rng: random.Random, template: str, currency: str) -> Style:
    """A vendor's own variation of a base layout. Uses its own RNG so invoice content is unaffected."""
    numeric = "%m/%d/%Y" if currency == "USD" else ("%d.%m.%Y" if currency == "EUR" else "%d/%m/%Y")
    labels = {k: rng.choice(v) for k, v in SYNONYMS.items()}
    if template == "minimal":
        for k in ("desc", "qty", "unit", "amount", "title"):
            labels[k] = ""
    return Style(
        template=template,
        font=rng.choice(["Courier", "Courier", "Helvetica"]) if template == "minimal"
        else rng.choice(["Helvetica", "Helvetica", "Times"]),
        accent=rng.choice(ACCENTS),
        columns=rng.choice(COLUMN_ORDERS),
        labels=labels,
        date_fmt=rng.choice(["%B %d, %Y", "%Y-%m-%d", "%b %d, %Y", "%d %B %Y", numeric, numeric]),
        money=rng.choice(["eu", "eu", "us"]) if currency == "EUR" else "us",
        upper_labels=rng.random() < 0.3,
        vendor_right=rng.random() < 0.3,
        bank_beside_totals=rng.random() < 0.3,
    )


def resolve_style(template: str, currency: str, style: Optional[Style]) -> Style:
    return style or default_style(template, currency)


# ---------- drawing helpers ----------

class _Pen:
    def __init__(self, c: canvas.Canvas, inv: Invoice, st: Style) -> None:
        self.c, self.inv, self.st = c, inv, st
        self.reg, self.bold, self.ital = FONTS[st.font]

    def lab(self, key: str) -> str:
        text = self.st.labels.get(key, "")
        return text.upper() if self.st.upper_labels else text

    def money(self, x: Decimal) -> str:
        return fmt_money(x, self.inv.currency, self.st.money)

    def unit(self, x: Decimal) -> str:
        return fmt_unit(x, self.inv.currency, self.st.money)

    def date(self, d: date) -> str:
        return fmt_date(self.st, d)

    def meta(self) -> list[tuple[str, str]]:
        inv = self.inv
        rows = [(self.lab("number"), inv.invoice_number), (self.lab("date"), self.date(inv.invoice_date)),
                (self.lab("due"), self.date(inv.due_date))]
        if inv.po_number:
            rows.append((self.lab("po"), inv.po_number))
        rows.append((self.lab("terms"), inv.payment_terms))
        return rows

    def tax_label(self) -> str:
        kind = "Tax" if self.inv.currency == "USD" else "VAT"
        if self.st.template == "modern" and kind == "Tax":
            kind = "Sales Tax"
        return f"{kind} ({fmt_rate(self.inv.tax_rate)})"

    def bank_lines(self) -> list[str]:
        b = self.inv.remit_to
        out = [b.bank_name, f"Account name: {b.account_name}"]
        if b.iban:
            out += [f"IBAN: {b.iban}", f"SWIFT/BIC: {b.swift}"]
        else:
            out += [f"Account no: {b.account_number}", f"Routing (ABA): {b.routing_number}"]
        return out

    def column_x(self, x0: float, x1: float) -> dict[str, tuple[float, str]]:
        """Right edge (numbers) or left edge (description) for each column, in the style's order."""
        widths = {"desc": 0.46, "qty": 0.12, "unit": 0.21, "amount": 0.21}
        out, x = {}, x0
        for key in self.st.columns:
            w = widths[key] * (x1 - x0)
            out[key] = (x, "left") if key == "desc" else (x + w, "right")
            x += w
        return out

    def text(self, x: float, y: float, s: str, align: str = "left") -> None:
        {"left": self.c.drawString, "right": self.c.drawRightString}[align](x, y, s)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _tint(hex_color: str, amount: float = 0.85) -> colors.Color:
    base = colors.HexColor(hex_color)
    return colors.Color(base.red + (1 - base.red) * amount, base.green + (1 - base.green) * amount,
                        base.blue + (1 - base.blue) * amount)


def _table(p: _Pen, x0: float, x1: float, y: float, row_h: float, shaded: bool) -> float:
    c, cols = p.c, p.column_x(x0, x1)
    if shaded:
        c.setFillColor(_tint(p.st.accent)); c.rect(x0, y - 6, x1 - x0, 20, stroke=0, fill=1)
        c.setFillColor(colors.black)
    c.setFont(p.bold, 8.5)
    for key, (x, align) in cols.items():
        p.text(x + (8 if align == "left" else -8 if shaded else 0), y, p.lab(key), align)
    if not shaded:
        c.line(x0, y - 4, x1, y - 4)
    c.setFont(p.reg, 9)
    for li in p.inv.line_items:
        y -= row_h
        values = {"desc": _clip(li.description, 44), "qty": fmt_qty(li.quantity),
                  "unit": p.unit(li.unit_price), "amount": p.money(li.amount)}
        for key, (x, align) in cols.items():
            p.text(x + (8 if align == "left" else -8 if shaded else 0), y, values[key], align)
    c.line(x0, y - 8, x1, y - 8)
    return y


def _bank_block(p: _Pen, x: float, y: float, label_font: str) -> None:
    c = p.c
    c.setFont(label_font, 9); c.drawString(x, y, p.lab("remit")); c.setFont(p.reg, 9)
    for line in p.bank_lines():
        y -= 12; c.drawString(x, y, line)


def _injections(c: canvas.Canvas, injections: list[Injection], font: str) -> None:
    notes = [i.text for i in injections if i.mode == "note"]
    if notes:
        y = 205
        c.setFont(font, 8); c.setFillColor(colors.HexColor("#444444"))
        c.drawString(50, y, "Notes:")
        for text in notes:
            for k in range(0, len(text), 95):
                y -= 10; c.drawString(50, y, text[k:k + 95])
        c.setFillColor(colors.black)
    for text in (i.text for i in injections if i.mode == "hidden"):
        c.setFont(font, 1); c.setFillColor(colors.white)  # invisible on paper, present in the text layer
        c.drawString(60, 60, text)
        c.setFillColor(colors.black)


# ---------- layouts ----------

def _classic(c: canvas.Canvas, inv: Invoice, st: Style) -> None:
    p = _Pen(c, inv, st)
    vx, tx = (W - 50, 50) if st.vendor_right else (50, W - 50)
    vdraw = c.drawRightString if st.vendor_right else c.drawString
    tdraw = c.drawString if st.vendor_right else c.drawRightString
    c.setFont(p.bold, 15); vdraw(vx, H - 60, _clip(inv.vendor_name, 40))
    c.setFont(p.reg, 9)
    y = H - 76
    for line in inv.vendor_address.lines() + [inv.vendor_email, f"Tax ID: {inv.vendor_tax_id}"]:
        vdraw(vx, y, line); y -= 12
    c.setFont(p.bold, 24); tdraw(tx, H - 64, p.lab("title"))
    c.setFont(p.reg, 9)
    y = H - 90
    for k, v in p.meta():
        if st.vendor_right:
            c.drawString(50, y, k); c.drawString(150, y, v)
        else:
            c.drawRightString(W - 150, y, k); c.drawRightString(W - 50, y, v)
        y -= 13
    y = H - 190
    c.setFont(p.bold, 9); c.drawString(50, y, p.lab("bill_to")); c.setFont(p.reg, 9)
    for line in [inv.bill_to_name] + inv.bill_to_address.lines():
        y -= 12; c.drawString(50, y, line)
    y = _table(p, 50, W - 50, H - 290, 18, shaded=False) - 26
    totals_top = y
    for k, v, bold in [(p.lab("subtotal"), inv.subtotal, False), (p.tax_label(), inv.tax_amount, False),
                       (p.lab("total"), inv.total, True)]:
        c.setFont(p.bold if bold else p.reg, 10 if bold else 9)
        c.drawRightString(460, y, k); c.drawRightString(W - 50, y, p.money(v)); y -= 15
    _bank_block(p, 50, totals_top if st.bank_beside_totals else 150, p.bold)
    c.setFont(p.ital, 8)
    c.drawCentredString(W / 2, 40, f"Thank you for your business. Questions? {inv.vendor_email}")


def _modern(c: canvas.Canvas, inv: Invoice, st: Style) -> None:
    p = _Pen(c, inv, st)
    band = colors.HexColor(st.accent)
    c.setFillColor(band); c.rect(0, H - 100, W, 100, stroke=0, fill=1)
    c.setFillColor(colors.white)
    if st.vendor_right:
        c.setFont(p.bold, 18); c.drawRightString(W - 40, H - 50, _clip(inv.vendor_name, 36))
        c.setFont(p.reg, 8.5)
        c.drawRightString(W - 40, H - 66, " · ".join(inv.vendor_address.lines()))
        c.drawRightString(W - 40, H - 78, f"{inv.vendor_email}   ·   VAT/Tax ID {inv.vendor_tax_id}")
        c.setFont(p.reg, 22); c.drawString(40, H - 52, p.lab("title"))
    else:
        c.setFont(p.bold, 18); c.drawString(40, H - 50, _clip(inv.vendor_name, 36))
        c.setFont(p.reg, 8.5)
        c.drawString(40, H - 66, " · ".join(inv.vendor_address.lines()))
        c.drawString(40, H - 78, f"{inv.vendor_email}   ·   VAT/Tax ID {inv.vendor_tax_id}")
        c.setFont(p.reg, 22); c.drawRightString(W - 40, H - 52, p.lab("title"))
    c.setFillColor(colors.black)
    x0, y0 = 330, H - 135
    for i, (k, v) in enumerate(p.meta()):
        x = x0 + (i % 2) * 130; y = y0 - (i // 2) * 30
        c.setFont(p.reg, 7.5); c.setFillColor(colors.grey); c.drawString(x, y, k.upper())
        c.setFont(p.bold, 10); c.setFillColor(colors.black); c.drawString(x, y - 12, v)
    c.setFont(p.reg, 7.5); c.setFillColor(colors.grey); c.drawString(40, y0, p.lab("bill_to").upper())
    c.setFillColor(colors.black); c.setFont(p.bold, 10); c.drawString(40, y0 - 12, inv.bill_to_name)
    c.setFont(p.reg, 9)
    y = y0 - 12
    for line in inv.bill_to_address.lines():
        y -= 12; c.drawString(40, y, line)
    y = _table(p, 40, W - 40, H - 260, 22, shaded=True) - 26
    totals_top = y
    c.setFont(p.reg, 9)
    c.drawString(360, y, p.lab("subtotal")); c.drawRightString(W - 48, y, p.money(inv.subtotal)); y -= 15
    c.drawString(360, y, p.tax_label()); c.drawRightString(W - 48, y, p.money(inv.tax_amount)); y -= 26
    c.setFillColor(band); c.rect(350, y - 8, W - 390, 26, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont(p.bold, 11)
    c.drawString(360, y, p.lab("total")); c.drawRightString(W - 48, y, p.money(inv.total))
    c.setFillColor(colors.black)
    _bank_block(p, 40, totals_top if st.bank_beside_totals else 140, p.bold)


def _minimal(c: canvas.Canvas, inv: Invoice, st: Style) -> None:
    p = _Pen(c, inv, st)
    c.setFont(p.bold, 12); c.drawString(60, H - 60, inv.vendor_name.upper()[:44])
    c.setFont(p.reg, 9)
    y = H - 74
    for line in inv.vendor_address.lines() + [f"{inv.vendor_email}  /  {inv.vendor_tax_id}"]:
        c.drawString(60, y, line); y -= 11
    y -= 20
    for k, v in p.meta():
        c.drawString(60, y, k); c.drawString(160, y, v); y -= 11
    y -= 14
    c.drawString(60, y, p.lab("bill_to")); y -= 11
    for line in [inv.bill_to_name] + inv.bill_to_address.lines():
        c.drawString(84, y, line); y -= 11
    y -= 18
    c.drawString(60, y, "-" * 82); y -= 12
    for li in inv.line_items:
        c.drawString(60, y, _clip(li.description, 38))
        c.drawRightString(370, y, f"{fmt_qty(li.quantity)} x {p.unit(li.unit_price)}")
        c.drawRightString(W - 60, y, p.money(li.amount))
        y -= 12
    c.drawString(60, y, "-" * 82); y -= 14
    for k, v in [(p.lab("subtotal"), inv.subtotal), (p.tax_label(), inv.tax_amount)]:
        c.drawRightString(440, y, k); c.drawRightString(W - 60, y, p.money(v)); y -= 12
    c.setFont(p.bold, 10)
    c.drawRightString(440, y - 2, p.lab("total")); c.drawRightString(W - 60, y - 2, p.money(inv.total))
    c.setFont(p.reg, 9)
    y = (y - 30) if st.bank_beside_totals else 120
    c.drawString(60, y, p.lab("remit"))
    for line in p.bank_lines():
        c.drawString(130, y, line); y -= 11


LAYOUTS = {"classic": _classic, "modern": _modern, "minimal": _minimal}


def _printable(inv: Invoice) -> Invoice:
    """Fonts can't draw line breaks (Faker's UK streets contain one); print them as spaces.
    Labels keep the original value; the eval compares addresses whitespace-insensitively."""
    def fix(a):
        return a.model_copy(update={k: " ".join(str(v).split()) for k, v in a.model_dump().items()})
    return inv.model_copy(update={"vendor_address": fix(inv.vendor_address),
                                  "bill_to_address": fix(inv.bill_to_address)})


def render_pdf_bytes(inv: Invoice, template: str, style: Optional[Style] = None,
                     injections: Optional[list[Injection]] = None) -> bytes:
    st = resolve_style(template, inv.currency, style)
    inv = _printable(inv)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)  # invariant -> byte-reproducible
    c.setTitle(f"Invoice {inv.invoice_number}")
    LAYOUTS[template](c, inv, st)
    if injections:
        _injections(c, injections, FONTS[st.font][0])
    c.showPage()
    c.save()
    return buf.getvalue()


def degrade_to_scan(pdf_bytes: bytes, rng: random.Random) -> tuple[bytes, dict[str, float]]:
    """Rasterise page 1 and apply scanner-like artefacts. Returns JPEG bytes + parameters."""
    params = {
        "dpi": float(rng.choice([110, 150, 200])),
        "rotation_deg": round(rng.uniform(-2.5, 2.5), 2),
        "blur_radius": round(rng.uniform(0.0, 1.0), 2),
        "noise_alpha": round(rng.uniform(0.03, 0.12), 3),
        "jpeg_quality": float(rng.randint(35, 80)),
    }
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=int(params["dpi"]), colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    noise = Image.frombytes("L", img.size, rng.randbytes(img.width * img.height))
    img = Image.blend(img, noise, params["noise_alpha"])
    if params["blur_radius"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    img = img.rotate(params["rotation_deg"], resample=Image.BICUBIC, expand=True, fillcolor=255)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=int(params["jpeg_quality"]))
    return out.getvalue(), params
