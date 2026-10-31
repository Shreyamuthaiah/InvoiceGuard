"""Renders an `Invoice` to PDF with one of several layouts, and degrades it into a 'scan'."""

from __future__ import annotations

import io
import random
from decimal import Decimal

import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from invoice_agent.schema import Invoice

W, H = letter
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


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
            s = s.replace(".", ",")
            return f"{s} {SYMBOLS[currency]}"
        return f"{SYMBOLS[currency]}{s}"
    return fmt_money(price, currency, style)


def fmt_qty(q: Decimal) -> str:
    return str(int(q)) if q == q.to_integral_value() else f"{q:.1f}"


def fmt_rate(rate: Decimal) -> str:
    return f"{rate * 100:.2f}".rstrip("0").rstrip(".") + "%"


# Per-template conventions — deliberately inconsistent, like real vendors.
STYLES = {
    "classic": {
        "date": lambda d, cur: d.strftime("%B %d, %Y"),
        "money": lambda cur: "us",
        "labels": {"number": "Invoice Number:", "date": "Invoice Date:", "due": "Due Date:",
                   "po": "PO Number:", "terms": "Terms:", "subtotal": "Subtotal",
                   "total": "Total Due", "bill_to": "BILL TO", "remit": "REMIT PAYMENT TO"},
    },
    "modern": {
        "date": lambda d, cur: d.strftime("%Y-%m-%d"),
        "money": lambda cur: "eu" if cur == "EUR" else "us",
        "labels": {"number": "Invoice #", "date": "Issued", "due": "Due", "po": "Purchase Order",
                   "terms": "Payment Terms", "subtotal": "Sub-total", "total": "Amount Due",
                   "bill_to": "Billed to", "remit": "Payment details"},
    },
    "minimal": {
        "date": lambda d, cur: d.strftime("%m/%d/%Y" if cur == "USD" else "%d/%m/%Y"),
        "money": lambda cur: "eu" if cur == "EUR" else "us",
        "labels": {"number": "Inv No.", "date": "Date", "due": "Payment due", "po": "Your ref",
                   "terms": "Terms", "subtotal": "Net", "total": "TOTAL",
                   "bill_to": "To:", "remit": "Bank:"},
    },
}


def money_style(template: str, currency: str) -> str:
    return STYLES[template]["money"](currency)


def _tax_label(inv: Invoice, template: str) -> str:
    kind = "Tax" if inv.currency == "USD" else "VAT"
    if template == "modern" and kind == "Tax":
        kind = "Sales Tax"
    return f"{kind} ({fmt_rate(inv.tax_rate)})"


def _bank_lines(inv: Invoice) -> list[str]:
    b = inv.remit_to
    out = [b.bank_name, f"Account name: {b.account_name}"]
    if b.iban:
        out += [f"IBAN: {b.iban}", f"SWIFT/BIC: {b.swift}"]
    else:
        out += [f"Account no: {b.account_number}", f"Routing (ABA): {b.routing_number}"]
    return out


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------- layouts ----------

def _classic(c: canvas.Canvas, inv: Invoice) -> None:
    st = STYLES["classic"]; L = st["labels"]; ms = money_style("classic", inv.currency)
    d = lambda x: st["date"](x, inv.currency)
    c.setFont("Helvetica-Bold", 15); c.drawString(50, H - 60, _clip(inv.vendor_name, 40))
    c.setFont("Helvetica", 9)
    y = H - 76
    for line in inv.vendor_address.lines() + [inv.vendor_email, f"Tax ID: {inv.vendor_tax_id}"]:
        c.drawString(50, y, line); y -= 12
    c.setFont("Helvetica-Bold", 26); c.drawRightString(W - 50, H - 64, "INVOICE")
    c.setFont("Helvetica", 9)
    meta = [(L["number"], inv.invoice_number), (L["date"], d(inv.invoice_date)), (L["due"], d(inv.due_date))]
    if inv.po_number:
        meta.append((L["po"], inv.po_number))
    meta.append((L["terms"], inv.payment_terms))
    y = H - 90
    for k, v in meta:
        c.drawRightString(W - 150, y, k); c.drawRightString(W - 50, y, v); y -= 13
    y = H - 190
    c.setFont("Helvetica-Bold", 9); c.drawString(50, y, L["bill_to"]); c.setFont("Helvetica", 9)
    for line in [inv.bill_to_name] + inv.bill_to_address.lines():
        y -= 12; c.drawString(50, y, line)
    y = H - 290
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y, "Description"); c.drawRightString(370, y, "Qty")
    c.drawRightString(460, y, "Unit Price"); c.drawRightString(W - 50, y, "Amount")
    c.line(50, y - 4, W - 50, y - 4)
    c.setFont("Helvetica", 9)
    for li in inv.line_items:
        y -= 18
        c.drawString(50, y, _clip(li.description, 48)); c.drawRightString(370, y, fmt_qty(li.quantity))
        c.drawRightString(460, y, fmt_unit(li.unit_price, inv.currency, ms))
        c.drawRightString(W - 50, y, fmt_money(li.amount, inv.currency, ms))
    c.line(50, y - 8, W - 50, y - 8)
    y -= 26
    for k, v, bold in [(L["subtotal"], inv.subtotal, False), (_tax_label(inv, "classic"), inv.tax_amount, False),
                       (L["total"], inv.total, True)]:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10 if bold else 9)
        c.drawRightString(460, y, k); c.drawRightString(W - 50, y, fmt_money(v, inv.currency, ms)); y -= 15
    y = 150
    c.setFont("Helvetica-Bold", 9); c.drawString(50, y, L["remit"]); c.setFont("Helvetica", 9)
    for line in _bank_lines(inv):
        y -= 12; c.drawString(50, y, line)
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(W / 2, 40, f"Thank you for your business. Questions? {inv.vendor_email}")


def _modern(c: canvas.Canvas, inv: Invoice) -> None:
    st = STYLES["modern"]; L = st["labels"]; ms = money_style("modern", inv.currency)
    d = lambda x: st["date"](x, inv.currency)
    band = colors.HexColor("#1F4E5F")
    c.setFillColor(band); c.rect(0, H - 100, W, 100, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18); c.drawString(40, H - 50, _clip(inv.vendor_name, 36))
    c.setFont("Helvetica", 8.5)
    c.drawString(40, H - 66, " · ".join(inv.vendor_address.lines()))
    c.drawString(40, H - 78, f"{inv.vendor_email}   ·   VAT/Tax ID {inv.vendor_tax_id}")
    c.setFont("Helvetica", 22); c.drawRightString(W - 40, H - 52, "Invoice")
    c.setFillColor(colors.black)
    # meta grid
    meta = [(L["number"], inv.invoice_number), (L["date"], d(inv.invoice_date)),
            (L["due"], d(inv.due_date)), (L["terms"], inv.payment_terms)]
    if inv.po_number:
        meta.append((L["po"], inv.po_number))
    x0, y0 = 330, H - 135
    for i, (k, v) in enumerate(meta):
        x = x0 + (i % 2) * 130; y = y0 - (i // 2) * 30
        c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey); c.drawString(x, y, k.upper())
        c.setFont("Helvetica-Bold", 10); c.setFillColor(colors.black); c.drawString(x, y - 12, v)
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey); c.drawString(40, y0, L["bill_to"].upper())
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10); c.drawString(40, y0 - 12, inv.bill_to_name)
    c.setFont("Helvetica", 9)
    y = y0 - 12
    for line in inv.bill_to_address.lines():
        y -= 12; c.drawString(40, y, line)
    # table
    y = H - 260
    c.setFillColor(colors.HexColor("#E8EEF0")); c.rect(40, y - 6, W - 80, 20, stroke=0, fill=1)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 8.5)
    c.drawString(48, y, "ITEM"); c.drawRightString(360, y, "QTY")
    c.drawRightString(460, y, "RATE"); c.drawRightString(W - 48, y, "TOTAL")
    c.setFont("Helvetica", 9)
    for li in inv.line_items:
        y -= 22
        c.drawString(48, y, _clip(li.description, 46)); c.drawRightString(360, y, fmt_qty(li.quantity))
        c.drawRightString(460, y, fmt_unit(li.unit_price, inv.currency, ms))
        c.drawRightString(W - 48, y, fmt_money(li.amount, inv.currency, ms))
        c.setStrokeColor(colors.HexColor("#DDDDDD")); c.line(40, y - 7, W - 40, y - 7)
    c.setStrokeColor(colors.black)
    y -= 34
    c.setFont("Helvetica", 9)
    c.drawString(360, y, L["subtotal"]); c.drawRightString(W - 48, y, fmt_money(inv.subtotal, inv.currency, ms))
    y -= 15
    c.drawString(360, y, _tax_label(inv, "modern")); c.drawRightString(W - 48, y, fmt_money(inv.tax_amount, inv.currency, ms))
    y -= 26
    c.setFillColor(band); c.rect(350, y - 8, W - 390, 26, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 11)
    c.drawString(360, y, L["total"]); c.drawRightString(W - 48, y, fmt_money(inv.total, inv.currency, ms))
    c.setFillColor(colors.black)
    y = 140
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey); c.drawString(40, y, L["remit"].upper())
    c.setFillColor(colors.black); c.setFont("Helvetica", 9)
    for line in _bank_lines(inv):
        y -= 12; c.drawString(40, y, line)


def _minimal(c: canvas.Canvas, inv: Invoice) -> None:
    st = STYLES["minimal"]; L = st["labels"]; ms = money_style("minimal", inv.currency)
    d = lambda x: st["date"](x, inv.currency)
    c.setFont("Courier-Bold", 12); c.drawString(60, H - 60, inv.vendor_name.upper()[:44])
    c.setFont("Courier", 9)
    y = H - 74
    for line in inv.vendor_address.lines() + [f"{inv.vendor_email}  /  {inv.vendor_tax_id}"]:
        c.drawString(60, y, line); y -= 11
    y -= 20
    rows = [(L["number"], inv.invoice_number), (L["date"], d(inv.invoice_date)),
            (L["due"], d(inv.due_date)), (L["terms"], inv.payment_terms)]
    if inv.po_number:
        rows.append((L["po"], inv.po_number))
    for k, v in rows:
        c.drawString(60, y, f"{k:<14}{v}"); y -= 11
    y -= 14
    c.drawString(60, y, L["bill_to"]); y -= 11
    for line in [inv.bill_to_name] + inv.bill_to_address.lines():
        c.drawString(84, y, line); y -= 11
    y -= 18
    c.drawString(60, y, "-" * 82); y -= 12
    for li in inv.line_items:
        c.drawString(60, y, _clip(li.description, 38))
        c.drawRightString(370, y, f"{fmt_qty(li.quantity)} x {fmt_unit(li.unit_price, inv.currency, ms)}")
        c.drawRightString(W - 60, y, fmt_money(li.amount, inv.currency, ms))
        y -= 12
    c.drawString(60, y, "-" * 82); y -= 14
    for k, v in [(L["subtotal"], inv.subtotal), (_tax_label(inv, "minimal"), inv.tax_amount)]:
        c.drawRightString(440, y, k); c.drawRightString(W - 60, y, fmt_money(v, inv.currency, ms)); y -= 12
    c.setFont("Courier-Bold", 10)
    c.drawRightString(440, y - 2, L["total"]); c.drawRightString(W - 60, y - 2, fmt_money(inv.total, inv.currency, ms))
    c.setFont("Courier", 9)
    y = 120
    c.drawString(60, y, L["remit"])
    for line in _bank_lines(inv):
        c.drawString(100, y, line); y -= 11


LAYOUTS = {"classic": _classic, "modern": _modern, "minimal": _minimal}


def render_pdf_bytes(inv: Invoice, template: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)  # invariant -> byte-reproducible
    c.setTitle(f"Invoice {inv.invoice_number}")
    LAYOUTS[template](c, inv)
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
