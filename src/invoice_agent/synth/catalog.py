"""Static reference data for the synthetic generator. All names are fictional."""

# category -> [(description, (min_unit_price, max_unit_price), (min_qty, max_qty))]
CATALOG: dict[str, list[tuple[str, tuple[float, float], tuple[int, int]]]] = {
    "IT services": [
        ("Managed IT support - monthly retainer", (800, 4500), (1, 1)),
        ("On-site technician hours", (85, 160), (2, 40)),
        ("Laptop provisioning & imaging", (45, 120), (3, 25)),
        ("Network security audit", (1500, 6000), (1, 1)),
        ("Help desk tickets (overage)", (25, 60), (5, 80)),
    ],
    "Office supplies": [
        ("Copy paper, letter size (case)", (28, 55), (2, 30)),
        ("Toner cartridge, black", (60, 180), (1, 12)),
        ("Ergonomic office chair", (180, 650), (1, 10)),
        ("Whiteboard markers (12-pack)", (9, 22), (2, 20)),
        ("Desk organizer set", (15, 45), (2, 15)),
    ],
    "Logistics": [
        ("LTL freight - regional", (250, 1800), (1, 6)),
        ("Pallet storage (per pallet/month)", (12, 35), (10, 120)),
        ("Expedited courier delivery", (35, 140), (1, 20)),
        ("Fuel surcharge", (20, 300), (1, 1)),
        ("Customs brokerage fee", (75, 400), (1, 4)),
    ],
    "Consulting": [
        ("Senior consultant hours", (150, 350), (4, 80)),
        ("Strategy workshop facilitation", (2000, 9000), (1, 2)),
        ("Data analysis deliverable", (1200, 7000), (1, 3)),
        ("Travel & expenses (pass-through)", (100, 2500), (1, 1)),
    ],
    "Cloud & software": [
        ("SaaS subscription - per seat/month", (8, 95), (5, 250)),
        ("Compute instances (vCPU-hours)", (0.03, 0.12), (1000, 40000)),
        ("Object storage (TB-month)", (18, 25), (1, 60)),
        ("Premium support plan", (300, 2500), (1, 1)),
        ("Implementation services hours", (120, 240), (4, 60)),
    ],
    "Facilities": [
        ("Janitorial services - monthly", (900, 4800), (1, 1)),
        ("HVAC preventive maintenance", (350, 1600), (1, 2)),
        ("Replacement LED fixtures", (25, 90), (4, 60)),
        ("After-hours emergency call-out", (180, 450), (1, 3)),
    ],
}

US_BANKS = [
    "First Harbor Bank", "Granite Trust Bank", "Prairie National Bank",
    "Bluewater Savings", "Summit Commerce Bank", "Redwood Federal Credit Union",
]
EU_BANKS = ["Nordhafen Bank AG", "Rheinland Kreditbank", "Alpen Volksbank eG", "Hansa Privatbank"]
UK_BANKS = ["Thames Mercantile Bank", "Pennine Building Society", "Clydeside Bank plc"]

# Locale config per invoicing currency.
CURRENCIES = {
    "USD": {"locale": "en_US", "country": "United States", "banks": US_BANKS,
            "tax_rates": ["0", "0.0625", "0.0725", "0.08", "0.0875"]},
    "EUR": {"locale": "de_DE", "country": "Germany", "banks": EU_BANKS, "tax_rates": ["0.19"]},
    "GBP": {"locale": "en_GB", "country": "United Kingdom", "banks": UK_BANKS, "tax_rates": ["0.20"]},
}
CURRENCY_WEIGHTS = {"USD": 0.7, "EUR": 0.15, "GBP": 0.15}

INVOICE_NUMBER_PATTERNS = ["INV-{year}-{n:05d}", "{prefix}-{n:06d}", "{n}", "{yy}{n:04d}"]

TEMPLATES = ["classic", "modern", "minimal"]

# The buyer: our own (fictional) company, whose AP team receives every invoice.
BUYER_NAME = "Larkspur Robotics, Inc."
BUYER_ADDRESS = {
    "street": "4100 Meridian Way, Suite 300",
    "city": "Columbus",
    "region": "OH",
    "postal_code": "43215",
    "country": "United States",
}
