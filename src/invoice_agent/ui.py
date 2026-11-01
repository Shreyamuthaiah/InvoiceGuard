"""Local demo UI: pick or upload an invoice, see extraction, findings, decision and payment.

    python -m invoice_agent.ui            # http://127.0.0.1:8765
    python -m invoice_agent.ui --port 9000 --today 2026-03-01

Sources: "labels" (instant, no model), "cached" (predictions from an eval run, no API calls),
"live" (calls the configured provider; default Gemini, uses free-tier quota).
Standard library only. Binds to 127.0.0.1. Document text is rendered with textContent,
never as HTML, because some invoices (adversarial set) contain hostile text.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from invoice_agent.checks_eval import load
from invoice_agent.pipeline import InvoicePipeline, ProcessResult, labels_extractor, live_extractor, predictions_extractor
from invoice_agent.schema import GroundTruth

def project_root() -> Path:
    """The repo folder (holds pyproject.toml), found from the current directory upwards, then
    from this file's location, so the UI works no matter where it is launched from."""
    for start in (Path.cwd(), Path(__file__).resolve()):
        for d in [start, *start.parents]:
            if (d / "pyproject.toml").exists() and (d / "src" / "invoice_agent").exists():
                return d
    return Path.cwd()


ROOT = project_root()

CACHED_RUNS = {  # dataset dir name -> eval runs to reuse, first existing wins
    "synthetic": ["runs/v2-gemini-3.1-flash-lite", "runs/gemini-3.1-flash-lite"],
    "adversarial": ["runs/adv-gemini-3.1-flash-lite"],
}


def preview(path: Path) -> Optional[str]:
    """Small PNG/JPEG data URL of page 1."""
    if not path.exists():
        return None
    if path.suffix.lower() == ".pdf":
        import pymupdf
        with pymupdf.open(path) as doc:
            png = doc[0].get_pixmap(dpi=72).tobytes("png")
        return "data:image/png;base64," + base64.b64encode(png).decode()
    from PIL import Image
    with Image.open(path) as img:
        img.thumbnail((700, 900))
        buf = io.BytesIO()
        img.convert("L").save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Dataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.vendors, self.pos, cases = load(path)
        self.cases: dict[str, GroundTruth] = {gt.doc_id: gt for gt in cases}
        self.predictions: Optional[Path] = next(
            (ROOT / r / "predictions.jsonl" for r in CACHED_RUNS.get(path.name, [])
             if (ROOT / r / "predictions.jsonl").exists()), None)


class Session:
    def __init__(self, datasets: list[Path], out_root: Path, provider: str, model: Optional[str],
                 today: date, explain_llm=None) -> None:
        self.datasets = {p.name: Dataset(p) for p in datasets if (p / "manifest.jsonl").exists()}
        if not self.datasets:
            raise SystemExit("no datasets found; run `python -m invoice_agent.synth` first")
        self.out_root, self.provider, self.model, self.today = out_root, provider, model, today
        self.explain_llm = explain_llm
        self._live = None
        self.reset()

    def reset(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = self.out_root / stamp
        self.pipes = {name: InvoicePipeline(ds.vendors, ds.pos, labels_extractor(ds.path), self.out / name,
                                            self.today, explain_llm=self.explain_llm)
                      for name, ds in self.datasets.items()}
        self.results: list[dict] = []
        self.uploads = 0

    def _extractor(self, ds: Dataset, source: str):
        if source == "labels":
            return labels_extractor(ds.path)
        if source == "cached":
            if not ds.predictions:
                raise ValueError(f"no cached eval run for {ds.path.name}")
            return predictions_extractor(ds.predictions)
        if source == "live":
            if self._live is None:
                self._live = live_extractor(self.provider, self.model)
            return self._live
        raise ValueError(f"unknown source {source!r}")

    def process(self, dataset: str, doc_id: str, source: str) -> dict:
        ds = self.datasets[dataset]
        gt = ds.cases.get(doc_id)
        if gt is None:
            raise ValueError("unknown document")
        path = ds.path / gt.render.file
        pipe = self.pipes[dataset]
        pipe.extract = self._extractor(ds, source)
        return self._record(pipe.process(doc_id, path), path, source, gt)

    def upload(self, filename: str, data: bytes) -> dict:
        if not data.startswith(b"%PDF"):
            raise ValueError("only PDF uploads are supported")
        self.uploads += 1
        doc_id = f"upload_{self.uploads:03d}"
        folder = self.out / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{doc_id}.pdf"
        path.write_bytes(data)
        name = next(iter(self.datasets))  # uploads are checked against the main vendor master
        pipe = self.pipes[name]
        pipe.extract = self._extractor(self.datasets[name], "live")
        return self._record(pipe.process(doc_id, path), path, "live", None, filename)

    def _record(self, r: ProcessResult, path: Path, source: str, gt: Optional[GroundTruth],
                filename: str = "") -> dict:
        rec = {
            "doc_id": r.doc_id, "filename": filename, "source": source, "preview": preview(path),
            "decision": r.decision.value, "reasons": r.reasons, "warnings": r.warnings,
            "extraction_error": r.extraction_error,
            "invoice": r.invoice.model_dump(mode="json") if r.invoice else None,
            "findings": [f.model_dump(mode="json") for f in r.report.findings] if r.report else [],
            "vendor_match": (r.report.vendor_match.method if r.report else None),
            "payment": r.payment.model_dump(mode="json") if r.payment else None,
            "payment_reused": bool(r.payment) and not r.payment_created,
            "explanation": r.explanation,
            "expected": ({"decision": gt.expected_decision.value, "anomalies": [a.value for a in gt.anomalies],
                          "notes": gt.notes} if gt else None),
        }
        self.results.append({k: rec[k] for k in ("doc_id", "decision", "payment")})
        return rec

    def docs(self) -> dict:
        return {name: {"cached": bool(ds.predictions),
                       "docs": [{"doc_id": g.doc_id, "template": g.render.template, "scanned": g.render.scanned,
                                 "currency": g.invoice.currency, "vendor": g.invoice.vendor_name,
                                 "expected": g.expected_decision.value} for g in ds.cases.values()]}
                for name, ds in self.datasets.items()}

    def stats(self) -> dict:
        counts = {"approve": 0, "hold": 0, "reject": 0}
        totals: dict[str, float] = {}
        for r in self.results:
            counts[r["decision"]] += 1
            p = r["payment"]
            if p:
                totals[p["currency"]] = totals.get(p["currency"], 0.0) + float(p["amount"])
        return {"processed": len(self.results), **counts, "payments": totals, "out": str(self.out)}


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/docs":
                self._json(session.docs())
            elif url.path == "/api/stats":
                self._json(session.stats())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            length = int(self.headers.get("Content-Length") or 0)
            if length > 20 * 1024 * 1024:
                return self._json({"error": "file too large"}, 413)
            body = self.rfile.read(length) if length else b""
            try:
                if url.path == "/api/process":
                    self._json(session.process(q.get("dataset", ""), q.get("doc_id", ""), q.get("source", "labels")))
                elif url.path == "/api/upload":
                    name = re.sub(r"[^\w.\- ]", "_", q.get("name", "upload.pdf"))[:80]
                    self._json(session.upload(name, body))
                elif url.path == "/api/reset":
                    session.reset()
                    self._json(session.stats())
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, RuntimeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invoice Agent</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#667085;--line:#e4e7ec;--accent:#2f5bd3;
--ok:#137a3d;--okbg:#e6f4ec;--warn:#9a5b00;--warnbg:#fdf3e1;--bad:#b42318;--badbg:#fdecea}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--panel:#1a1d24;--ink:#e6e8ee;--muted:#98a2b3;
--line:#2b303b;--accent:#7b9cff;--ok:#6fd39a;--okbg:#14301f;--warn:#f0b65a;--warnbg:#33270f;--bad:#ff8a80;--badbg:#3a1714}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:16px;margin:0 12px 0 0}.stat{color:var(--muted)}.stat b{color:var(--ink)}
button,select,input[type=file]{font:inherit;color:inherit}select{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 6px;width:100%}button{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}main{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 54px)}
aside{border-right:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0}
.controls{padding:12px;border-bottom:1px solid var(--line);display:grid;gap:8px}
.controls label{font-size:12px;color:var(--muted)}#list{overflow:auto;flex:1;max-height:calc(100vh - 250px)}
.doc{padding:7px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;justify-content:space-between;gap:8px}
.doc:hover,.doc.active{background:var(--bg)}.doc small{color:var(--muted);display:block}
.chip{font-size:11px;padding:1px 7px;border-radius:999px;white-space:nowrap;align-self:center}
.approve{background:var(--okbg);color:var(--ok)}.hold{background:var(--warnbg);color:var(--warn)}.reject{background:var(--badbg);color:var(--bad)}
#view{padding:16px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:16px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
#preview img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.decision{font-size:22px;font-weight:700;text-transform:uppercase;padding:6px 12px;border-radius:8px;display:inline-block}
ul.findings{list-style:none;padding:0;margin:0}ul.findings li{padding:6px 0;border-bottom:1px solid var(--line)}
.sev{font-size:11px;font-weight:600;margin-right:6px}.sev.error{color:var(--bad)}.sev.warning{color:var(--warn)}.sev.info{color:var(--muted)}
table{border-collapse:collapse;width:100%}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:500;width:38%}td.num{text-align:right;font-variant-numeric:tabular-nums}
.stack{display:grid;gap:16px}.muted{color:var(--muted)}.empty{padding:40px;color:var(--muted)}
@media (max-width:900px){main{grid-template-columns:1fr}#view{grid-template-columns:1fr}#list{max-height:300px}}
</style></head><body>
<header><h1>Invoice Agent</h1><span class="stat" id="stats"></span><span style="flex:1"></span>
<button id="reset" title="Start a new session (clears the duplicate ledger and payments)">New session</button></header>
<main><aside><div class="controls">
<div><label>Dataset</label><br><select id="dataset"></select></div>
<div><label>Extraction source</label><br><select id="source">
<option value="labels">Labels (perfect extractor, instant)</option>
<option value="cached">Cached Gemini run (no API calls)</option>
<option value="live">Live model (uses API quota)</option></select></div>
<div><label>Upload a PDF (live model)</label><br><input type="file" id="file" accept="application/pdf"></div>
</div><div id="list"></div></aside>
<section id="view"><div class="empty">Pick an invoice on the left. Process them in order to see duplicate and split-invoice detection build up.</div></section></main>
<script>
const $=s=>document.querySelector(s);
function el(tag,props={},kids=[]){const n=document.createElement(tag);for(const[k,v]of Object.entries(props)){if(k==='class')n.className=v;else if(k==='text')n.textContent=v;else n.setAttribute(k,v)}for(const c of[].concat(kids)){if(c!=null)n.append(c instanceof Node?c:document.createTextNode(String(c)))}return n}
let DOCS={};
const money=(v,c)=>v==null?'':Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})+' '+c;
async function api(path,opts){const r=await fetch(path,opts);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
async function refreshStats(){const s=await api('/api/stats');const pays=Object.entries(s.payments).map(([c,v])=>money(v,c)).join(' · ')||'none';
$('#stats').replaceChildren(el('span',{},['Processed ',el('b',{text:s.processed}),' · approved ',el('b',{text:s.approve}),' · held ',el('b',{text:s.hold}),' · rejected ',el('b',{text:s.reject}),' · payments ',el('b',{text:pays})]))}
function renderList(){const ds=$('#dataset').value;const list=$('#list');list.replaceChildren();
for(const d of DOCS[ds].docs){const row=el('div',{class:'doc','data-id':d.doc_id},[el('div',{},[el('b',{text:d.doc_id}),el('small',{text:`${d.vendor} · ${d.template}${d.scanned?' · scan':''}`})]),el('span',{class:'chip '+d.expected,text:'label: '+d.expected})]);
row.onclick=()=>run(d.doc_id,row);list.append(row)}}
async function run(docId,row){document.querySelectorAll('.doc.active').forEach(n=>n.classList.remove('active'));if(row)row.classList.add('active');
$('#view').replaceChildren(el('div',{class:'empty',text:'Processing '+docId+'…'}));
try{const r=await api(`/api/process?dataset=${encodeURIComponent($('#dataset').value)}&doc_id=${encodeURIComponent(docId)}&source=${$('#source').value}`,{method:'POST'});show(r)}
catch(e){$('#view').replaceChildren(el('div',{class:'card'},[el('h2',{text:'Error'}),el('div',{text:e.message})]))}refreshStats()}
function kv(rows){return el('table',{},rows.filter(r=>r[1]!=null&&r[1]!=='').map(([k,v,cls])=>el('tr',{},[el('th',{text:k}),el('td',{class:cls||'',text:v})])))}
function show(r){const inv=r.invoice;const left=el('div',{class:'stack'});const right=el('div',{class:'stack'});
left.append(el('div',{class:'card',id:'preview'},[el('h2',{text:r.filename||r.doc_id}),r.preview?el('img',{src:r.preview,alt:'Invoice preview'}):el('div',{class:'muted',text:'No preview'})]));
const dec=el('div',{class:'card'},[el('h2',{text:'Decision'}),el('span',{class:'decision '+r.decision,text:r.decision})]);
if(r.expected){const ok=r.expected.decision===r.decision;dec.append(el('div',{class:'muted',style:'margin-top:8px',text:`Label says ${r.expected.decision}${r.expected.anomalies.length?' ('+r.expected.anomalies.join(', ')+')':''} ${ok?'✓ match':'✗ mismatch'}`}))}
dec.append(el('div',{class:'muted',style:'margin-top:4px',text:'Extraction: '+r.source+(r.vendor_match?' · vendor matched by '+r.vendor_match:'')}));right.append(dec);
if(r.payment){const p=r.payment;right.append(el('div',{class:'card'},[el('h2',{text:'Payment'}),kv([['Payment',p.payment_id],['Status',p.status+(r.payment_reused?' (already scheduled earlier, not paid again)':'')],['Amount',money(p.amount,p.currency),'num'],['Pay on',p.pay_on+(p.late?' (late)':'')],['To',p.destination.bank_name+' · '+(p.destination.iban||p.destination.account_number)],['Account source','vendor master (never the invoice)']])]))}
const f=el('ul',{class:'findings'});if(r.extraction_error)f.append(el('li',{},[el('span',{class:'sev error',text:'ERROR'}),r.extraction_error]));
for(const x of r.findings){f.append(el('li',{},[el('span',{class:'sev '+x.severity,text:x.severity.toUpperCase()}),x.message]))}
if(!f.children.length)f.append(el('li',{class:'muted',text:'No findings: all checks passed.'}));right.append(el('div',{class:'card'},[el('h2',{text:'Findings'}),f]));
if(r.explanation){const ex=r.explanation;const box=el('div',{class:'card'},[el('h2',{text:'Explanation (review agent)'})]);
if(ex.summary)box.append(el('p',{text:ex.summary}));const ul=el('ul',{class:'findings'});
for(const i of ex.issues||[])ul.append(el('li',{},[el('span',{class:'sev warning',text:i.check.toUpperCase()}),i.explanation,el('div',{class:'muted',text:'Next: '+i.next_step})]));
if(ex.error)ul.append(el('li',{class:'muted',text:'Agent error: '+ex.error}));box.append(ul);
box.append(el('div',{class:'muted',text:'The rules made the decision; the agent only explains it. Tools used: '+((ex.tool_calls||[]).join(', ')||'none')}));right.append(box)}
if(inv){right.append(el('div',{class:'card'},[el('h2',{text:'Extracted fields'}),kv([['Vendor',inv.vendor_name],['Tax ID',inv.vendor_tax_id],['Invoice #',inv.invoice_number],['Invoice date',inv.invoice_date],['Due date',inv.due_date],['PO',inv.po_number],['Subtotal',money(inv.subtotal,inv.currency),'num'],['Tax',money(inv.tax_amount,inv.currency),'num'],['Total',money(inv.total,inv.currency),'num'],['Remit to',inv.remit_to.bank_name+' · '+(inv.remit_to.iban||inv.remit_to.account_number)]]),
el('table',{style:'margin-top:10px'},[el('tr',{},[el('th',{text:'Line item'}),el('th',{text:'Qty'}),el('th',{text:'Amount'})])].concat(inv.line_items.map(li=>el('tr',{},[el('td',{text:li.description}),el('td',{class:'num',text:li.quantity}),el('td',{class:'num',text:money(li.amount,inv.currency)})]))))]))}
if(r.expected&&r.expected.notes.length)left.append(el('div',{class:'card'},[el('h2',{text:'What this test case contains'}),el('div',{class:'muted',text:r.expected.notes.join(' ')})]));
$('#view').replaceChildren(left,right)}
$('#dataset').onchange=()=>{renderList();const o=$('#source option[value=cached]');o.disabled=!DOCS[$('#dataset').value].cached};
$('#reset').onclick=async()=>{await api('/api/reset',{method:'POST'});refreshStats();$('#view').replaceChildren(el('div',{class:'empty',text:'New session started.'}))};
$('#file').onchange=async e=>{const file=e.target.files[0];if(!file)return;$('#view').replaceChildren(el('div',{class:'empty',text:'Extracting '+file.name+' with the live model…'}));
try{const r=await api('/api/upload?name='+encodeURIComponent(file.name),{method:'POST',body:await file.arrayBuffer()});show(r)}catch(err){$('#view').replaceChildren(el('div',{class:'card'},[el('h2',{text:'Error'}),el('div',{text:err.message})]))}refreshStats();e.target.value=''};
(async()=>{DOCS=await api('/api/docs');for(const name of Object.keys(DOCS))$('#dataset').append(el('option',{value:name,text:name}));$('#dataset').onchange();refreshStats()})();
</script></body></html>"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=None,
                   help="dataset folders (default: data/synthetic and data/adversarial in the project)")
    p.add_argument("--out", default=None, help="session output root (default: runs/ui in the project)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    p.add_argument("--explain", action="store_true", help="explain held/rejected invoices with the review agent")
    p.add_argument("--provider", default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--today", default=None, help="YYYY-MM-DD for payment scheduling; default today")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    today = date.fromisoformat(a.today) if a.today else date.today()
    data = [Path(d) for d in a.data] if a.data else [ROOT / "data" / "synthetic", ROOT / "data" / "adversarial"]
    found = [d for d in data if (d / "manifest.jsonl").exists()]
    if not found:
        raise SystemExit(f"No datasets found in {', '.join(str(d) for d in data)}.\n"
                         f"Create one with:  python -m invoice_agent.synth --out {ROOT / 'data' / 'synthetic'}")
    explain_llm = None
    if a.explain:
        from invoice_agent.extract.openai_compat import OpenAICompatExtractor
        explain_llm = OpenAICompatExtractor("gemini", model=a.model)
    session = Session(found, Path(a.out) if a.out else ROOT / "runs" / "ui", a.provider, a.model, today,
                      explain_llm)
    try:
        server = HTTPServer(("127.0.0.1", a.port), make_handler(session))
    except OSError as e:
        raise SystemExit(f"Port {a.port} is busy ({e}). Try:  python -m invoice_agent.ui --port 8800")
    url = f"http://127.0.0.1:{a.port}"
    print(f"Datasets: {', '.join(str(d) for d in found)}")
    print(f"Invoice Agent UI running at {url}  (keep this window open; Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()  # server is already bound
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
