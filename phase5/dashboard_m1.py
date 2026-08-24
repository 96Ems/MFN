"""Dashboard temps réel MFN pour Mac M1 — zéro dépendance (stdlib).

  .venv/bin/python phase5/dashboard_m1.py [--port 8766]
  -> http://localhost:8766

Diffère de dashboard.py (laptop/Linux) :
  - sources macOS (loadavg stdlib, pmset thermique, vm_stat, swap) — pas de
    /proc ni nvidia-smi ;
  - COURBES SVG (loss EMA, val par epoch, tok/s, allocation MPS) lues depuis
    les flux logs/metrics_<tag>.jsonl écrits par train_deep/train_sft ;
  - auto-découverte : tout metrics_*.jsonl apparaît sans toucher à ce fichier.
"""
import glob
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# (label, chemin log, motif pgrep — [x] anti auto-match, tag métriques)
WATCH = [
    ("Z30 campagne",  "p5_z30.log",     "arch [z]30",      "deep_z30"),
    ("SFT z30",       "p5_sft_z30.log", "base [z]30",      "sft_deep_z30"),
    ("build corpus",  "p5_build_bigdata.log", "build_bigdata_[s]tream", None),
    ("data SFT",      "p5_sft_data.log",      "sft_[d]ata",             None),
]

STEP_RE = re.compile(r"step (\d+)/(\d+) loss ([\d.]+) \[(\d+)s\]")
EPOCH_RE = re.compile(r"epoch (\d+)/(\d+): train ([\d.]+) \| val ([\d.]+) ppl ([\d.]+)")
TEST_RE = re.compile(r"TEST loss ([\d.]+) ppl ([\d.]+)")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=4).stdout.strip()
    except Exception:
        return ""


def sysinfo():
    load = os.getloadavg()
    therm = sh("pmset -g therm")
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    cpu_limit = int(m[1]) if m else None
    swap = sh("sysctl -n vm.swapusage")
    mem_free_pct = None
    mp = sh("memory_pressure | tail -1")   # ex: "System-wide memory free ...: 42%"
    mm = re.search(r":\s*(\d+)%", mp)
    if mm:
        mem_free_pct = int(mm[1])
    procs = []
    for line in sh("ps -axo pid=,etimes=,rss=,command=").splitlines():
        if not any(k in line for k in ("train_deep", "train_sft",
                                       "build_bigdata", "dashboard_m1")):
            continue
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)$", line)
        if m and "grep" not in m[4]:
            procs.append({"pid": m[1], "etime_s": int(m[2]),
                          "rss_mb": int(m[3]) // 1024,
                          "cmd": m[4][:110]})
    return {"load": [round(x, 2) for x in load],
            "cpu_limit": cpu_limit, "swap": swap,
            "mem_free_pct": mem_free_pct, "procs": procs,
            "time": time.strftime("%H:%M:%S")}


def read_metrics(path):
    """-> (steps[], epochs[], test, dernier rec STEP) depuis metrics_*.jsonl."""
    steps, epochs, test, last_step = [], [], None, None
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = r.get("kind")
                if k == "step":
                    steps.append(r)
                    last_step = r
                elif k == "epoch":
                    epochs.append(r)
                elif k == "test":
                    test = r
    except Exception:
        pass
    return steps, epochs, test, last_step


def svg_curve(pts, color="#58a6ff", h=110, unit=""):
    """pts = [(x_label, y)] -> courbe SVG inline avec min/max/downsample."""
    pts = [(x, y) for x, y in pts if y is not None]
    if len(pts) < 2:
        return "<em class=dim>pas encore assez de points</em>"
    if len(pts) > 300:                       # downsample uniforme
        stride = len(pts) / 300.0
        pts = [pts[int(i * stride)] for i in range(300)]
    ys = [y for _, y in pts]
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1e-9
    W, H = 620, h
    pad = 26
    xy = [(pad + i * (W - 2 * pad) / (len(pts) - 1),
           H - 18 - (y - y0) / (y1 - y0) * (H - 30))
          for i, (_, y) in enumerate(pts)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    first_x, last_y = pts[0][0], pts[-1][1]
    return (f"<svg width={W} height={H} style='background:#0b0f14;border-radius:6px'>"
            f"<polyline points='{poly}' fill='none' stroke='{color}' stroke-width='1.6'/>"
            f"<text x={pad} y=12 fill='#8b949e' font-size='10'>max {max(ys):.4g}</text>"
            f"<text x={pad} y={H-6} fill='#8b949e' font-size='10'>min {min(ys):.4g} {unit}</text>"
            f"<text x={W-140} y={H-6} fill='#c9d1d9' font-size='10'>"
            f"dernier: {last_y:.4g} @ {first_x}</text></svg>")


def run_block(label, logf, pat, mtag=None, mfile_path=None):
    alive = bool(sh(f"pgrep -f '{pat}'")) if pat else False
    path = os.path.join(LOGS, logf) if logf != "-" else None
    tail, prog, epochs_log, test_log = [], None, [], None
    if path and os.path.exists(path):
        txt = open(path, errors="ignore").read()
        tail = [l for l in txt.splitlines() if l.strip()][-12:]
        for m in STEP_RE.finditer(txt):
            prog = {"step": int(m[1]), "total": int(m[2]), "loss": float(m[3])}
        em = EPOCH_RE.findall(txt)
        epochs_log = [{"epoch": int(a), "val": float(d), "ppl": float(e)}
                      for a, b, c, d, e in em][-6:]
        tm = TEST_RE.search(txt)
        test_log = {"loss": float(tm[1]), "ppl": float(tm[2])} if tm else None

    # série JSONL : tag explicite, chemin imposé (auto-découverte), sinon rien
    mfile = mfile_path or (
        os.path.join(LOGS, f"metrics_{mtag}.jsonl") if mtag else None)
    if not (mfile and os.path.exists(mfile)):
        mfile = None
    steps, epochs_j, test_j, last_step = read_metrics(mfile) if mfile else \
        ([], [], None, None)

    eta_min, rate, alloc = None, None, None
    if last_step:
        rate, alloc = last_step.get("tok_s"), last_step.get("alloc_gb")
        rem_tok = max(last_step["steps"] - last_step["step"], 0) * \
            last_step.get("batch", 64) * last_step.get("seq", 128)
        if rate and (last_step["step"] > 2 or alive):
            eta_min = round(rem_tok / rate / 60)

    return {"label": label, "log": logf, "alive": alive, "tail": tail,
            "prog": prog, "epochs": epochs_log + epochs_j[-6:],
            "test": test_j or test_log, "eta_min": eta_min,
            "rate": rate, "alloc": alloc,
            "metrics_file": os.path.basename(mfile) if mfile else None,
            "steps": steps}


def status():
    runs = []
    seen_metrics = set()
    for label, logf, pat, mtag in WATCH:
        b = run_block(label, logf, pat, mtag)
        if b["metrics_file"]:
            seen_metrics.add(os.path.join(LOGS, b["metrics_file"]))
        if b["alive"] or b["tail"] or b["steps"]:
            runs.append(b)
    # runs connus uniquement par leurs métriques (ex: bench ad hoc)
    for mf in sorted(glob.glob(os.path.join(LOGS, "metrics_*.jsonl")),
                     key=os.path.getmtime):
        if mf in seen_metrics:
            continue
        tag = os.path.basename(mf)[len("metrics_"):-len(".jsonl")]
        b = run_block(tag, "-", None, mfile_path=mf)
        if b["steps"]:
            b["label"] = tag
            b["alive"] = (time.time() -
                          (b["steps"][-1]["ts"] if b["steps"] else 0)) < 300
            runs.append(b)
    res = {}
    for jf in ["phase5/results_deep.json", "phase5/results_deep_m1.json",
               "phase5/results_sft.json"]:
        p = os.path.join(ROOT, jf)
        if os.path.exists(p):
            try:
                data = json.load(open(p))
                items = data.items() if isinstance(data, dict) else []
                if isinstance(data, list):
                    items = [(r.get("tag", "?"), r) for r in data]
                for tg, r in items:
                    res[f"{tg}"] = {
                        "params": r.get("params"),
                        "val": r.get("best_val_loss") or r.get("val_loss"),
                        "test_ppl": r.get("test_ppl"), "lr": r.get("lr")}
            except Exception:
                pass
    return {"sys": sysinfo(), "runs": runs, "results": res}


HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>MFN dashboard M1</title><meta http-equiv=refresh content=15>
<style>
body{background:#0d1117;color:#c9d1d9;font:14px/1.45 monospace;margin:18px}
h2{color:#58a6ff;font-size:15px;margin:22px 0 6px}
h3{color:#8b949e;font-size:12px;margin:10px 0 2px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:10px 14px;margin-bottom:12px}
.ok{color:#3fb950}.bad{color:#f85149}.dim{color:#8b949e}.hl{color:#f2cc60}
.warn{background:#3d2e00;border:1px solid #9e6a03;border-radius:6px;padding:6px 10px;display:inline-block}
table{border-collapse:collapse}td,th{padding:2px 10px;text-align:left;
border-bottom:1px solid #21262d}
.bar{background:#21262d;border-radius:4px;height:14px;width:320px;
display:inline-block;vertical-align:middle}
.fill{background:#1f6feb;height:100%;border-radius:4px}
pre{white-space:pre-wrap;margin:4px 0;color:#a5b1bd;font-size:12px}
</style></head><body>
<h1>MFN dashboard · Mac M1 <span class=dim>- refresh 15s</span></h1>
<div class=card id=sys>chargement...</div>
<div id=runs></div>
<h2>Resultats</h2><div class=card id=res></div>
<script>
var d = __DATA__;
var el = function(id){return document.getElementById(id);};
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}

var s = d.sys;
var th = (s.cpu_limit !== null && s.cpu_limit < 100)
  ? ' <span class="warn">THERMIQUE CPU a ' + s.cpu_limit + '% !</span>' : '';
el('sys').innerHTML =
  'load ' + s.load.join(' / ') + ' (8 coeurs)' +
  ' &nbsp;|&nbsp; RAM libre ' + (s.mem_free_pct===null?'?':s.mem_free_pct+'%') +
  ' &nbsp;|&nbsp; swap ' + (s.swap||'?') +
  th + ' &nbsp;|&nbsp; maj ' + s.time;
el('sys').innerHTML += '<br><span class="dim">' + (s.procs.length ? s.procs.length
  + ' process MFN actif(s): ' + s.procs.map(function(p){
      return '#' + p.pid + ' ' + Math.floor(p.etime_s/60) + 'min ' + p.rss_mb + 'Mo';
    }).join(' · ') : 'aucun processus MFN') + '</span>';

function fmt_eta(m){
  if (!m) return '';
  if (m < 90) return 'ETA ' + m + ' min';
  return 'ETA ' + (m/1440).toFixed(1) + ' jours';
}

var h = '';
for (var i=0;i<d.runs.length;i++){
  var r = d.runs[i];
  var st = r.alive ? '<span class="ok">● RUN</span>' : '<span class="dim">○ idle</span>';
  h += '<h2>' + esc(r.label) + ' ' + st;
  if (r.rate) h += ' <span class="hl">' + r.rate + ' tok/s</span>';
  if (r.alloc) h += ' <span class="dim">alloc ' + r.alloc + ' Go</span>';
  if (r.eta_min) h += ' <span class="dim">' + fmt_eta(r.eta_min) + '</span>';
  h += '</h2><div class="card">';
  var c = r.curves || {};
  if (c.loss) h += '<h3>loss par step (EMA)</h3>' + c.loss;
  if (c.tok) h += '<h3>débit tok/s</h3>' + c.tok;
  if (c.alloc) h += '<h3>MPS alloc (Go)</h3>' + c.alloc;
  if (c.val) h += '<h3>val loss par epoch</h3>' + c.val;
  if (!c.loss && !c.tok) h += '<span class="dim">pas de métriques JSONL — '
    + 'lancé avant l\'upgrade logging ?</span>';
  if (r.test) h += '<div class="hl">TEST ppl ' + r.test.ppl + '</div>';
  h += '<pre>' + esc(r.tail.join('\\n')) + '</pre></div>';
}
if (!d.runs.length) h = '<em>aucun run — lance un training ou un bench</em>';
el('runs').innerHTML = h;

var t = '<table><tr><th>run</th><th>params</th><th>val</th><th>test ppl</th><th>lr</th></tr>';
for (var k in d.results){
  var v = d.results[k];
  t += '<tr><td>' + esc(k) + '</td><td>' + v.params + '</td><td>' + (v.val||'')
     + '</td><td>' + (v.test_ppl||'') + '</td><td>' + (v.lr||'') + '</td></tr>';
}
el('res').innerHTML = t + '</table>';
</script></body></html>"""


def render():
    data = status()
    for r in data["runs"]:
        steps = r.pop("steps", [])
        epochs = r.get("epochs", [])
        r["curves"] = {
            "loss": svg_curve([(f"s{x['step']}", x["ema"]) for x in steps],
                              "#f2cc60"),
            "tok": svg_curve([(f"s{x['step']}", x.get("tok_s")) for x in steps],
                             "#3fb950", unit="tok/s"),
            "alloc": svg_curve([(f"s{x['step']}", x.get("alloc_gb"))
                                for x in steps if x.get("alloc_gb")],
                               "#1f6feb", unit="Go"),
            "val": svg_curve([(f"ep{x['epoch']}", x.get("val_loss", x.get("val")))
                              for x in epochs], "#bc8cff"),
        }
    return html_replace(data)


def html_replace(data):
    return HTML.replace("__DATA__", json.dumps(data))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps(status()).encode()
            ctype = "application/json"
        else:
            body = render().encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"dashboard M1 -> http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("", a.port), H).serve_forever()
