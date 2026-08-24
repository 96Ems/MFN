"""Dashboard temps réel MFN — zéro dépendance (stdlib).

  python dashboard.py [--port 8765]
  -> http://localhost:8765

Agrège : GPU (nvidia-smi), RAM/load, processus actifs, progression parsée des
logs (step X/Y + [Ns] -> ETA), queue, historique des résultats JSON.
"""
import json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# (label, chemin log, motif pgrep — [x] bracket trick anti auto-match)
WATCH = [
    ("2Z train",      "p5_2z.log",     "arch [2]z"),
    ("3L big-corpus", "p5_3l_big.log", "deep_[3]l_big"),
    ("grille 3l3t",   "p5_3l3t.log",   "arch [3]l3t"),
    ("bigdata",       "bigdata.log",   "build_bigdata_[s]tream"),
    ("SFT UltraChat", "p5_sft_2zf.log", "train_[s]ft"),
    ("Z10",           "p5_z10.log",    "arch [z]10"),
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


def gpu():
    out = sh("nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,"
             "memory.used,memory.total --format=csv,noheader,nounits")
    if "," in out:
        t, u, mu, mt = [x.strip() for x in out.split(",")]
        return {"temp": int(t), "util": int(u), "mem_used": int(mu),
                "mem_total": int(mt)}
    return {"temp": None, "util": None, "mem_used": None, "mem_total": None}


def sysinfo():
    load = open("/proc/loadavg").read().split()[:3]
    mem = sh("free -m | awk '/Mem:/{print $3\"/\"$2}'")
    return {"load": " ".join(load), "ram": mem,
            "uptime_s": int(float(open('/proc/uptime').read().split()[0]))}


def procs():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
            if any(k in cmd for k in ("train_deep", "build_bigdata",
                                      "train_sft", "dashboard.py")):
                ete = sh(f"ps -o etimes= -p {pid}").strip()
                out.append({"pid": pid, "cmd": cmd[:120], "etime_s": int(ete or 0)})
        except Exception:
            pass
    return out


def parse_log(path):
    """(dernières lignes, progression, epochs vus, test)"""
    if not os.path.exists(path):
        return [], None, [], None
    txt = open(path, errors="ignore").read()
    lines = [l for l in txt.splitlines() if l.strip()][-14:]
    if "Traceback" in txt:
        # aplatir les vieux tracebacks : ne garder que le message final
        tail = [l for l in txt.splitlines() if l.strip()]
        tail = tail[-2:]
        lines = ["\u26a0 \u00e9chec archiv\u00e9 (traceback) — cause :"]
        lines += tail
    prog = None
    for m in STEP_RE.finditer(txt):
        s, tot, loss, el = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        prog = {"step": s, "total": tot, "loss": loss}
        if s > 10:
            rate = el / s                       # s/step
            prog["eta_min"] = round((tot - s) * rate / 60)
    age = int(time.time() - os.path.getmtime(path))
    if prog:
        prog["age_s"] = age                     # pulsation : âge du dernier point
    epochs = [{"ep": int(m[1]), "tot": int(m[2]), "train": float(m[3]),
               "val": float(m[4]), "ppl": float(m[5])}
              for m in EPOCH_RE.finditer(txt)]
    tm = TEST_RE.search(txt)
    return lines, prog, epochs, ({"loss": float(tm[1]), "ppl": float(tm[2])}
                                 if tm else None)


def status():
    runs = []
    for label, logf, pat in WATCH:
        alive = bool(sh(f"pgrep -f '{pat}'"))
        lines, prog, epochs, test = parse_log(os.path.join(LOGS, logf))
        if alive or lines:
            runs.append({"label": label, "log": logf, "alive": alive,
                         "prog": prog, "epochs": epochs[-4:], "test": test,
                         "tail": lines})
    res = {}
    for jf in ["phase5/results_deep.json", "phase4/results.json"]:
        p = os.path.join(ROOT, jf)
        if os.path.exists(p):
            try:
                for tag, r in json.load(open(p)).items():
                    res[f"{jf.split('/')[0][:2]}:{tag}"] = {
                        "params": r.get("params"), "val": r.get("best_val_loss") or r.get("val_loss"),
                        "test_ppl": r.get("test_ppl"), "lr": r.get("lr")}
            except Exception:
                pass
    q = os.path.join(LOGS, "p5_queue.log")
    return {"time": time.strftime("%H:%M:%S"), "gpu": gpu(), "sys": sysinfo(),
            "procs": procs(), "runs": runs, "results": res,
            "queue": open(q, errors="ignore").read().splitlines()[-8:]
            if os.path.exists(q) else []}


HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>MFN dashboard</title><meta http-equiv=refresh content=10>
<style>
body{background:#0d1117;color:#c9d1d9;font:14px/1.45 monospace;margin:18px}
h2{color:#58a6ff;font-size:15px;margin:22px 0 6px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:10px 14px;margin-bottom:12px}
.ok{color:#3fb950}.bad{color:#f85149}.dim{color:#8b949e}.hl{color:#f2cc60}
table{border-collapse:collapse}td,th{padding:2px 10px;text-align:left;
border-bottom:1px solid #21262d}
.bar{background:#21262d;border-radius:4px;height:14px;width:280px;display:inline-block}
.fill{background:#1f6feb;height:100%;border-radius:4px}
pre{white-space:pre-wrap;margin:4px 0;color:#a5b1bd}
</style></head><body>
<h1>MFN dashboard <span class=dim>- refresh 10s</span></h1>
<div class=card id=sys>chargement...</div>
<div id=runs></div>
<h2>Queue</h2><div class=card><pre id=queue></pre></div>
<h2>Resultats</h2><div class=card id=res></div>
<script>
var d = __DATA__;
var el = function(id){return document.getElementById(id);};
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}

el('sys').innerHTML = 'GPU ' + d.gpu.temp + 'C - util ' + d.gpu.util +
 '% - VRAM ' + d.gpu.mem_used + '/' + d.gpu.mem_total + ' Mo' +
 ' &nbsp;|&nbsp; CPU load ' + d.sys.load +
 ' &nbsp;|&nbsp; RAM ' + d.sys.ram + ' Mo' +
 ' &nbsp;|&nbsp; maj ' + d.time;

var h = '';
for (var i=0;i<d.runs.length;i++){
  var r = d.runs[i];
  var st = r.alive ? '<span class="ok">RUN</span>' : '<span class="dim">idle</span>';
  h += '<h2>' + esc(r.label) + ' ' + st + '</h2><div class="card">';
  if (r.prog){
    var p = Math.min(100, r.prog.step/r.prog.total*100);
    var beat = '';
    if (r.prog.age_s !== undefined){
      var a = Math.floor(r.prog.age_s/60);
      beat = ' <span class="dim">[point maj il y a ' + a + ' min]</span>';
      if (r.prog.step === 0 && a < 10) beat += ' <span class="ok">actif, prochain point ~step 250</span>';
    }
    h += '<div class="bar"><div class="fill" style="width:'+p+'%"></div></div>'
       + ' step ' + r.prog.step + '/' + r.prog.total
       + ' (' + p.toFixed(0) + '%) loss ' + r.prog.loss
       + (r.prog.eta_min ? ' · ETA ' + r.prog.eta_min + ' min' : '') + beat;
  }
  h += '<pre>' + esc(r.tail.join('\\n')) + '</pre>';
  if (r.test) h += '<div class="hl">TEST ppl ' + r.test.ppl + '</div>';
  h += '</div>';
}
if (!d.runs.length) h = '<em>aucun run</em>';
el('runs').innerHTML = h;

el('queue').innerHTML = esc(d.queue.join('\\n'));

var t = '<table><tr><th>run</th><th>params</th><th>val</th><th>test ppl</th><th>lr</th></tr>';
for (var k in d.results){
  var v = d.results[k];
  t += '<tr><td>' + esc(k) + '</td><td>' + v.params + '</td><td>' + (v.val||'')
     + '</td><td>' + (v.test_ppl||'') + '</td><td>' + (v.lr||'') + '</td></tr>';
}
el('res').innerHTML = t + '</table>';
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        data = status()
        if self.path == "/api/status":
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = HTML.replace("__DATA__", json.dumps(data)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"dashboard -> http://localhost:{a.port} (et LAN)", flush=True)
    ThreadingHTTPServer(("", a.port), H).serve_forever()
