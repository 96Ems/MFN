"""Benchmark variantes MFN sur la GTX 960M : tok/s fwd+bwd+step, VRAM pic, T°.
Objectif : choisir la variante qui rentre ET chauffe le moins pour un run nocturne."""
import subprocess, sys, time, json
sys.path.insert(0, "/home/emericclement/dev/MFN")
import torch
from deepmfn import build_deep

VOCAB, SEQ, STEPS = 4096, 128, 30
VARIANTS = [
    ("1l_192", dict(widths=[192],            topdown=False, skip_fb=False, n_threads=1)),
    ("2l",     dict(widths=[192,96],         topdown=True,  skip_fb=False, n_threads=1)),
    ("3l",     dict(widths=[192,128,96],     topdown=True,  skip_fb=True,  n_threads=1)),
    ("2l2t",   dict(widths=[192,96],         topdown=True,  skip_fb=False, n_threads=2, thread_fb=True)),
    ("3l2t",   dict(widths=[192,128,96],     topdown=True,  skip_fb=True,  n_threads=2, thread_fb=True)),
    ("3l3t",   dict(widths=[192,128,96],     topdown=True,  skip_fb=True,  n_threads=3, thread_fb=True)),
    # --- petites unités : la taille vient du NOMBRE de modules, pas de la largeur ---
    ("s4x64",  dict(widths=[64]*4,           topdown=True,  skip_fb=True,  n_threads=1)),
    ("s6x64",  dict(widths=[64]*6,           topdown=True,  skip_fb=True,  n_threads=1)),
    ("s8x64",  dict(widths=[64]*8,           topdown=True,  skip_fb=True,  n_threads=1)),
    ("s6x96",  dict(widths=[96]*6,           topdown=True,  skip_fb=True,  n_threads=1)),
    ("s8x96",  dict(widths=[96]*8,           topdown=True,  skip_fb=True,  n_threads=1)),
    ("t2s4x64",dict(widths=[64]*4,           topdown=True,  skip_fb=True,  n_threads=2, thread_fb=True)),
    ("t2s6x64",dict(widths=[64]*6,           topdown=True,  skip_fb=True,  n_threads=2, thread_fb=True)),
]
BATCHES = [32, 64]

def gpu_temp():
    out = subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out

def smi_mem():
    out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)

results = []
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
print(f"{'variant':<8}{'B':>4}{'params':>12}{'tok/s':>9}{'ms/step':>9}{'VRAMpic':>9}{'smi':>6}{'T°':>5}", flush=True)
for name, kw in VARIANTS:
    if ONLY and name not in ONLY:
        continue
    for B in BATCHES:
        torch.manual_seed(0)
        try:
            model = build_deep(name, VOCAB, **kw).to("cuda")
        except Exception as e:
            print(f"{name:<8}{B:>4}  BUILD FAIL {e}", flush=True); continue
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
        y = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
        # warmup 3 steps
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=y, return_dict=True); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); smi0 = smi_mem()
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=y, return_dict=True); out.loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn): opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        tps = STEPS * B * SEQ / dt
        ms = dt / STEPS * 1000
        t_after = gpu_temp()
        row = dict(variant=name, B=B, params=sum(p.numel() for p in model.parameters()),
                   tok_s=round(tps), ms_step=round(ms,1), vram_mb=round(peak),
                   smi_mb=smi_mem(), temp=t_after)
        results.append(row)
        print(f"{name:<8}{B:>4}{row['params']:>12,}{row['tok_s']:>9}{row['ms_step']:>9}"
              f"{row['vram_mb']:>8}M{row['smi_mb']:>6}{row['temp']:>5}", flush=True)
        del model, opt, x, y
        torch.cuda.empty_cache()
        time.sleep(20)  # laisser retomber la chaleur entre configs
json.dump(results, open("../logs/bench_variants.json","w"), indent=2)
print("DONE -> ../logs/bench_variants.json", flush=True)
