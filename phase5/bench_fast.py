import time, sys, torch
sys.path.insert(0, "/home/emericclement/dev/MFN")
from deepmfn import build_deep
V, SEQ, STEPS = 4096, 128, 40
kw = dict(widths=[160,112,80], topdown=True, skip_fb=True, n_threads=2,
          thread_fb=True, fast=True, zone_widths=[[160,112,80],[96,64,48]],
          zone_betas=[0.2,0.6], zone_rates=[1,2])
print(f"{'B':>4} {'thr':>4} {'tok/s':>8} {'s/step':>8} {'VRAM':>7}", flush=True)
for B in (32, 48, 64):
  for thr in (2, 4):
    torch.set_num_threads(thr); torch.manual_seed(0)
    m = build_deep("2zf", V, **kw).to("cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, SEQ), device="cuda")
    y = torch.randint(0, V, (B, SEQ), device="cuda")
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        o = m(x, labels=y, return_dict=True); o.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        o = m(x, labels=y, return_dict=True); o.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    torch.cuda.synchronize(); dt = time.time()-t0
    print(f"{B:>4} {thr:>4} {STEPS*B*SEQ/dt:>8.0f} {dt/STEPS*1000:>7.0f}ms "
          f"{torch.cuda.max_memory_allocated()/2**20:>5.0f}M", flush=True)
    del m, opt, x, y; torch.cuda.empty_cache(); time.sleep(10)
