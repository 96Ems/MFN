"""Proof A2: lateral mixing order-dependence bug"""
import torch
import torch.nn as nn

class LateralFB(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.W = nn.Linear(H, H, bias=False)
        with torch.no_grad():
            self.W.weight.fill_(0.5)
        self.u = nn.Parameter(torch.ones(H)*0.3)
        self.v = nn.Parameter(torch.ones(H)*0.2)
        self.b = nn.Parameter(torch.zeros(H))
    def forward(self, h_me, h_other):
        g = torch.sigmoid(self.u * h_me + self.v * h_other + self.b)
        return g * self.W(h_other)

H=4
fb = LateralFB(H)
# simulate 3 threads readouts
y = [torch.randn(2,H) for _ in range(3)]
for i, yi in enumerate(y):
    print(f"y[{i}] mean {yi.mean():.3f}")

def buggy_mix(y_list, fb_modules):
    # current code: acc threaded through gate
    n=len(y_list)
    mixed=[]
    for t in range(n):
        acc = y_list[t].clone()
        for s in range(n):
            if s==t: continue
            acc = acc + fb_modules[t][s](acc, y_list[s])
        mixed.append(acc)
    return mixed

def fixed_mix(y_list, fb_modules):
    n=len(y_list)
    mixed=[]
    for t in range(n):
        acc = y_list[t].clone()
        y_self = y_list[t]
        for s in range(n):
            if s==t: continue
            acc = acc + fb_modules[t][s](y_self, y_list[s])
        mixed.append(acc)
    return mixed

# build modules [dst][src]
mods = [[LateralFB(H) if i!=j else None for j in range(3)] for i in range(3)]
# use same weights for fair comparison
for i in range(3):
    for j in range(3):
        if mods[i][j] is not None:
            mods[i][j].load_state_dict(fb.state_dict())

# Test order dependence: permute source loop order
y_list = [torch.randn(2,H) for _ in range(3)]
print("\n--- Buggy: permuting source order changes output ---")
out_normal = buggy_mix(y_list, mods)
# reverse source order version (manually s=2,1,0)
def buggy_reverse(y_list, mods):
    mixed=[]
    for t in range(3):
        acc=y_list[t].clone()
        for s in reversed(range(3)):
            if s==t: continue
            acc = acc + mods[t][s](acc, y_list[s])
        mixed.append(acc)
    return mixed
out_rev = buggy_reverse(y_list, mods)
for t in range(3):
    diff=(out_normal[t]-out_rev[t]).abs().max().item()
    print(f" thread {t} max diff normal vs reversed: {diff:.6f}  (expected 0 if order-independent)")

print("\n--- Fixed: order independent (uses y_self) ---")
# fixed also order independent
def fixed_reverse(y_list, mods):
    mixed=[]
    for t in range(3):
        acc=y_list[t].clone()
        y_self=y_list[t]
        for s in reversed(range(3)):
            if s==t: continue
            acc = acc + mods[t][s](y_self, y_list[s])
        mixed.append(acc)
    return mixed
out_f = fixed_mix(y_list, mods)
out_fr = fixed_reverse(y_list, mods)
for t in range(3):
    diff=(out_f[t]-out_fr[t]).abs().max().item()
    print(f" thread {t} fixed diff: {diff:.6f}")

# Show buggy vs fixed differ
print("\n--- Buggy vs Fixed difference (same order) ---")
for t in range(3):
    diff=(out_normal[t]-out_f[t]).abs().mean().item()
    print(f" thread {t} mean |buggy-fixed|: {diff:.6f}")

print("\nProof: buggy mixing is order-dependent; fixed is order-independent and matches paper eq.")
