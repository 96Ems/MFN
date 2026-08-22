"""Proof A3/B2: phi gamma =0.5 under wd, phi clamped, bar calc"""
import math
# B2: gamma init 0 -> sigmoid 0.5, tau ~2 steps, wd pulls to 0
import torch
gamma0 = torch.sigmoid(torch.tensor(0.0)).item()
print(f"gamma init (w=0): {gamma0}  tau~{1/(1-gamma0):.1f} steps (EMA half-life ~1 step) -> 'fatigue' is quasi-instantaneous")

# With wd=0.1, AdamW does p <- p - lr*wd*p  => w_gamma decays to 0 each step unless grad overcomes
# Show phi bounded but needs clamp if |h|>1 ever
print("\nphi EMA bounded proof: if |h|<=1, phi in [0,1] by induction (convex combo).")
print(" GRU h in [-1,1] => |h|<=1 => phi stays in [0,1] without clamp. But readout y unbounded => future risk.")
# clamp proof
for phi in [0.0, 0.5, 0.9, 1.0]:
    for h_abs in [0.2, 1.0, 1.5]:
        for g in [0.5, 0.9]:
            nxt = g*phi + (1-g)*h_abs
            print(f" phi {phi:.1f} h {h_abs:.1f} gamma {g:.1f} -> nxt {nxt:.3f}", end="; ")
    print()
print(" -> if h_abs>1, phi>1 possible -> 1-phi negative => sign flip. clamp needed.")

print("\n--- Bar calc proof ---")
L0=2.4013
N0=1153440
N4=2612384
for alpha in [0.057, 0.076, 0.095]:
    bar = L0 * (N4/N0)**(-alpha)
    bar_nonemb_ratio = 2.37  # from scaling.md? compute actual from param counts non-emb
    # non-emb: emb = V*H0, V=4096
    # 1L: H0=144 emb=589824 nonemb=1153440-589824=563616
    # 3L: H0=192 emb=786432 nonemb=2612384-786432=1825952 ratio=3.239
    ratio_nonemb = 1825952/563616
    bar_nonemb = L0 * ratio_nonemb**(-alpha)
    print(f" alpha {alpha:.3f}  total bar {bar:.4f}  non-emb bar (r={ratio_nonemb:.3f}) {bar_nonemb:.4f}")
print(f"\nOld paper bar 2.30 corresponds to alpha~{math.log(L0/2.30)/math.log(N4/N0):.4f} on total params -- off by {2.30-2.256:.3f} nat")
