import argparse
import json
import time

import torch
import torch.nn as nn

from mfn import MyelinFatigueNet, parameter_count, run_sequence

torch.set_num_threads(max(1, torch.get_num_threads()))


class GRUBaseline(nn.Module):
    def __init__(self, d, H, dout):
        super().__init__()
        self.cell = nn.GRUCell(d, H)
        self.readout = nn.Linear(H, dout)
        self.H = H

    def forward(self, x_seq):
        B = x_seq.shape[1]
        h = torch.zeros(B, self.H)
        outs = []
        for x_t in x_seq:
            h = self.cell(x_t, h)
            outs.append(self.readout(h))
        return torch.stack(outs)


def make_batch(batch_size, T, d):
    return torch.randn(T, batch_size, d)


def targets_for(x, delays):
    return {k: torch.cat([torch.zeros(k, x.shape[1], x.shape[2]), x[:-k]], dim=0)
            for k in delays}


class OneGate(nn.Module):
    def forward(self, x):
        return torch.ones_like(x[..., :x.shape[-1] // 2])


class FusedMFN(nn.Module):
    """MFN variantes: fatigue/bidir/gates off, ou shared decay."""

    def __init__(self, input_size, hidden_size, output_size, beta=0.2,
                 no_fatigue=False, no_bidir=False, no_gates=False,
                 shared_decay=False):
        super().__init__()
        self.core = MyelinFatigueNet(input_size, hidden_size, output_size,
                                     beta)
        self.no_fatigue = no_fatigue
        if shared_decay:
            self.core.decay_psi = self.core.decay_lam
        if no_gates:
            self.core.gate_psi2lam = OneGate()
            self.core.gate_lam2psi = OneGate()
        if no_bidir:
            self.core.gate_lam2psi = OneGate()
            with torch.no_grad():
                self.core.W_lam2psi.weight.zero_()
            self.core.W_lam2psi.requires_grad_(False)

    def forward(self, x):
        B = x.shape[1]
        h_lam, h_psi, phi_lam, phi_psi = self.core.init_state(B, self.core.H)
        outs = []
        for x_t in x:
            if self.no_fatigue:
                phi_lam = torch.zeros_like(phi_lam)
                phi_psi = torch.zeros_like(phi_psi)
            h_lam, h_psi, phi_lam, phi_psi, y_t = self.core(
                x_t, h_lam, h_psi, phi_lam, phi_psi)
            outs.append(y_t)
        return torch.stack(outs)


def build_model(kind, d, H, dout):
    if kind == 'mfn':
        return FusedMFN(d, H, dout)
    if kind == 'mfnnofatigue':
        return FusedMFN(d, H, dout, no_fatigue=True)
    if kind == 'mfnnobidir':
        return FusedMFN(d, H, dout, no_bidir=True)
    if kind == 'mfnnogates':
        return FusedMFN(d, H, dout, no_gates=True)
    if kind == 'mfnshareddecay':
        return FusedMFN(d, H, dout, shared_decay=True)
    raise ValueError(kind)


def train(kind, d, T, dout, delays, B, n_epochs, seed, H=64):
    torch.manual_seed(seed)
    if kind == 'gru':
        model = GRUBaseline(d, 170, dout)
    else:
        model = build_model(kind, d, H, dout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    final = {}
    for epoch in range(n_epochs):
        x = make_batch(B, T, d)
        tgt = targets_for(x, delays)
        opt.zero_grad()
        y = model(x)
        total = 0.0
        for k in delays:
            mask = torch.zeros(T, 1, 1)
            mask[k:] = 1.0
            n_el = mask.sum() * B * dout
            lk = ((y - tgt[k]) ** 2 * mask).sum() / n_el
            total = total + lk
            final[k] = lk.item()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return final, parameter_count(model)


def true_param_count(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['mfn', 'gru'])
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--d', type=int, default=10)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--dout', type=int, default=10)
    ap.add_argument('--B', type=int, default=32)
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--out', type=str, default='results.json')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    delays = [5, 10, 20]

    results = {}
    for kind in args.models:
        pcount = None
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            res, pc = train(kind, args.d, args.T, args.dout, delays,
                            args.B, args.epochs, seed, args.H)
            per_seed.append(res)
            if kind == 'gru':
                pcount = true_param_count(GRUBaseline(args.d, 170, args.dout))
            else:
                pcount = true_param_count(build_model(kind, args.d, args.H,
                                                      args.dout))
            print(f"[{kind} seed={seed}] {res}  #params={pcount} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        # aggregate mean/std over seeds
        agg = {}
        for k in delays:
            vals = [s[k] for s in per_seed]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            agg[f'k{k}'] = {'mean': mean, 'std': var ** 0.5, 'runs': vals}
        results[kind] = {'params': pcount, 'seeds': seeds, **agg}

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", args.out)

    print(f"\n{'Model':<16}" + ''.join(f"{'k='+str(k):>14}" for k in delays))
    for kind, r in results.items():
        row = f"{kind:<16}"
        for k in delays:
            row += f"{r[f'k{k}']['mean']:.4f}±{r[f'k{k}']['std']:.4f}".rjust(14)
        print(row)


if __name__ == '__main__':
    main()