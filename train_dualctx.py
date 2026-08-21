import argparse
import json
import time

import torch
import torch.nn as nn

from mfn import MyelinFatigueNet
from baselines_t1 import LSTMBaseline, SLSTMLike

torch.set_num_threads(int(torch.get_num_threads()))


def make_batch(batch_size, T):
    a = torch.randn(T, batch_size, 5)
    b = torch.randn(T, batch_size, 5)
    return torch.cat([a, b], dim=-1), a, b


def dual_targets(a, b, ka, kb):
    def shift(x, k):
        return torch.cat([torch.zeros(k, x.shape[1], x.shape[2]), x[:-k]],
                         dim=0)
    return torch.cat([shift(a, ka), shift(b, kb)], dim=-1)


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


class FusedMFN(nn.Module):
    def __init__(self, d, H, dout, no_bidir=False):
        super().__init__()
        self.core = MyelinFatigueNet(d, H, dout)
        self.no_bidir = no_bidir
        if no_bidir:
            for p in self.core.gate_psi2lam.parameters():
                p.requires_grad_(False)
            for p in self.core.gate_lam2psi.parameters():
                p.requires_grad_(False)
            for p in self.core.W_lam2psi.parameters():
                p.requires_grad_(False)

    def forward(self, x_seq):
        B = x_seq.shape[1]
        h_lam, h_psi, phi_lam, phi_psi = self.core.init_state(B, self.core.H)
        outs = []
        for x_t in x_seq:
            h_lam, h_psi, phi_lam, phi_psi, y_t = self.core(
                x_t, h_lam, h_psi, phi_lam, phi_psi)
            outs.append(y_t)
        return torch.stack(outs)


def train(kind, d, T, dout, B, n_epochs, seed, ka=5, kb=10):
    torch.manual_seed(seed)
    if kind == 'mfn':
        model = FusedMFN(d, 64, dout)
    elif kind == 'mfnnobidir':
        model = FusedMFN(d, 64, dout, no_bidir=True)
    elif kind == 'gru':
        model = GRUBaseline(d, 170, dout)
    elif kind == 'lstm':
        model = LSTMBaseline(d, 144, dout)
    else:
        raise ValueError(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    final = {}
    for epoch in range(n_epochs):
        x, a, b = make_batch(B, T)
        tgt = dual_targets(a, b, ka, kb)
        opt.zero_grad()
        y = model(x)
        n_el = B * dout * (T - ka)
        la = ((y[:, :, :5] - tgt[:, :, :5]) ** 2)[ka:].sum() / n_el
        lb = ((y[:, :, 5:] - tgt[:, :, 5:]) ** 2)[kb:].sum() / n_el
        total = la + lb
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        final['mse_a'] = la.item()
        final['mse_b'] = lb.item()
        final['total'] = total.item()
    return final, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['mfn', 'mfnnobidir', 'gru', 'lstm'])
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--out', type=str, default='dualctx.json')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    results = {}
    for kind in args.models:
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            res, pc = train(kind, 10, 100, 10, args.B if hasattr(args, 'B')
                            else 32, args.epochs, seed)
            per_seed.append(res)
            print(f"[{kind} seed={seed}] {res} #params={pc} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            results.setdefault(kind, {})['params'] = pc
            results.setdefault(kind, {})['seeds'] = seeds
            for m in ('mse_a', 'mse_b', 'total'):
                results[kind].setdefault(m, {}).setdefault('runs', []).append(
                    res[m])
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
        for m in ('mse_a', 'mse_b', 'total'):
            vals = [s[m] for s in per_seed]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            results[kind][m]['mean'] = mean
            results[kind][m]['std'] = var ** 0.5
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()