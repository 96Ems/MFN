import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfn import MyelinFatigueNet

torch.set_num_threads(int(torch.get_num_threads()))


def make_batch(batch_size, T, d):
    return torch.randn(T, batch_size, d)


def targets_for(x, delays):
    return {k: torch.cat([torch.zeros(k, x.shape[1], x.shape[2]), x[:-k]], dim=0)
            for k in delays}


class LSTMBaseline(nn.Module):
    def __init__(self, d, H, dout):
        super().__init__()
        self.cell = nn.LSTMCell(d, H)
        self.readout = nn.Linear(H, dout)
        self.H = H

    def forward(self, x_seq):
        B = x_seq.shape[1]
        h = torch.zeros(B, self.H)
        c = torch.zeros(B, self.H)
        outs = []
        for x_t in x_seq:
            h, c = self.cell(x_t, (h, c))
            outs.append(self.readout(h))
        return torch.stack(outs)


class MambaLike(nn.Module):
    def __init__(self, d, d_model=64, d_state=16, expand=8, dout=10,
                 kernel=4):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = d_model
        self.kernel = kernel
        self.in_proj = nn.Linear(d, self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=kernel,
                              groups=self.d_inner, padding=kernel - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state,
                                bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dout)

    def forward(self, x_seq):
        T, B, d = x_seq.shape
        u = self.in_proj(x_seq)
        u = u.permute(1, 2, 0)
        u = F.silu(self.conv(u)[..., :T])
        u = u.permute(2, 0, 1)
        u = F.silu(u)
        x_dbl = self.x_proj(u)
        dt = self.dt_proj(x_dbl[..., :self.dt_rank])
        dt = F.softplus(dt)
        dt = torch.exp(torch.clamp(dt, max=10.0))
        Bw = x_dbl[..., self.dt_rank:self.dt_rank + self.d_state]
        Cw = x_dbl[..., self.dt_rank + self.d_state:]
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state)
        outs = []
        for t in range(T):
            a = torch.exp(dt[t].unsqueeze(-1) * A.unsqueeze(0))
            h = a * h + dt[t].unsqueeze(-1) * \
                Bw[t].unsqueeze(1) * u[t].unsqueeze(-1)
            y = (h * Cw[t].unsqueeze(1)).sum(-1) + self.D * u[t]
            outs.append(self.out_proj(y))
        return torch.stack(outs)


class SLSTMLike(nn.Module):
    def __init__(self, d, H, dout, heads=4):
        super().__init__()
        self.H = H
        self.heads = heads
        self.Wz = nn.Linear(d, H)
        self.Wi = nn.Linear(H, heads)
        self.Wf = nn.Linear(H, heads)
        self.Wo = nn.Linear(H, H)
        self.readout = nn.Linear(H, dout)

    def forward(self, x_seq):
        B = x_seq.shape[1]
        H, heads, hh = self.H, self.heads, self.H // self.heads
        c = torch.zeros(B, heads, hh)
        n = torch.zeros(B, heads, 1)
        h = torch.zeros(B, H)
        outs = []
        for x_t in x_seq:
            z = torch.tanh(self.Wz(x_t)).view(B, heads, hh)
            i = torch.exp(torch.clamp(self.Wi(h), max=5.0))[:, :, None]
            f = torch.sigmoid(self.Wf(h))[:, :, None]
            o = torch.sigmoid(self.Wo(h))
            c = f * c + i * z
            n = f * n + i
            htilde = c / torch.clamp(n, min=1.0)
            h = o * htilde.view(B, H)
            outs.append(self.readout(h))
        return torch.stack(outs)


MODELS = {
    'lstm': lambda d, H, dout: LSTMBaseline(d, H, dout),
    'mamba': lambda d, H, dout: MambaLike(d, d_model=64, dout=dout),
    'xlstm': lambda d, H, dout: SLSTMLike(d, H, dout),
}
HS = {'lstm': 144, 'mamba': 1, 'xlstm': 288}


def train(kind, d, T, dout, delays, B, n_epochs, seed):
    torch.manual_seed(seed)
    model = MODELS[kind](d, HS[kind], dout)
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
    return final, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['lstm', 'mamba', 'xlstm'])
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--out', type=str, default='baselines_t1.json')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    delays = [5, 10, 20]
    results = {}
    for kind in args.models:
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            res, pc = train(kind, 10, 100, 10, delays, 32, args.epochs, seed)
            per_seed.append(res)
            print(f"[{kind} seed={seed}] {res} #params={pc} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            results.setdefault(kind, {})['params'] = pc
            results.setdefault(kind, {})['seeds'] = seeds
            for k in delays:
                results.setdefault(kind, {}).setdefault(f'k{k}', {}).setdefault(
                    'runs', []).append(res[k])
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)
        agg = {}
        for k in delays:
            vals = [s[k] for s in per_seed]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            results[kind][f'k{k}']['mean'] = mean
            results[kind][f'k{k}']['std'] = var ** 0.5
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()