import argparse
import time

import torch

from mfn_lm import build_lm


def bench_cell(model, hidden, batch, seq, n_steps, device="cpu"):
    """Per-step cost of the *recurrence only* (no embedding, no logits).

    Calls model.embed_tokens / model.cell instead of the full forward, so the
    V x H output projection does not dominate the measurement.
    """
    e = torch.randn(batch, seq, hidden)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=1e-4)
    dense = hasattr(model, "core") or "dense_mfn" in type(model).__name__.lower()
    t0 = time.time()
    for step in range(n_steps):
        h_lam = torch.zeros(batch, hidden)
        h_psi = torch.zeros(batch, hidden)
        phi_lam = torch.zeros(batch, hidden)
        phi_psi = torch.zeros(batch, hidden)
        for t in range(seq):
            if dense:
                h_lam, h_psi, phi_lam, phi_psi, _ = model.core(
                    e[:, t], h_lam, h_psi, phi_lam, phi_psi)
            elif hasattr(model, "cell"):
                h_lam, h_psi, phi_lam, phi_psi = model.cell(
                    e[:, t], h_lam, h_psi, phi_lam, phi_psi)
            else:
                h = model.gru_cell(e[:, t], h_lam)
                h_lam = h
        loss = (h_lam.mean() if dense or hasattr(model, "cell") else h.mean())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    per_step_s = (time.time() - t0) / n_steps
    # timesteps processed per second (batch x seq x n_steps / elapsed)
    tps = batch * seq * n_steps / max(time.time() - t0, 1e-9)
    return per_step_s, tps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mfndiag", "gru", "dense_mfn"])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    hidden = args.hidden or {"mfndiag": 256, "gru": 199, "dense_mfn": 75}[args.model]
    model = build_lm(args.model, vocab_size=4096, hidden_size=hidden)
    model.train()
    per_step, tps = bench_cell(model, hidden, args.batch, args.seq, args.steps)
    print(f"{args.model} H={hidden} cell-only: {per_step*1000:.0f} ms/step "
          f"({tps:.0f} timesteps/s, {args.threads} thread{'s' if args.threads>1 else ''})")