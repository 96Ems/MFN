import argparse
import json
import time

import torch

from mfn import MyelinFatigueNet, run_sequence

torch.set_num_threads(int(torch.get_num_threads()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=500)
    args = ap.parse_args()
    torch.manual_seed(0)
    d, T, dout, B = 10, 100, 10, 32
    delays = [10]
    n_epochs = args.epochs

    model = MyelinFatigueNet(d, 64, dout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def make_batch():
        return torch.randn(T, B, d)

    def targets(x):
        k = delays[0]
        return torch.cat([torch.zeros(k, B, dout), x[:-k]], dim=0)

    t0 = time.time()
    for epoch in range(n_epochs):
        x = make_batch()
        tgt = targets(x)
        opt.zero_grad()
        y, _, _ = run_sequence(model, x, B, 64)
        mask = torch.zeros(T, 1, 1)
        mask[10:] = 1.0
        loss = ((y - tgt) ** 2 * mask).sum() / (mask.sum() * B * dout)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    print(f"trained MFN k=10: loss={loss.item():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        all_phi_lam, all_phi_psi = [], []
        n_seqs = 100
        for s in range(n_seqs):
            x = torch.randn(T, 1, d)
            h_lam, h_psi, phi_lam, phi_psi = model.init_state(1, 64)
            phi_lam_seq, phi_psi_seq = [], []
            for t in range(T):
                h_lam, h_psi, phi_lam, phi_psi, _ = model(
                    x[t], h_lam, h_psi, phi_lam, phi_psi)
                phi_lam_seq.append(phi_lam.squeeze(0).clone())
                phi_psi_seq.append(phi_psi.squeeze(0).clone())
            all_phi_lam.append(torch.stack(phi_lam_seq))
            all_phi_psi.append(torch.stack(phi_psi_seq))
    P_lam = torch.stack(all_phi_lam).mean(0)   # (T, H)
    P_psi = torch.stack(all_phi_psi).mean(0)

    mean_lam = P_lam.mean().item()
    mean_psi = P_psi.mean().item()
    sat_lam = (P_lam > 0.5).float().mean().item()
    sat_psi = (P_psi > 0.5).float().mean().item()
    corr = torch.corrcoef(torch.stack([
        P_lam.mean(1), P_psi.mean(1)]))[0, 1].item()

    L = torch.stack(all_phi_lam)   # (S, T, H)
    Ps = torch.stack(all_phi_psi)
    per_neuron = []
    for j in range(P_lam.shape[1]):
        vl = torch.var(L[:, :, j])
        vp = torch.var(Ps[:, :, j])
        if vl > 1e-8 and vp > 1e-8:
            c = torch.corrcoef(
                torch.stack([L[:, :, j].reshape(-1).float(),
                             Ps[:, :, j].reshape(-1).float()]))[0, 1].item()
            per_neuron.append(c)
        per_neuron_mean = sum(per_neuron) / len(per_neuron)
    per_neuron_max = max(per_neuron)
    per_neuron_min = min(per_neuron)
    sat03_lam = (P_lam > 0.3).float().mean().item()
    sat03_psi = (P_psi > 0.3).float().mean().item()

    results = {
        'mean_fatigue_lam': mean_lam,
        'mean_fatigue_psi': mean_psi,
        'saturated_fraction_lam': sat_lam,
        'saturated_fraction_psi': sat_psi,
        'fraction_above_0.3_lam': sat03_lam,
        'fraction_above_0.3_psi': sat03_psi,
        'pearson_corr_phi_lam_phi_psi': corr,
        'pearson_per_neuron_mean': per_neuron_mean,
        'pearson_per_neuron_min': per_neuron_min,
        'pearson_per_neuron_max': per_neuron_max,
        'final_loss_k10': loss.item(),
        'params': sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(results, indent=2), flush=True)
    with open('fatigue.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()