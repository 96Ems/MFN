"""Probing de spécialisation des zones (MFN-2Z).

Principe : on masque (mise à zéro propre, puis restauration) les contributions
de chaque zone et on mesure la dégradation de la perte sur des CLASSES de
tokens :
  - frequent  : top 20% tokens du train
  - rare      : queue (1 occurrence) -> capacité "lexique"
  - punct     : ponctuation/fonction -> syntaxe locale
Si l'ablation de la zone A touche surtout une classe et celle de la zone B une
autre -> spécialisation fonctionnelle émergente entre zones couplées.

Usage:
  python probe_zones.py --model ../phase5/checkpoints/deep_2z \
                        --data ../phase4/data_subset [--max-batches 40]
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deepmfn import MFNDeepForCausalLM  # noqa: E402


def load_val(data_dir):
    val = np.load(os.path.join(data_dir, "validation.npy")).copy()
    return torch.from_numpy(val).long()


def token_classes(train_ids, vocab):
    counts = np.bincount(train_ids, minlength=vocab)
    order = np.argsort(counts)
    n = len(order)
    cls = torch.zeros(vocab, dtype=torch.long)   # 0 mid
    cls[torch.from_numpy(order[:n // 5])] = 2    # rare
    cls[torch.from_numpy(order[-n // 5:])] = 1   # frequent
    for t in [b".", b",", b"'", b'"']:
        pass
    return cls, counts


@torch.no_grad()
def eval_masked(model, ids, cls_of_tok, device, mask_fn=None,
                batch=32, seq=128, max_batches=40):
    """Perte moyenne par classe de token, avec masque optionnel appliqué."""
    model.eval()
    L = (len(ids) - 1) // (batch * seq) * (batch * seq)
    x = ids[0:L].view(batch, -1, seq)
    y = ids[1:L + 1].view(batch, -1, seq)
    saved = []
    if mask_fn is not None:
        saved = mask_fn(model)          # applique et retourne les tenseurs sauvés
    sums = {0: 0.0, 1: 0.0, 2: 0.0}
    cnts = {0: 0, 1: 0, 2: 0}
    total = 0.0
    nb = min(x.shape[1], max_batches)
    for s in range(nb):
        xi, yi = x[:, s].to(device), y[:, s].to(device)
        out = model(xi, return_dict=True)
        logits = out.logits[:, :-1]
        tgt = yi[:, 1:]
        loss_tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
            reduction="none").view(tgt.shape)
        c = cls_of_tok[tgt]             # (B, T-1)
        for k in (0, 1, 2):
            m = (c == k)
            sums[k] += loss_tok[m].sum().item(); cnts[k] += int(m.sum())
        total += loss_tok.mean().item()
    if saved:                            # restauration
        for p, v in saved:
            p.copy_(v)
    per_cls = {["mid", "freq", "rare"][k]: round(sums[k] / max(cnts[k], 1), 4)
               for k in (0, 1, 2)}
    return round(total / nb, 4), per_cls


# ---- interventions par masquage de poids (propres + réversibles) ----

def mask_zone_head(stack, t):
    """Neutralise la contribution READOUT de la zone t dans final_proj."""
    cols = slice(2 * stack.zone_widths[t][-1] * t,
                 2 * stack.zone_widths[t][-1] * (t + 1))
    W = stack.final_proj.weight
    saved = [(W, W.clone())]
    with torch.no_grad():
        W[:, cols] = 0.0
    return saved


def mask_lat_edge(stack, src, dst):
    """Coupe l'arête latérale dirigée src->dst à tous les étages."""
    saved = []
    for l in range(len(stack.lat)):
        fb = stack.lat[l][dst][src]
        for p in [fb.W.weight, fb.W.bias, fb.u, fb.v, fb.b]:
            saved.append((p, p.clone()))
            with torch.no_grad():
                p.zero_()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "phase4/data_subset"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MFNDeepForCausalLM.from_pretrained(args.model,
                                               trust_remote_code=True).to(device)
    stack = model.stack
    assert getattr(stack, "zone_mode", False), "le modèle n'est pas en mode zones"

    train_ids = np.load(os.path.join(args.data, "train.npy"))[:5_000_000]
    ids = load_val(args.data)
    vocab = model.config.vocab_size
    cls_of_tok, _ = token_classes(train_ids, vocab)

    res = {}
    res["baseline"] = eval_masked(model, ids, cls_of_tok, device,
                                  batch=args.batch, seq=args.seq,
                                  max_batches=args.max_batches)
    res["ablate_zoneB_head"] = eval_masked(
        model, ids, cls_of_tok, device,
        mask_fn=lambda st: mask_zone_head(st, 1),
        batch=args.batch, seq=args.seq, max_batches=args.max_batches)
    res["ablate_zoneA_head"] = eval_masked(
        model, ids, cls_of_tok, device,
        mask_fn=lambda st: mask_zone_head(st, 0),
        batch=args.batch, seq=args.seq, max_batches=args.max_batches)
    res["cut_B_to_A"] = eval_masked(
        model, ids, cls_of_tok, device,
        mask_fn=lambda st: mask_lat_edge(st, 1, 0),
        batch=args.batch, seq=args.seq, max_batches=args.max_batches)
    res["cut_A_to_B"] = eval_masked(
        model, ids, cls_of_tok, device,
        mask_fn=lambda st: mask_lat_edge(st, 0, 1),
        batch=args.batch, seq=args.seq, max_batches=args.max_batches)

    print(json.dumps(res, indent=2))
    out = os.path.join(ROOT, "phase5", "probe_zones.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
