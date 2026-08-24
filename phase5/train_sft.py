import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lm2  # noqa: F401  (mfn_dense_lm)
import mfn_lm  # noqa: F401  (gru_lm)
import deepmfn  # noqa: F401  (mfn_deep_lm)
from lm2 import true_param_count  # noqa: E402

DATA = os.path.join(ROOT, "phase5", "data_sft")
P4 = os.path.join(ROOT, "phase4", "checkpoints")
P5 = os.path.join(ROOT, "phase5", "checkpoints")
CKPT = os.path.join(ROOT, "phase5", "checkpoints")

BASES = {
    "mfn_dense": os.path.join(P4, "mfn_dense"),
    "gru": os.path.join(P4, "gru"),
    "deep_2l": os.path.join(P5, "deep_2l"),
    "deep_3l": os.path.join(P5, "deep_3l"),
    "deep_2zf": os.path.join(P5, "deep_2zf"),
    "deep_z10": os.path.join(P5, "deep_z10"),
    "deep_z30": os.path.join(P5, "deep_z30"),
}


def load_data():
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    t = lambda n: torch.from_numpy(np.load(os.path.join(DATA, f"{n}.npy")).copy()).long()
    m = lambda n: torch.from_numpy(np.load(os.path.join(DATA, f"{n}_mask.npy")).copy())
    return (t("train"), m("train"), t("validation"), m("validation"), stats)


def masked_loss(logits, labels, mask):
    """CE restricted to masked (assistant) positions."""
    lsm = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
    flat_mask = mask.reshape(-1).float()
    n = flat_mask.sum().clamp(min=1.0)
    return -(lsm.gather(1, labels.reshape(-1, 1)).squeeze(-1) * flat_mask).sum() / n


@torch.no_grad()
def evaluate(model, ids, mask, device, batch=16, seq=256):
    model.eval()
    L = (ids.shape[0] - 1) // (batch * seq) * (batch * seq)
    x = ids[0:L].view(batch, -1, seq)
    y = ids[1:L + 1].view(batch, -1, seq)
    mk = mask[0:L].view(batch, -1, seq)
    total, n = 0.0, 0
    for s in range(x.shape[1]):
        out = model(x[:, s].to(device), return_dict=True, labels=None)
        l = masked_loss(out.logits, y[:, s].to(device), mk[:, s].to(device))
        total += l.item() * mk[:, s].sum().item()
        n += mk[:, s].sum().item()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=sorted(BASES))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--device", default=None,
                    help="force device: cuda | mps | cpu (défaut: auto)")
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--out", default="results_sft.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        device = args.device
    elif args.cuda and torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}", flush=True)

    train_ids, train_mask, val_ids, val_mask, stats = load_data()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "phase4", "tokenizer_subset"))
    base_dir = BASES[args.base]
    print(f"loading base {args.base} from {base_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_dir, trust_remote_code=True).to(device)
    n_params = true_param_count(model)
    tag = f"sft_{args.base}"
    print(f"[{tag}] params={n_params} epochs={args.epochs} "
          f"train_tokens={stats['train']['tokens']} "
          f"loss_tokens={stats['train']['loss_tokens']}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    N = train_ids.shape[0]
    L = (N - 1) // (args.batch * args.seq) * (args.batch * args.seq)
    x = train_ids[0:L].view(args.batch, -1, args.seq)
    y = train_ids[1:L + 1].view(args.batch, -1, args.seq)
    mk = train_mask[0:L].view(args.batch, -1, args.seq)
    steps = x.shape[1]
    if args.limit_steps:
        steps = min(steps, args.limit_steps)

    best_val, history = float("inf"), []
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n_tok, t0 = 0.0, 0, time.time()
        for s in range(steps):
            out = model(x[:, s].to(device), return_dict=True)
            loss = masked_loss(out.logits, y[:, s].to(device), mk[:, s].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * mk[:, s].sum().item()
            n_tok += mk[:, s].sum().item()
            if s % 200 == 0:
                print(f"    step {s}/{steps} loss {loss.item():.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        lr_sched.step()
        val_loss = evaluate(model, val_ids, val_mask, device)
        ppl = float(np.exp(val_loss))
        tok_s = n_tok / (time.time() - t0)
        print(f"[{tag}] epoch {ep}/{args.epochs}: SFT loss {total/n_tok:.4f} | "
              f"val {val_loss:.4f} ppl {ppl:.2f} ({tok_s:.0f} tok/s, "
              f"{time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": ep, "sft_loss": round(total / n_tok, 4),
                        "val_loss": round(val_loss, 4), "ppl": round(ppl, 2)})
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.join(CKPT, tag), exist_ok=True)
            model.save_pretrained(os.path.join(CKPT, tag))
            tokenizer.save_pretrained(os.path.join(CKPT, tag))

    # qualitative check: empty assistant reply to a user question
    best_dir = os.path.join(CKPT, tag)
    best_model = type(model).from_pretrained(best_dir).to(device)
    best_model.eval()
    samples = {}
    for q in ("What is the capital of France?", "Write a short story about a dog.",
              "Explain gravity simply."):
        pre = tokenizer.encode("User: " + q + "\nAssistant: ")
        inp = torch.tensor([pre], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = best_model.generate_stream(
                inp, max_new_tokens=80, temperature=0.8, top_k=40,
                repetition_penalty=1.1, greedy=False)
        samples[q] = tokenizer.decode(out_ids[0].tolist())
        print(f"[{tag}] Q: {q}\n  A: {samples[q][len('User: '+q+chr(10)+'Assistant: '):][:200]}",
              flush=True)

    res = {"tag": tag, "base": args.base, "params": n_params,
           "epochs": args.epochs, "batch": args.batch, "seq": args.seq,
           "lr": args.lr, "seed": args.seed, "device": device,
           "best_val_loss": round(best_val, 4), "val_ppl": round(float(np.exp(best_val)), 2),
           "history": history, "samples": samples}
    path = os.path.join(ROOT, "phase5", args.out)
    all_res = json.load(open(path)) if os.path.exists(path) else {}
    all_res[tag] = res
    json.dump(all_res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()