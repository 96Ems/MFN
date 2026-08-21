import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lm2  # registers mfn_dense_* / gpt2mini plumbing
from lm2 import build_phase3, true_param_count

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, "tokenizer_subset")
DATA = os.path.join(ROOT, "data_subset")
CKPT = os.path.join(ROOT, "checkpoints")


def load_data():
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    t = lambda n: torch.from_numpy(np.load(os.path.join(DATA, f"{n}.npy")).copy()).long()
    return t("train"), t("validation"), t("test"), stats


def run_epoch(model, train_ids, optimizer, args, device):
    model.train()
    N = train_ids.shape[0]
    L = (N - 1) // (args.batch * args.seq) * (args.batch * args.seq)
    x = train_ids[0:L].view(args.batch, -1, args.seq)
    y = train_ids[1:L + 1].view(args.batch, -1, args.seq)
    steps = x.shape[1] if not args.limit_steps else min(x.shape[1], args.limit_steps)
    warmup = 300
    total, n_tok, t0 = 0.0, 0, time.time()
    for s in range(steps):
        lr = args.lr * (s + 1) / warmup if s < warmup else \
             args.lr * 0.5 * (1 + np.cos(np.pi * min((s - warmup) / max(steps * args.epochs - warmup, 1), 1)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        out = model(x[:, s].to(device), labels=y[:, s].to(device), return_dict=True)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * y[:, s].numel()
        n_tok += y[:, s].numel()
        if s % 250 == 0:
            print(f"    step {s}/{steps} loss {loss.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
    return total / n_tok, n_tok / (time.time() - t0)


@torch.no_grad()
def evaluate(model, ids, device, batch=16, seq=128):
    model.eval()
    L = (ids.shape[0] - 1) // (batch * seq) * (batch * seq)
    x = ids[0:L].view(batch, -1, seq)
    y = ids[1:L + 1].view(batch, -1, seq)
    total, n = 0.0, 0
    for s in range(x.shape[1]):
        out = model(x[:, s].to(device), labels=y[:, s].to(device), return_dict=True)
        total += out.loss.item() * y[:, s].numel()
        n += y[:, s].numel()
    return total / n


@torch.no_grad()
def generate(model, tokenizer, device, n=90):
    model.eval()
    prompts = ["Once upon a time,", "The little girl found a",
               "Tim and his dog were walking in the park when", "The robot wanted to"]
    out = {}
    for p in prompts:
        ids = tokenizer.encode(p)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        try:
            s = model.generate(inp, max_new_tokens=n, do_sample=True,
                               temperature=0.8, top_k=40, repetition_penalty=1.1)
            g = model.generate(inp, max_new_tokens=n, do_sample=False)
            out[p] = {"sample": tokenizer.decode(s[0].tolist()),
                      "greedy": tokenizer.decode(g[0].tolist())}
        except Exception as exc:
            print(f"    gen failed {p!r}: {exc}", flush=True)
            out[p] = {"sample": "<failed>", "greedy": "<failed>"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mfn_dense", "mfn_dense_nobidir", "mfn_dense_nofatigue",
                             "gru", "gpt2mini"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    train_ids, val_ids, test_ids, stats = load_data()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOK)
    vocab = tokenizer.vocab_size

    if args.model == "mfn_dense":
        hidden = 144
    elif args.model == "mfn_dense_nobidir":
        hidden = 144
    elif args.model == "gru":
        ref = build_phase3("mfn_dense", vocab, 144)
        from lm2 import matched_gru_hidden
        hidden = matched_gru_hidden(true_param_count(ref), vocab)
    elif args.model == "gpt2mini":
        hidden = 120
    else:
        hidden = 144

    model = build_phase3(args.model, vocab, hidden).to(device)
    n_params = true_param_count(model)
    tag = args.tag or args.model
    print(f"[{tag}] params={n_params} hidden={hidden} epochs={args.epochs} "
          f"batch={args.batch} train_tokens={stats['train']['tokens']}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)

    best_val, history = float("inf"), []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, tok_s = run_epoch(model, train_ids, optimizer, args, device)
        val_loss = evaluate(model, val_ids, device)
        ppl = float(np.exp(val_loss))
        bpc = val_loss * stats["validation"]["tokens_per_char"] / np.log(2.0)
        print(f"[{tag}] epoch {ep}/{args.epochs}: train {train_loss:.4f} | "
              f"val {val_loss:.4f} ppl {ppl:.2f} bpc {bpc:.4f} "
              f"({tok_s:.0f} tok/s, {time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": ep, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4), "ppl": round(ppl, 2),
                        "bpc": round(bpc, 4), "tok_s": round(float(tok_s), 1)})
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.join(CKPT, tag), exist_ok=True)
            model.save_pretrained(os.path.join(CKPT, tag))
            tokenizer.save_pretrained(os.path.join(CKPT, tag))

    best_dir = os.path.join(CKPT, tag)
    best_model = type(model).from_pretrained(best_dir)
    best_model = best_model.to(device)
    test_loss = evaluate(best_model, test_ids, device)
    ppl_t = float(np.exp(test_loss))
    bpc_t = test_loss * stats["test"]["tokens_per_char"] / np.log(2.0)
    print(f"[{tag}] TEST loss {test_loss:.4f} ppl {ppl_t:.2f} bpc {bpc_t:.4f}", flush=True)

    gen = generate(best_model, tokenizer, device)
    res = {"tag": tag, "model": args.model, "hidden": hidden, "params": n_params,
           "epochs": args.epochs, "batch": args.batch, "seq": args.seq,
           "lr": args.lr, "seed": args.seed, "device": device,
           "best_val_loss": round(best_val, 4), "test_loss": round(float(test_loss), 4),
           "test_ppl": round(ppl_t, 2), "test_bpc": round(bpc_t, 4),
           "history": history, "samples": gen, "tokenizer": TOK}
    path = os.path.join(ROOT, args.out)
    all_res = json.load(open(path)) if os.path.exists(path) else {}
    all_res[tag] = res
    json.dump(all_res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()