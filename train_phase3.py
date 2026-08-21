import argparse
import json
import os
import time

import numpy as np
import torch

from lm2 import build_phase3, matched_gru_hidden, true_param_count

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.join(ROOT, "tokenizer_stories")
ENCODED = os.path.join(ROOT, "data", "stories_encoded")
CKPT = os.path.join(ROOT, "stories_checkpoints")


def load_data():
    stats = json.load(open(os.path.join(ENCODED, "stats.json")))
    train = np.load(os.path.join(ENCODED, "train.npy"))
    val = np.load(os.path.join(ENCODED, "validation.npy"))
    test = np.load(os.path.join(ENCODED, "test.npy"))
    return (torch.from_numpy(train.copy()).long(),
            torch.from_numpy(val.copy()).long(),
            torch.from_numpy(test.copy()).long(), stats)


def run_train_epoch(model, train_ids, optimizer, args, device):
    model.train()
    N = train_ids.shape[0]
    L = (N - 1) // (args.batch * args.seq) * (args.batch * args.seq)
    x = train_ids[0:L].view(args.batch, -1, args.seq)
    y = train_ids[1 : L + 1].view(args.batch, -1, args.seq)
    steps = x.shape[1]
    if args.limit_steps:
        steps = min(steps, args.limit_steps)

    warmup = 200
    total_loss = 0.0
    n_tok = 0
    step = 0
    t0 = time.time()
    for s in range(steps):
        xi = x[:, s].to(device)
        yi = y[:, s].to(device)
        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            frac = (step - warmup) / max(steps * args.epochs - warmup, 1)
            lr = args.lr * 0.5 * (1.0 + np.cos(np.pi * min(frac, 1.0)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        out = model(xi, labels=yi, return_dict=True)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * yi.numel()
        n_tok += yi.numel()
        step += 1
        if step % 200 == 0:
            print(f"    step {step}/{steps} loss {loss.item():.4f} "
                  f"(lr {lr:.2e}) [{time.time() - t0:.0f}s]", flush=True)
    elapsed = time.time() - t0
    return total_loss / n_tok, n_tok / elapsed


@torch.no_grad()
def evaluate(model, ids, device, eval_batch=16, seq=128):
    model.eval()
    L = (ids.shape[0] - 1) // (eval_batch * seq) * (eval_batch * seq)
    x = ids[0:L].view(eval_batch, -1, seq)
    y = ids[1 : L + 1].view(eval_batch, -1, seq)
    total_loss = 0.0
    n_tok = 0
    for s in range(x.shape[1]):
        out = model(x[:, s].to(device), labels=y[:, s].to(device),
                    return_dict=True)
        total_loss += out.loss.item() * y[:, s].numel()
        n_tok += y[:, s].numel()
    return total_loss / n_tok


@torch.no_grad()
def generate_samples(model, tokenizer, device, n=90):
    model.eval()
    prompts = [
        "Once upon a time,",
        "The little girl found a",
        "Tim and his dog were walking in the park when",
        "The robot wanted to",
    ]
    samples = {}
    for p in prompts:
        ids = tokenizer.encode(p)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        try:
            s = model.generate(
                inp, max_new_tokens=n, do_sample=True, temperature=0.8,
                top_k=40, repetition_penalty=1.1)
            s_dec = tokenizer.decode(s[0].tolist())
            g = model.generate(inp, max_new_tokens=n, do_sample=False)
            g_dec = tokenizer.decode(g[0].tolist())
        except Exception as exc:  # gpt2 generation quirks
            print(f"    generation failed for {p!r}: {exc}", flush=True)
            s_dec = g_dec = "<failed>"
        samples[p] = {"sample": s_dec, "greedy": g_dec}
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mfn_dense", "mfn_dense_nobidir",
                             "mfn_dense_nofatigue", "gru", "gpt2mini"])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--out", default="stories_results.json")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    train_ids, val_ids, test_ids, stats = load_data()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    vocab = tokenizer.vocab_size

    if args.hidden is not None:
        hidden = args.hidden
    elif args.model == "mfn_dense":
        hidden = 144
    elif args.model == "mfn_dense_nobidir":
        hidden = 144
    elif args.model == "mfn_dense_nofatigue":
        hidden = 144
    elif args.model == "gru":
        ref = build_phase3("mfn_dense", vocab_size=vocab, hidden_size=144)
        hidden = matched_gru_hidden(true_param_count(ref), vocab)
    elif args.model == "gpt2mini":
        hidden = 120
    else:
        raise ValueError(args.model)

    model = build_phase3(args.model, vocab_size=vocab, hidden_size=hidden,
                         dropout=args.dropout)
    n_params = true_param_count(model)
    tag = args.tag or args.model
    print(f"[{tag}] params={n_params} hidden={hidden} epochs={args.epochs} "
          f"batch={args.batch} seq={args.seq} lr={args.lr} "
          f"train_tokens={stats['train']['tokens']}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)

    best_val = float("inf")
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, tok_s = run_train_epoch(model, train_ids, optimizer, args, device)
        val_loss = evaluate(model, val_ids, device)
        elapsed = time.time() - t0
        ppl = np.exp(val_loss)
        bpc = val_loss * stats["validation"]["tokens_per_char"] / np.log(2.0)
        print(f"[{tag}] epoch {ep}/{args.epochs}: train {train_loss:.4f} | "
              f"val {val_loss:.4f} ppl {ppl:.2f} bpc {bpc:.4f} "
              f"({tok_s:.0f} tok/s, {elapsed:.0f}s)", flush=True)
        history.append({
            "epoch": ep, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "ppl": round(float(ppl), 2),
            "bpc": round(float(bpc), 4), "tok_s": round(float(tok_s), 1),
            "elapsed_s": round(elapsed, 1),
        })
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.join(CKPT, tag), exist_ok=True)
            model.save_pretrained(os.path.join(CKPT, tag))
            tokenizer.save_pretrained(os.path.join(CKPT, tag))

    best_dir = os.path.join(CKPT, tag)
    best_model = type(model).from_pretrained(best_dir)
    test_loss = evaluate(best_model, test_ids, device)
    test_ppl = np.exp(test_loss)
    test_bpc = test_loss * stats["test"]["tokens_per_char"] / np.log(2.0)
    print(f"[{tag}] TEST (best ckpt) loss {test_loss:.4f} ppl {test_ppl:.2f} "
          f"bpc {test_bpc:.4f}")

    samples = generate_samples(best_model, tokenizer, device)

    result = {
        "tag": tag, "model": args.model, "hidden": hidden,
        "params": n_params, "epochs": args.epochs, "batch": args.batch,
        "seq": args.seq, "lr": args.lr, "dropout": args.dropout,
        "seed": args.seed, "threads": args.threads,
        "best_val_loss": round(best_val, 4),
        "test_loss": round(float(test_loss), 4),
        "test_ppl": round(float(test_ppl), 2),
        "test_bpc": round(float(test_bpc), 4),
        "tokens_per_char_val": stats["validation"]["tokens_per_char"],
        "history": history, "samples": samples,
        "tokenizer": TOK_DIR,
    }
    out_path = os.path.join(ROOT, args.out)
    all_res = {}
    if os.path.exists(out_path):
        all_res = json.load(open(out_path))
    all_res[tag] = result
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"[{tag}] saved -> {out_path}")


if __name__ == "__main__":
    main()