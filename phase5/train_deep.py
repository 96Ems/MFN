import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import deepmfn  # noqa: E402
from deepmfn import build_deep, MFNDeepForCausalLM  # noqa: E402
from lm2 import true_param_count  # noqa: E402

TOK = os.path.join(ROOT, "phase4", "tokenizer_subset")
DATA = os.path.join(ROOT, "phase4", "data_subset")
CKPT = os.path.join(ROOT, "phase5", "checkpoints")

ARCHES = {
    "2l": dict(widths=[192, 96], topdown=True, skip_fb=False),
    "3l": dict(widths=[192, 128, 96], topdown=True, skip_fb=True),
    "1l_192": dict(widths=[192], topdown=False, skip_fb=False),
    "5l": dict(widths=[192, 128, 128, 128, 96], topdown=True, skip_fb=True),
    "10l": dict(widths=[192, 128, 128, 128, 128, 128, 128, 128, 128, 96],
                topdown=True, skip_fb=True),
}


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
    warmup = 200
    total, n_tok, t0, bad_steps = 0.0, 0, time.time(), 0
    ema = None
    # Ombre des poids : jamais touchée avant un step *réussi* -> permet de
    # rollback si un batch aberrant (loss NaN / outlier / grad NaN) tente de
    # corrompre l'état. Un mauvais batch ne peut plus faire diverger le run.
    with torch.no_grad():
        shadow = [p.detach().clone() for p in model.parameters()]

    def rollback():
        with torch.no_grad():
            for p, sp in zip(model.parameters(), shadow):
                p.copy_(sp)

    def abort(why):
        raise SystemExit(f"ABORT: {why} ({bad_steps} steps mauvais consécutifs) "
                         f"— lower --lr ou inspecter les données")

    for s in range(steps):
        lr = args.lr * (s + 1) / warmup if s < warmup else \
            args.lr * 0.5 * (1 + np.cos(np.pi * min((s - warmup) / max(steps * args.epochs - warmup, 1), 1)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        out = model(x[:, s].to(device), labels=y[:, s].to(device), return_dict=True)
        loss = out.loss
        li = loss.item()
        outlier = ema is not None and li > 4.0 * ema + 4.0
        if not torch.isfinite(loss) or outlier:
            bad_steps += 1
            why = "NON-FINITE loss" if not torch.isfinite(loss) else \
                  f"outlier loss {li:.1f} (EMA {ema:.2f})"
            print(f"    step {s}: {why}, rollback + skip "
                  f"({bad_steps} consécutifs)", flush=True)
            rollback()
            optimizer.zero_grad(set_to_none=True)
            if bad_steps >= 50:
                abort("divergence détectée par rollback")
            continue
        bad_steps = 0
        ema = li if ema is None else 0.99 * ema + 0.01 * li
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            bad_steps += 1
            print(f"    step {s}: non-finite grad norm, rollback + skip "
                  f"({bad_steps} consécutifs)", flush=True)
            rollback()
            optimizer.zero_grad(set_to_none=True)
            if bad_steps >= 50:
                abort("divergence détectée par grad NaN")
            continue
        optimizer.step()
        with torch.no_grad():
            for p, sp in zip(model.parameters(), shadow):
                sp.copy_(p)
        total += li * y[:, s].numel()
        n_tok += y[:, s].numel()
        if s % 250 == 0:
            print(f"    step {s}/{steps} loss {li:.4f} [{time.time()-t0:.0f}s]", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHES))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--out", default="results_deep.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--start-epoch", type=int, default=1,
                    help="epoch de reprise (1 par défaut). >1 -> charge le "
                         "checkpoint phase5/checkpoints/<tag> et backfill "
                         "l'historique des epochs déjà faites")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile mode reduce-overhead (CUDA graphs) — "
                         "fallback eager si indisponible sur cette carte")
    ap.add_argument("--eval-batch", type=int, default=16)
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

    kw = ARCHES[args.arch]
    tag = args.tag or f"deep_{args.arch}"
    ckpt_dir = os.path.join(CKPT, tag)
    # Historique des epochs 1-2 du run deep_2l interrompu (logs p5_2l.log) —
    # backfillé pour garder un historique complet dans results_deep.json.
    BACKFILL = {"2l": [{"epoch": 1, "train_loss": 3.6595, "val_loss": 3.0371,
                        "ppl": 20.84, "bpc": 1.1055, "tok_s": 5993.0},
                       {"epoch": 2, "train_loss": 2.9341, "val_loss": 2.8161,
                        "ppl": 16.71, "bpc": 1.0250, "tok_s": 5318.0}],
                "3l": [{"epoch": 1, "train_loss": 3.6944, "val_loss": 3.0541,
                        "ppl": 21.20, "bpc": 1.1117, "tok_s": 4290.0},
                       {"epoch": 2, "train_loss": 2.9220, "val_loss": 2.7945,
                        "ppl": 16.35, "bpc": 1.0172, "tok_s": 4279.0},
                       {"epoch": 3, "train_loss": 2.7316, "val_loss": 2.6582,
                        "ppl": 14.27, "bpc": 0.9675, "tok_s": 4125.0}]}
    if args.start_epoch > 1 and os.path.isdir(ckpt_dir):
        model = MFNDeepForCausalLM.from_pretrained(ckpt_dir).to(device)
        history = [e for e in BACKFILL.get(args.arch, [])
                   if e["epoch"] < args.start_epoch]
        print(f"[{tag}] RESUMING from {ckpt_dir} — reprise à l'epoch "
              f"{args.start_epoch}", flush=True)
    else:
        if args.start_epoch > 1:
            print(f"[{tag}] --start-epoch {args.start_epoch} mais pas de "
                  f"checkpoint -> init fraîche", flush=True)
        model = build_deep(tag, vocab, **kw).to(device)
        history = []
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[{tag}] torch.compile ACTIVE (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"[{tag}] torch.compile indisponible ({e}) -> eager",
                  flush=True)
    n_params = true_param_count(model)
    print(f"[{tag}] widths={kw['widths']} skip={kw['skip_fb']} "
          f"topdown={kw['topdown']} params={n_params} epochs={args.epochs} "
          f"train_tokens={stats['train']['tokens']}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)
    best_val, history = float("inf"), history
    for ep in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, tok_s = run_epoch(model, train_ids, optimizer, args, device)
        val_loss = evaluate(model, val_ids, device, batch=args.eval_batch)
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
    best_model = type(model).from_pretrained(best_dir).to(device)
    test_loss = evaluate(best_model, test_ids, device, batch=args.eval_batch)
    ppl_t, bpc_t = float(np.exp(test_loss)), \
        test_loss * stats["test"]["tokens_per_char"] / np.log(2.0)
    print(f"[{tag}] TEST loss {test_loss:.4f} ppl {ppl_t:.2f} bpc {bpc_t:.4f}", flush=True)

    res = {"tag": tag, "arch": args.arch, "widths": kw["widths"],
           "topdown": kw["topdown"], "skip_fb": kw["skip_fb"],
           "params": n_params, "epochs": args.epochs, "batch": args.batch,
           "seq": args.seq, "lr": args.lr, "seed": args.seed,
           "device": device, "best_val_loss": round(best_val, 4),
           "test_loss": round(float(test_loss), 4), "test_ppl": round(ppl_t, 2),
           "test_bpc": round(bpc_t, 4), "history": history}
    path = os.path.join(ROOT, "phase5", args.out)
    all_res = json.load(open(path)) if os.path.exists(path) else {}
    all_res[tag] = res
    json.dump(all_res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()