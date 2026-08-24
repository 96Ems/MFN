"""Chat bot generaliste (format UltraChat User:/Assistant:) sur les modeles
MFN 2ZF / Z10 SFT.

Usage:
  .venv/bin/python phase5/chat_ultra.py                    # sft_deep_2zf par defaut
  .venv/bin/python phase5/chat_ultra.py --tag deep_z10
  .venv/bin/python phase5/chat_ultra.py --ask "Explique ce qu'est un chat"
  --t 0.7 --k 40 --max-new 150 --threads 4
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import deepmfn  # noqa: F401  (enregistre mfn_deep_lm dans les Auto classes)

CKPT = os.path.join(ROOT, "phase5", "checkpoints")
TOK = os.path.join(ROOT, "phase4", "tokenizer_subset")
USER, ASST, EOS = "User: ", "\nAssistant: ", "<|endoftext|>"


def load(tag):
    ckpt = os.path.join(CKPT, tag)
    if not os.path.exists(ckpt):
        raise SystemExit(f"pas de checkpoint: {ckpt}")
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TOK)
    model.eval()
    return model, tokenizer


def fmt_history(hist):
    s = ""
    for u, a in hist:
        s += USER + u + ASST + a + "\n"
    return s


def reply(model, tokenizer, hist, args):
    prompt = fmt_history(hist) + ASST.rstrip()
    ids = tokenizer.encode(prompt)
    inp = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_stream(
            inp, max_new_tokens=args.max_new,
            temperature=(args.t if not args.greedy else 0.0),
            top_k=args.k, repetition_penalty=args.rp, greedy=args.greedy)
    dt = time.time() - t0
    text = tokenizer.decode(out[0].tolist())
    # couper au premier marqueur de role ou EOS
    for cut in (EOS, "\n" + USER):
        i = text.find(cut)
        if i > len(prompt):
            text = text[:i]
    ans = text[len(prompt):].strip()
    n_new = max(out.shape[1] - inp.shape[1], 1)
    return ans, n_new / max(dt, 1e-6)


def best_tag():
    p = os.path.join(ROOT, "phase5", "results_sft.json")
    if os.path.exists(p):
        res = json.load(open(p))
        tags = [t for t in res if os.path.exists(os.path.join(CKPT, t))]
        if tags:
            return min(tags, key=lambda t: res[t].get("test_ppl", 1e9))
    for cand in ("sft_deep_2zf", "sft_deep_z10", "deep_2zf"):
        if os.path.exists(os.path.join(CKPT, cand)):
            return cand
    raise SystemExit("aucun checkpoint SFT trouve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ask", default=None, help="one-shot: question unique")
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--t", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--rp", type=float, default=1.1, help="repetition penalty")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tag = args.tag or best_tag()
    model, tokenizer = load(tag)
    print(f"bot: {tag} | t={args.t} k={args.k} rp={args.rp} "
          f"max-new={args.max_new}")

    if args.ask:
        ans, tps = reply(model, tokenizer, [(args.ask, "")], args)
        print(f">> {args.ask}")
        print(f"< {ans}  ({tps:.0f} tok/s)")
        return

    hist = []
    print("chat pret. commandes: quit | /reset | t=0.6 | k=30 | greedy | sample")
    while True:
        try:
            prompt = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "quit":
            break
        if prompt == "/reset":
            hist = []; continue
        if prompt.startswith("t="):
            args.t = float(prompt[2:]); continue
        if prompt.startswith("k="):
            args.k = int(prompt[2:]); continue
        if prompt in ("greedy", "sample"):
            args.greedy = (prompt == "greedy"); continue
        ans, tps = reply(model, tokenizer, hist + [(prompt, "")], args)
        print(f"< {ans}  ({tps:.0f} tok/s)")
        hist.append((prompt, ans))
        if len(hist) > 6:
            hist = hist[-6:]


if __name__ == "__main__":
    main()