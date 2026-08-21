"""Interactive story continuation with the Phase-3 TinyStories models.

Usage:
  .venv/bin/python chat_stories.py                 # best checkpoint (lowest test ppl)
  .venv/bin/python chat_stories.py --tag mfn_dense
  .venv/bin/python chat_stories.py --tag gpt2mini --t 0.6 --k 30
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm2  # noqa: F401  (registers mfn_dense / mfn_dense_* in the Auto classes)
import mfn_lm  # noqa: F401  (registers gru_lm)

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "stories_checkpoints")


def best_tag(phase4=False):
    if phase4:
        res = json.load(open(os.path.join(ROOT, "phase4", "results.json")))
        tags = [t for t in res
                if os.path.exists(os.path.join(ROOT, "phase4", "checkpoints", t))]
        return min(tags, key=lambda t: res[t]["test_ppl"])
    res = json.load(open(os.path.join(ROOT, "stories_results.json")))
    tags = [t for t in res if os.path.exists(os.path.join(CKPT, t))]
    return min(tags, key=lambda t: res[t]["test_ppl"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--phase4", action="store_true",
                    help="use the phase-4 checkpoints (100K stories, GPU)")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--t", type=float, default=0.8, help="temperature")
    ap.add_argument("--k", type=int, default=40, help="top-k")
    ap.add_argument("--greedy", action="store_true", help="greedy (no sampling)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ckpt_root = os.path.join(ROOT, "phase4", "checkpoints") if args.phase4 else CKPT
    tag = args.tag or best_tag(phase4=args.phase4)
    ckpt_dir = os.path.join(ckpt_root, tag)
    if not os.path.exists(ckpt_dir):
        raise SystemExit(f"no checkpoint at {ckpt_dir}")
    print(f"loading {tag} from {ckpt_dir}")
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model.eval()
    print("gen: generate_stream (O(n), état coulé) "
          f"t={args.t} k={args.k} greedy={args.greedy}")
    print("écris le début d'une histoire, ou `quit` / `greedy` / `t=0.6` / `k=20`")
    while True:
        try:
            prompt = input(">> ")
        except (EOFError, KeyboardInterrupt):
            break
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt == "quit":
            break
        if prompt.startswith("t="):
            args.t = float(prompt[2:]); continue
        if prompt.startswith("k="):
            args.k = int(prompt[2:]); continue
        if prompt == "greedy":
            args.greedy = True; continue
        if prompt == "sample":
            args.greedy = False; continue
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long)
        t0 = time.time()
        with torch.no_grad():
            if hasattr(model, "generate_stream"):
                out = model.generate_stream(
                    inp, max_new_tokens=args.max_new,
                    temperature=(args.t if not args.greedy else 0.0),
                    top_k=args.k, repetition_penalty=1.1, greedy=args.greedy)
            elif type(model).__name__ == "GPT2LMHeadModel":
                # manual KV loop (faster than generate() on CPU)
                cur, past, mask = inp, None, torch.ones_like(inp)
                for _ in range(args.max_new):
                    o = model(cur[:, -1:], past_key_values=past,
                              attention_mask=mask, use_cache=True)
                    past, mask = o.past_key_values, torch.cat(
                        [mask, mask.new_ones((1, 1))], 1)
                    logits = o.logits[:, -1] / (args.t if not args.greedy else 1.0)
                    if args.k and logits.shape[1] > args.k:
                        kth = torch.topk(logits, args.k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    nxt = logits.argmax(-1, keepdim=True) if args.greedy \
                        else torch.multinomial(torch.softmax(logits, -1), 1)
                    cur = torch.cat([cur, nxt], 1)
                out = cur
            else:
                gen_kwargs = dict(max_new_tokens=args.max_new,
                                  do_sample=not args.greedy,
                                  temperature=args.t, top_k=args.k,
                                  repetition_penalty=1.1)
                out = model.generate(inp, **gen_kwargs)
        dt = time.time() - t0
        n_new = out.shape[1] - inp.shape[1]
        text = tokenizer.decode(out[0].tolist())
        print("---")
        print(text)
        print(f"--- ({n_new} tokens en {dt:.1f}s = {n_new / max(dt, 1e-6):.0f} tok/s)")


if __name__ == "__main__":
    main()