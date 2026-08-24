"""Corpus 'big' anti-famine : 400K train (+2K val / 2K test), même tokenizer
BPE-4096 subset -> comparabilité directe avec P0..P4. Exclut les 100K du
subset original (indices recomputés seed 0) puis sample 400K du complément."""
import json, os, random
import numpy as np
from transformers import PreTrainedTokenizerFast

ROOT = "/home/emericclement/dev/MFN/phase4"
CACHE = os.path.join(ROOT, "..", ".cache_ts")
TRAIN_TXT = "/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt"
VALID_TXT = "/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-valid.txt"
TOK = os.path.join(ROOT, "tokenizer_subset")
DATA = os.path.join(ROOT, "data_big")
STOR = os.path.join(ROOT, "data_big_stories")
MARKER = "<|endoftext|>"
N_TRAIN, N_VAL, N_TEST = 400_000, 2_000, 2_000

def count_stories(path):
    return sum(1 for line in open(path, encoding="utf-8") if line.strip() == MARKER)

def stream_select(path, keep):
    buf, idx = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == MARKER:
                if idx in keep and buf:
                    yield "\n".join(buf)
                idx += 1; buf = []
            elif s:
                buf.append(s)
    if idx in keep and buf:
        yield "\n".join(buf)

def build_split(path, n, tag, exclude=None):
    total = count_stories(path)
    rng = random.Random({"train": 10, "validation": 11, "test": 12}[tag])
    pool = set(range(total)) - (exclude or set())
    keep = set(rng.sample(sorted(pool), min(n, len(pool))))
    print(f"[{tag}] total={total} keep={len(keep)}", flush=True)
    stories = list(stream_select(path, keep))
    text = f"\n{MARKER}\n".join(stories) + f"\n{MARKER}\n"
    os.makedirs(STOR, exist_ok=True)
    open(os.path.join(STOR, f"{tag}.txt"), "w", encoding="utf-8").write(text)
    return text, len(stories), keep

# exclusion des 100K du subset original (seed 0, mêmes fonctions)
total_tr = count_stories(TRAIN_TXT)
old = set(random.Random(0).sample(range(total_tr), 100_000))
print(f"exclusion des {len(old)} stories du subset original", flush=True)

train_text, n_tr, _ = build_split(TRAIN_TXT, N_TRAIN, "train", exclude=old)
val_text, n_v, _ = build_split(VALID_TXT, N_VAL, "validation")
test_text, n_t, _ = build_split(VALID_TXT, N_TEST, "test")
print(f"sampled: {n_tr}/{n_v}/{n_t}", flush=True)

fast = PreTrainedTokenizerFast.from_pretrained(TOK)
os.makedirs(DATA, exist_ok=True)
stats = {}
for name, text in (("train", train_text), ("validation", val_text), ("test", test_text)):
    ids = fast.encode(text)
    np.save(os.path.join(DATA, f"{name}.npy"), np.asarray(ids, dtype=np.int32))
    n_chars = len(text) - text.count(MARKER) * len(MARKER)
    stats[name] = {"tokens": int(len(ids)), "chars": int(n_chars),
                   "tokens_per_char": round(len(ids)/n_chars, 4)}
    print(f"{name}: {len(ids):,} tokens ({len(ids)/1e6:.1f}M)", flush=True)
json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=2)
print("DONE big corpus", flush=True)
