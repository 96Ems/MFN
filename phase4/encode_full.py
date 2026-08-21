import json
import os
import random

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = "/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TRAIN_TXT = os.path.join(CACHE, "TinyStories-train.txt")
VALID_TXT = os.path.join(CACHE, "TinyStories-valid.txt")
TOK = os.path.join(ROOT, "tokenizer_full")
DATA = os.path.join(ROOT, "data")
N_TEST = 2000
SEED = 0
MARKER = "<|endoftext|>"
CHUNK_LINES = 50000


def line_chunks(path, chunk_lines):
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def encode_stream(path, enc, out_path):
    parts, total = [], 0
    with open(path + ".progress", "w") as log:
        for i, chunk in enumerate(line_chunks(path, CHUNK_LINES)):
            ids = enc.encode(chunk)
            parts.append(np.asarray(ids, dtype=np.int32))
            total += len(ids)
            if i % 5 == 0:
                log.write(f"chunk {i} tokens {total}\n")
                log.flush()
                print(f"  chunk {i}: total {total:,} tokens", flush=True)
    arr = np.concatenate(parts)
    np.save(out_path, arr)
    print(f"saved {out_path} ({len(arr):,} tokens)", flush=True)
    return len(arr)


def main():
    enc = AutoTokenizer.from_pretrained(TOK)
    os.makedirs(DATA, exist_ok=True)
    stats = {}
    train_path = os.path.join(DATA, "train.npy")
    if not os.path.exists(train_path):
        print("encoding full train (streamed)...", flush=True)
        n = encode_stream(TRAIN_TXT, enc, train_path)
        stats["train"] = {"tokens": n, "stories": None}
    else:
        stats["train"] = {"tokens": int(np.load(train_path).shape[0]),
                          "stories": None}
        print(f"train.npy exists ({stats['train']['tokens']:,} tokens)",
              flush=True)

    valid_text = open(VALID_TXT, encoding="utf-8").read()
    stories = [p for p in valid_text.split(MARKER) if p.strip()]
    print(f"valid stories: {len(stories)}", flush=True)
    idx = list(range(len(stories)))
    rng = random.Random(SEED)
    test_idx = set(rng.sample(idx, min(N_TEST, len(idx))))
    valid_stories = [stories[i] for i in idx if i not in test_idx]
    test_stories = [stories[i] for i in test_idx]
    for name, slist in (("validation", valid_stories), ("test", test_stories)):
        text = "".join(s + f"\n{MARKER}\n" for s in slist)
        ids = enc.encode(text)
        np.save(os.path.join(DATA, f"{name}.npy"),
                np.asarray(ids, dtype=np.int32))
        n_markers = text.count(MARKER)
        n_chars = len(text) - n_markers * len(MARKER)
        stats[name] = {
            "tokens": int(len(ids)), "chars": int(n_chars),
            "tokens_per_char": round(len(ids) / n_chars, 4),
        }
        print(f"{name}: {len(ids):,} tokens", flush=True)
    with open(os.path.join(DATA, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()