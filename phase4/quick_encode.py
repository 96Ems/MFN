import json
import os

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, "tokenizer_subset", "tokenizer.json")
STORIES = os.path.join(ROOT, "data_subset_stories")
DATA = os.path.join(ROOT, "data_subset")
CHUNK_LINES = 10000


def encode_file(path, enc):
    ids_tot = []
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line)
            if len(buf) >= CHUNK_LINES:
                seg = "".join(buf)
                buf = []
                ids_tot.extend(enc.encode(seg).ids)
                # source the memory pressure from long lists
        if buf:
            ids_tot.extend(enc.encode("".join(buf)).ids)
    return np.asarray(ids_tot, dtype=np.int32)


def main():
    enc = Tokenizer.from_file(TOK)
    os.makedirs(DATA, exist_ok=True)
    stats = {}
    for name in ("train", "validation", "test"):
        arr = encode_file(os.path.join(STORIES, f"{name}.txt"), enc)
        np.save(os.path.join(DATA, f"{name}.npy"), arr)
        raw = open(os.path.join(STORIES, f"{name}.txt"), encoding="utf-8").read()
        n_chars = len(raw) - raw.count("<|endoftext|>") * len("<|endoftext|>")
        stats[name] = {"tokens": int(arr.shape[0]), "chars": int(n_chars),
                       "tokens_per_char": round(arr.shape[0] / n_chars, 4)}
        print(f"{name}: {arr.shape[0]:,} tokens, "
              f"{arr.shape[0] / n_chars:.3f} tok/char", flush=True)
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()