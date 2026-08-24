"""Corpus 'big' anti-famine — VERSION STREAMING (mémoire constante).

v1 matérialisait tout le texte (~6 GB) -> swap -> crash machine. Ici :
  - sélection au fil du fichier, écriture .txt incrémentale
  - encodage par lots de 2000 histoires -> petits blocs int32
  - np.concatenate final seulement (360 MB pour 90M tokens)
Exclut les 100K histoires du subset original (indices seed 0 recomputés).
"""
import json, os, random
import numpy as np
from transformers import PreTrainedTokenizerFast

ROOT = "/home/emericclement/dev/MFN/phase4"
TRAIN_TXT = ("/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--"
             "TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/"
             "TinyStories-train.txt")
VALID_TXT = ("/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--"
             "TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/"
             "TinyStories-valid.txt")
TOK = os.path.join(ROOT, "tokenizer_subset")
DATA = os.path.join(ROOT, "data_big")
STOR = os.path.join(ROOT, "data_big_stories")
MARKER = "<|endoftext|>"
N_TRAIN, N_VAL, N_TEST = 400_000, 2_000, 2_000


def iter_stories(path):
    """Yield (idx, story_text) sans jamais charger tout le fichier."""
    buf, idx = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == MARKER:
                if buf:
                    yield idx, "\n".join(buf)
                idx += 1; buf = []
                if idx % 200_000 == 0:
                    print(f"  ...{idx} histoires scannées", flush=True)
            elif s:
                buf.append(s)
    if buf:
        yield idx, "\n".join(buf)


def build_split(path, n, tag, exclude=None):
    os.makedirs(STOR, exist_ok=True)
    total_hint = {"train": 2_119_718, "validation": 21_989, "test": 21_989}[tag]
    rng = random.Random({"train": 10, "validation": 11, "test": 12}[tag])
    pool_max = total_hint
    keep = set(rng.sample(range(pool_max), n))
    if exclude:
        keep -= exclude
    print(f"[{tag}] cible={n} gardés(après excl.)={len(keep)}", flush=True)

    out_txt = open(os.path.join(STOR, f"{tag}.txt"), "w", encoding="utf-8")
    blocks, buf_texts, kept, seen_idx = [], [], 0, 0
    first = True

    def flush_encode(tok, force=False):
        nonlocal buf_texts, blocks, first
        if not buf_texts:
            return
        if len(buf_texts) >= 2000 or force:
            ids = tok.encode(f"\n{MARKER}\n".join(buf_texts) + f"\n{MARKER}\n")
            blocks.append(np.asarray(ids, dtype=np.int32))
            buf_texts = []

    # passe unique : sélection à la volée
    tok = None
    n_chars = 0
    for idx, story in iter_stories(path):
        seen_idx = idx + 1
        if idx not in keep:
            continue
        if tok is None:                      # lazy: tokenizer chargé au 1er hit
            tok = PreTrainedTokenizerFast.from_pretrained(TOK)
        out_txt.write(("" if first else "\n") + story + f"\n{MARKER}\n")
        first = False
        n_chars += len(story)
        buf_texts.append(story)
        flush_encode(tok)
        kept += 1
        if kept % 50_000 == 0:
            print(f"[{tag}] {kept}/{len(keep)} encodées", flush=True)
    flush_encode(tok, force=True)
    out_txt.close()
    arr = np.concatenate(blocks) if blocks else np.array([], dtype=np.int32)
    return arr, kept, n_chars, seen_idx


def main():
    old = set(random.Random(0).sample(range(2_119_718), 100_000))
    print(f"exclusion des {len(old)} stories du subset original", flush=True)

    os.makedirs(DATA, exist_ok=True)
    stats = {}
    for tag, path, n, excl in (("train", TRAIN_TXT, N_TRAIN, old),
                               ("validation", VALID_TXT, N_VAL, None),
                               ("test", VALID_TXT, N_TEST, None)):
        arr, kept, n_chars, _ = build_split(path, n, tag, excl)
        np.save(os.path.join(DATA, f"{tag}.npy"), arr)
        stats[tag] = {"tokens": int(len(arr)), "chars": int(n_chars),
                      "tokens_per_char": round(len(arr) / max(n_chars, 1), 4)}
        print(f"{tag}: {len(arr):,} tokens ({len(arr)/1e6:.1f}M)", flush=True)
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=2)
    print("DONE big corpus (streaming)", flush=True)


if __name__ == "__main__":
    main()
