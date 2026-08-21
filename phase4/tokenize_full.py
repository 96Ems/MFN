import json
import os
import random

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = "/home/emericclement/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TRAIN_TXT = os.path.join(CACHE, "TinyStories-train.txt")
VALID_TXT = os.path.join(CACHE, "TinyStories-valid.txt")
OUT_DIR = os.path.join(ROOT, "tokenizer_full")
DATA = os.path.join(ROOT, "data")
VOCAB_SIZE = 4096
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


def encode_stream(path, enc, chunk_lines=CHUNK_LINES):
    """Encode a big text file without loading it whole; return np int32 array."""
    parts = []
    total = 0
    for chunk in line_chunks(path, chunk_lines):
        ids = enc.encode(chunk)
        total += len(ids)
        parts.append(np.asarray(ids, dtype=np.int32))
    arr = np.concatenate(parts)
    assert arr.shape[0] == total
    return arr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    tokenizer = Tokenizer(models.BPE(unk_token=MARKER))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[MARKER],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    print("BPE training on full train (streamed)...", flush=True)
    tokenizer.train_from_iterator(line_chunks(TRAIN_TXT, CHUNK_LINES),
                                  trainer=trainer)
    tokenizer.save(tok_path)
    print(f"tokenizer trained: vocab {tokenizer.get_vocab_size()}", flush=True)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=MARKER, bos_token=MARKER, eos_token=MARKER, pad_token=MARKER)
    fast.save_pretrained(OUT_DIR)
    enc = fast

    print("encoding full train (streamed)...", flush=True)
    train_arr = encode_stream(TRAIN_TXT, enc)
    np.save(os.path.join(DATA, "train.npy"), train_arr)
    print(f"train: {len(train_arr):,} tokens", flush=True)

    # valid/test: valid.txt is small enough to split in memory
    valid_text = open(VALID_TXT, encoding="utf-8").read()
    stories = [p for p in valid_text.split(MARKER) if p.strip()]
    print(f"valid stories: {len(stories)}", flush=True)
    idx = list(range(len(stories)))
    rng = random.Random(SEED)
    test_idx = set(rng.sample(idx, min(N_TEST, len(idx))))
    valid_stories = [stories[i] for i in idx if i not in test_idx]
    test_stories = [stories[i] for i in test_idx]

    splits = {
        "validation": "".join(s + f"\n{MARKER}\n" for s in valid_stories),
        "test": "".join(s + f"\n{MARKER}\n" for s in test_stories),
    }
    stats = {"train": {"tokens": int(len(train_arr)), "stories": None}}
    for name, text in splits.items():
        ids = enc.encode(text)
        arr = np.asarray(ids, dtype=np.int32)
        np.save(os.path.join(DATA, f"{name}.npy"), arr)
        n_markers = text.count(MARKER)
        n_chars = len(text) - n_markers * len(MARKER)
        stats[name] = {
            "tokens": int(len(arr)), "chars": int(n_chars),
            "tokens_per_char": round(len(arr) / n_chars, 4),
        }
        print(f"{name}: {len(arr):,} tokens, "
              f"{len(arr) / n_chars:.3f} tok/char", flush=True)

    with open(os.path.join(DATA, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()