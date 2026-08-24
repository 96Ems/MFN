import json
import os
import random
import sys

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
SYSP = os.path.dirname(ROOT)
for p in (SYSP, os.path.join(SYSP, "phase5")):
    if p not in sys.path:
        sys.path.insert(0, p)
from build_bigdata_stream import find_tinystories_txt  # noqa: E402
TRAIN_TXT = find_tinystories_txt("train")
VALID_TXT = find_tinystories_txt("valid")
OUT_DIR = os.path.join(ROOT, "tokenizer_subset")
DATA = os.path.join(ROOT, "data_subset")
VOCAB_SIZE = 4096
N_TRAIN, N_VALID, N_TEST, SEED = 100_000, 2_000, 2_000, 0
MARKER = "<|endoftext|>"


def count_stories(path):
    return sum(1 for line in open(path, encoding="utf-8")
               if line.strip() == MARKER)


def sample_indices(total, n, seed):
    return set(random.Random(seed).sample(range(total), n))


def stream_select(path, keep):
    buf, idx = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == MARKER:
                if idx in keep and buf:
                    yield "\n".join(buf)
                idx += 1
                buf = []
            elif s:
                buf.append(s)
    if idx in keep and buf:
        yield "\n".join(buf)


def build_split(path, n, tag):
    total = count_stories(path)
    keep = sample_indices(total, n, (SEED + 1) if tag == "validation" else (SEED + 2) if tag == "test" else SEED)
    stories = list(stream_select(path, keep))
    text = f"\n{MARKER}\n".join(stories) + f"\n{MARKER}\n"
    open(os.path.join(ROOT, "data_subset_stories", f"{tag}.txt"), "w",
         encoding="utf-8").write(text)
    return text, len(stories)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data_subset_stories"), exist_ok=True)
    print(f"train stories: {count_stories(TRAIN_TXT)}", flush=True)
    train_text, n_train = build_split(TRAIN_TXT, N_TRAIN, "train")
    valid_text, n_valid = build_split(VALID_TXT, N_VALID, "validation")
    test_text, n_test = build_split(VALID_TXT, N_TEST, "test")
    print(f"sampled: {n_train} train / {n_valid} valid / {n_test} test", flush=True)

    tokenizer = Tokenizer(models.BPE(unk_token=MARKER))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                                  special_tokens=[MARKER],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(
        (train_text[i:i+400_000] for i in range(0, len(train_text), 400_000)),
        trainer=trainer)
    tokenizer.save(os.path.join(OUT_DIR, "tokenizer.json"))
    print(f"tokenizer: vocab {tokenizer.get_vocab_size()}", flush=True)

    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token=MARKER,
                                   bos_token=MARKER, eos_token=MARKER, pad_token=MARKER)
    fast.save_pretrained(OUT_DIR)
    stats = {}
    for name, text in (("train", train_text), ("validation", valid_text), ("test", test_text)):
        ids = fast.encode(text)
        np.save(os.path.join(DATA, f"{name}.npy"), np.asarray(ids, dtype=np.int32))
        n_chars = len(text) - text.count(MARKER) * len(MARKER)
        stats[name] = {"tokens": int(len(ids)), "chars": int(n_chars),
                       "tokens_per_char": round(len(ids) / n_chars, 4)}
        print(f"{name}: {len(ids):,} tokens, {len(ids)/n_chars:.3f} tok/char", flush=True)
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
