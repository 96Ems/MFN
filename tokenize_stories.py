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
OUT_DIR = os.path.join(ROOT, "tokenizer_stories")
ENCODED = os.path.join(ROOT, "data", "stories_encoded")
STORIES = os.path.join(ROOT, "data", "stories")
VOCAB_SIZE = 4096
N_TRAIN = 40000
N_VALID = 2000
N_TEST = 2000
SEED = 0
MARKER = "<|endoftext|>"


def count_stories(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == MARKER:
                n += 1
    return n


def sample_indices(total, n, seed):
    rng = random.Random(seed)
    return set(rng.sample(range(total), n))


def stream_select(path, keep):
    """Yield stories (joined paragraphs) whose index is in `keep`."""
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


def build_split(path, n, tag, seed):
    total = count_stories(path)
    keep = sample_indices(total, n, seed)
    stories = list(stream_select(path, keep))
    text = f"\n{MARKER}\n".join(stories) + f"\n{MARKER}\n"
    os.makedirs(STORIES, exist_ok=True)
    with open(os.path.join(STORIES, f"{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text, len(stories)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ENCODED, exist_ok=True)

    print(f"train stories: {count_stories(TRAIN_TXT)}", flush=True)
    print(f"valid stories: {count_stories(VALID_TXT)}", flush=True)

    train_text, n_train = build_split(TRAIN_TXT, N_TRAIN, "train", SEED)
    valid_text, n_valid = build_split(VALID_TXT, N_VALID, "validation", SEED + 1)
    test_text, n_test = build_split(VALID_TXT, N_TEST, "test", SEED + 2)
    print(f"sampled: train {n_train} valid {n_valid} test {n_test}", flush=True)

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    tokenizer = Tokenizer(models.BPE(unk_token=MARKER))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[MARKER],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        (train_text[i : i + 300000] for i in range(0, len(train_text), 300000)),
        trainer=trainer,
    )
    tokenizer.save(tok_path)
    print(f"tokenizer trained: vocab {tokenizer.get_vocab_size()}", flush=True)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=MARKER, bos_token=MARKER, eos_token=MARKER, pad_token=MARKER,
    )
    fast.save_pretrained(OUT_DIR)

    stats = {}
    for split, text in (("train", train_text), ("validation", valid_text),
                        ("test", test_text)):
        ids = fast.encode(text)
        arr = np.asarray(ids, dtype=np.int32)
        np.save(os.path.join(ENCODED, f"{split}.npy"), arr)
        # strip the trailing marker for the chars/tokens ratio
        n_chars = len(text.replace(MARKER, "").replace("\n", ""))
        stats[split] = {
            "tokens": int(len(arr)),
            "chars": int(n_chars),
            "stories": {"train": n_train, "validation": n_valid, "test": n_test}[split],
            "tokens_per_char": round(len(arr) / n_chars, 4),
        }
        print(f"{split}: {len(arr)} tokens, {n_chars} chars, "
              f"{len(arr) / n_chars:.3f} tok/char, {stats[split]['stories']} stories",
              flush=True)
    with open(os.path.join(ENCODED, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()