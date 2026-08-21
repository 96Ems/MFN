import json
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(ROOT, "data", "wt2_parquet")
OUT_DIR = os.path.join(ROOT, "tokenizer_wt2")
ENCODED = os.path.join(ROOT, "data", "wt2_encoded")
VOCAB_SIZE = 4096


def load_text(split):
    path = os.path.join(PARQUET, f"{split}.parquet")
    table = pq.read_table(path)
    lines = [str(t) for t in table.column("text").to_pylist() if str(t).strip()]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ENCODED, exist_ok=True)

    train_text = load_text("train")

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
        print(f"tokenizer loaded: vocab {tokenizer.get_vocab_size()}")
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(
            (train_text[i : i + 200000] for i in range(0, len(train_text), 200000)),
            trainer=trainer,
        )
        tokenizer.save(tok_path)
        print(f"tokenizer trained: vocab {tokenizer.get_vocab_size()}")

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    fast.save_pretrained(OUT_DIR)

    stats = {}
    for split in ("train", "validation", "test"):
        text = load_text(split)
        ids = fast.encode(text)
        arr = np.asarray(ids, dtype=np.int32)
        np.save(os.path.join(ENCODED, f"{split}.npy"), arr)
        stats[split] = {
            "tokens": int(len(arr)),
            "chars": int(len(text)),
            "tokens_per_char": round(len(arr) / len(text), 4),
            "bytes": int(os.path.getsize(os.path.join(ENCODED, f"{split}.npy"))),
        }
        print(f"{split}: {len(arr)} tokens, {len(text)} chars, "
              f"{len(arr) / len(text):.3f} tok/char")
    with open(os.path.join(ENCODED, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
