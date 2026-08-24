import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "ultrachat", "train_sft-00.parquet")
TOK = os.path.join(ROOT, "phase4", "tokenizer_subset")
OUT = os.path.join(ROOT, "phase5", "data_sft")
N_TRAIN, N_VAL, SEED = 14700, 300, 0
USER, ASST, EOS = "User: ", "\nAssistant: ", "<|endoftext|>"


def ensure_parquet():
    """Télécharge la tranche UltraChat SFT (~250 Mo) si absente — portable
    laptop/Mac (le .parquet n'est pas dans git)."""
    if os.path.exists(SRC):
        return SRC
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    print("parquet UltraChat absent -> téléchargement ~250 Mo "
          "(une seule fois)...", flush=True)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="stingning/ultrachat",
                        filename="data/train_sft-00.parquet",
                        repo_type="dataset",
                        local_dir=os.path.dirname(SRC))
    os.replace(p, SRC)
    return SRC


def fmt_dialogue(msgs, tok, masks_out):
    ids, masks = [], []
    for m in msgs:
        role, content = m["role"], m["content"]
        if role == "user":
            ids += tok.encode(USER + content + ASST)
            masks += [0] * (len(USER + content + ASST))
        else:
            a_ids = tok.encode(content)
            ids += a_ids + tok.encode(EOS) + tok.encode("\n")
            masks += [1] * (len(a_ids) + 1 + 1)
    return ids, masks


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK)
    src = ensure_parquet()
    t = pq.read_table(src)
    dials = t.column("messages").to_pylist()
    print(f"total dialogues: {len(dials)}", flush=True)

    rng = random.Random(SEED)
    idx = rng.sample(range(len(dials)), N_TRAIN + N_VAL)
    split = {"train": idx[:N_TRAIN], "validation": idx[N_TRAIN:]}

    stats = {}
    for name, idxs in split.items():
        ids_all, mask_all = [], []
        for i in idxs:
            ids, masks = fmt_dialogue(dials[i], tok, [])
            ids_all += ids
            mask_all += masks
        arr = np.asarray(ids_all, dtype=np.int32)
        msk = np.asarray(mask_all, dtype=np.int8)
        np.save(os.path.join(OUT, f"{name}.npy"), arr)
        np.save(os.path.join(OUT, f"{name}_mask.npy"), msk)
        stats[name] = {
            "tokens": int(arr.shape[0]),
            "loss_tokens": int(msk.sum()),
            "dialogues": len(idxs),
            "mask_frac": round(float(msk.mean()), 4),
        }
        print(f"{name}: {arr.shape[0]:,} tokens ({msk.sum():,} loss), "
              f"{len(idxs)} dialogues", flush=True)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()