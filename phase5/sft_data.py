"""Data SFT UltraChat -> phase5/data_sft/ (User:/Assistant:, masque CE côté assistant).

NOTE M1 (24/08/26) : l'ancien parquet `stingning/ultrachat data/train_sft-00.parquet`
n'existe plus (repo déplacé vers openbmb/UltraChat, réorganisé en shards jsonl).
On télécharge désormais le shard `train_0.jsonl` (~1 Go, même ordre de grandeur que
la tranche parquet d'origine) et on convertit le format plat alterné
{"id": ..., "data": [q1, a1, q2, a2, ...]} en messages [{"role"},{"content"},...].
Le pipeline aval (formatage User:/Assistant:, masks, sampling seed 0) est inchangé.
"""
import json
import os
import random

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "ultrachat", "train_0.jsonl")
TOK = os.path.join(ROOT, "phase4", "tokenizer_subset")
OUT = os.path.join(ROOT, "phase5", "data_sft")
N_TRAIN, N_VAL, SEED = 14700, 300, 0
USER, ASST, EOS = "User: ", "\nAssistant: ", "<|endoftext|>"


def ensure_jsonl():
    """Télécharge le shard UltraChat si absent — portable laptop/Mac."""
    if os.path.exists(SRC):
        return SRC
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    print("shard UltraChat absent -> téléchargement train_0.jsonl (~1 Go, "
          "une seule fois)...", flush=True)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="openbmb/UltraChat",
                        filename="train_0.jsonl",
                        repo_type="dataset",
                        local_dir=os.path.dirname(SRC))
    os.replace(p, SRC)
    return SRC


def iter_dialogues(path):
    """Yield les dialogues convertis en messages [{"role","content"},...] —
    lecture au fil du fichier (mémoire constante)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)["data"]
            msgs = []
            for i in range(0, len(data) - 1, 2):
                msgs.append({"role": "user", "content": str(data[i])})
                msgs.append({"role": "assistant", "content": str(data[i + 1])})
            if msgs:
                yield msgs


def fmt_dialogue(msgs, tok):
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
    src = ensure_jsonl()

    # passe 1 : compter (1 ligne jsonl = 1 dialogue)
    n_dials = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_dials += 1
    print(f"total dialogues: {n_dials}", flush=True)

    rng = random.Random(SEED)
    idx = rng.sample(range(n_dials), N_TRAIN + N_VAL)
    wanted = {}
    for k, i in enumerate(idx):
        wanted[i] = "train" if k < N_TRAIN else "validation"

    acc = {"train": ([], []), "validation": ([], [])}
    for i, msgs in enumerate(iter_dialogues(src)):
        if i not in wanted:
            continue
        ids_all, mask_all = acc[wanted[i]]
        ids, masks = fmt_dialogue(msgs, tok)
        ids_all += ids
        mask_all += masks

    stats = {}
    for name in ("train", "validation"):
        ids_all, mask_all = acc[name]
        arr = np.asarray(ids_all, dtype=np.int32)
        msk = np.asarray(mask_all, dtype=np.int8)
        np.save(os.path.join(OUT, f"{name}.npy"), arr)
        np.save(os.path.join(OUT, f"{name}_mask.npy"), msk)
        stats[name] = {
            "tokens": int(arr.shape[0]),
            "loss_tokens": int(msk.sum()),
            "dialogues": sum(1 for v in wanted.values() if v == name),
            "mask_frac": round(float(msk.mean()), 4),
        }
        print(f"{name}: {arr.shape[0]:,} tokens ({msk.sum():,} loss), "
              f"{stats[name]['dialogues']} dialogues", flush=True)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
