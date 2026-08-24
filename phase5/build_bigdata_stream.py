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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "phase4", "tokenizer_subset")
DATA = os.path.join(ROOT, "phase4", "data_big")
STOR = os.path.join(ROOT, "phase4", "data_big_stories")
MARKER = "<|endoftext|>"
N_TRAIN, N_VAL, N_TEST = 400_000, 2_000, 2_000
FULL = os.environ.get("MFN_FULL", "0") == "1"


def hf_cache_dir(dataset):
    """~/.cache/huggingface/hub/datasets--<org>--<name>  (portable M1/laptop)."""
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                        "hub", "datasets--" + dataset.replace("/", "--"))


def find_tinystories_txt(kind):
    """Localise TinyStories-{train,valid}.txt dans le cache HF, sinon
    télécharge le corpus (~2 Go la 1re fois). Même chemin sur laptop et Mac."""
    name = f"TinyStories-{kind}.txt"
    base = hf_cache_dir("roneneldan/TinyStories")
    snaps = os.path.join(base, "snapshots")
    if os.path.isdir(snaps):
        for snap in os.listdir(snaps):
            p = os.path.join(snaps, snap, name)
            if os.path.exists(p):
                print(f"TinyStories {kind} trouvé: {p}", flush=True)
                return p
    print(f"TinyStories introuvable dans le cache HF -> téléchargement "
          f"({name}, ~1 Go), une seule fois...", flush=True)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="roneneldan/TinyStories", filename=name, repo_type="dataset")


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


def build_split(path, n, tag, exclude=None, total_hint=2_119_718):
    os.makedirs(STOR, exist_ok=True)
    hints = {"train": 2_119_718, "validation": 21_989, "test": 21_989}
    total_hint = hints[tag] if total_hint is None else total_hint
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
    global DATA
    TRAIN_TXT = find_tinystories_txt("train")
    VALID_TXT = find_tinystories_txt("valid")
    if FULL:
        DATA = os.path.join(ROOT, "phase4", "data_big2")
        print("== MODE FULL: corpus TinyStories complet (~530M tokens) -> "
              "data_big2 (cible z30/z300) ==", flush=True)
    old = set(random.Random(0).sample(range(2_119_718), 100_000))
    print(f"exclusion des {len(old)} stories du subset original", flush=True)

    os.makedirs(DATA, exist_ok=True)
    stats = {}
    for tag, path, n, excl in (
            ("train", TRAIN_TXT,
             2_119_718 if FULL else N_TRAIN, None if FULL else old),
            ("validation", VALID_TXT, 21_989 if FULL else N_VAL, None),
            ("test", VALID_TXT, 21_989 if FULL else N_TEST, None)):
        arr, kept, n_chars, _ = build_split(path, n, tag, excl)
        np.save(os.path.join(DATA, f"{tag}.npy"), arr)
        stats[tag] = {"tokens": int(len(arr)), "chars": int(n_chars),
                      "tokens_per_char": round(len(arr) / max(n_chars, 1), 4)}
        print(f"{tag}: {len(arr):,} tokens ({len(arr)/1e6:.1f}M)", flush=True)
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=2)
    print("DONE big corpus (streaming)", flush=True)


if __name__ == "__main__":
    main()
