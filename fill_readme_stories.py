import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, "README.md")
JSON = os.path.join(ROOT, "stories_results.json")

START, END = "<!--STORIES_RESULTS_START-->", "<!--STORIES_RESULTS_END-->"

MODEL_NAMES = {
    "mfn_dense": "MFN dense dual-stream v2 (H=144)",
    "mfn_dense_nobidir": "MFN dense no-bidir (ablation)",
    "mfn_dense_nofatigue": "MFN dense no-fatigue (ablation)",
    "gru": "GRU + LN (H=215, appariée)",
    "gpt2mini": "GPT-2 mini (transformers, 4 couches, d=120)",
}


def main():
    if not os.path.exists(JSON):
        print("stories_results.json missing"); return
    res = json.load(open(JSON))
    order = ["mfn_dense", "mfn_dense_nobidir", "gru", "gpt2mini",
             "mfn_dense_nofatigue"]
    order = [t for t in order if t in res]
    rows = []
    for tag in order:
        r = res[tag]
        rows.append(
            f"| {MODEL_NAMES[tag]} | {r['hidden']} | {r['params']:,} | "
            f"{r['best_val_loss']:.4f} / {r['test_loss']:.4f} | "
            f"{r['test_ppl']:.1f} | {r['test_bpc']:.4f} | "
            f"{r['history'][-1]['tok_s']:.0f} | {r['epochs']} |"
        )
    table = (
        "| Modèle | H | Paramètres | Val (best) / Test (loss) | Test ppl | "
        "Test BPC | tok/s | Epochs |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    )
    md = open(README).read()
    if START not in md or END not in md:
        print("markers missing in README"); return
    head, _, tail = md.partition(START)
    _, _, tail = tail.partition(END)
    block = f"{START}\n{table}\n{END}"
    open(README, "w").write(head + block + tail)
    print("README stories results updated")


if __name__ == "__main__":
    main()