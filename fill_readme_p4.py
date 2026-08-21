import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, "README.md")
JSON = os.path.join(ROOT, "phase4", "results.json")
START, END = "<!--P4_RESULTS_START-->", "<!--P4_RESULTS_END-->"

MODEL_NAMES = {
    "mfn_dense": "MFN dense v2 (dual-stream + LN)",
    "gru": "GRU + LN (H appariée)",
    "gpt2mini": "GPT-2 mini (transformers)",
}


def main():
    if not os.path.exists(JSON):
        print("phase4/results.json missing"); return
    res = json.load(open(JSON))
    order = ["mfn_dense", "gru", "gpt2mini"]
    rows = []
    for t in order:
        if t not in res:
            continue
        r = res[t]
        rows.append(
            f"| {MODEL_NAMES[t]} | {r['params']:,} | {r['epochs']} | "
            f"{r['best_val_loss']:.4f} | {r['test_ppl']:.2f} | "
            f"{r['test_bpc']:.4f} | {r['history'][-1]['tok_s']:.0f} | "
            f"{r['device']} |"
        )
    table = (
        "| Modèle | Paramètres | Epochs | Val (best) loss | Test ppl | "
        "Test BPC | tok/s | Device |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    )
    md = open(README).read()
    if START not in md or END not in md:
        print("markers missing in README"); return
    head, _, tail = md.partition(START)
    _, _, tail = tail.partition(END)
    open(README, "w").write(head + f"{START}\n{table}\n{END}" + tail)
    print("README phase-4 results updated")


if __name__ == "__main__":
    main()