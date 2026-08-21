import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, "README.md")
JSON = os.path.join(ROOT, "lm_results.json")

START, END = "<!--LM_RESULTS_START-->", "<!--LM_RESULTS_END-->"

MODEL_NAMES = {
    "mfndiag": "MFN diagonal O(H) (complet)",
    "mfndiag_nobidir": "MFN diagonal no-bidir (ablation)",
    "gru": "GRU (H=199, appariée)",
    "dense_mfn": "MFN dense O(H²) (référence)",
}


def main():
    if not os.path.exists(JSON):
        print("lm_results.json missing"); return
    res = json.load(open(JSON))
    rows = []
    order = ["mfndiag", "mfndiag_nobidir", "gru", "dense_mfn"]
    order = [t for t in order if t in res]
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
    print("README LM results updated")


if __name__ == "__main__":
    main()