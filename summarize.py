import json
import os
import sys


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_mean_std(m, s, nd=4):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def task1_table(data, names):
    rows = []
    for name in names:
        if name not in data:
            rows.append(f"| {name} | — | — | — |")
            continue
        r = data[name]
        if not all('mean' in r[f'k{k}'] for k in (5, 10, 20)):
            cells = " | ".join(
                f"{len(r[f'k{k}'].get('runs', []))}/5 seeds" if f'k{k}' in r
                else "—" for k in (5, 10, 20))
            rows.append(f"| {name} | {cells} |")
            continue
        cells = " | ".join(
            fmt_mean_std(r[f'k{k}']['mean'], r[f'k{k}']['std'])
            for k in (5, 10, 20))
        rows.append(f"| {name} ({int(r['params'] / 1000)}K, "
                    f"{len(r['seeds'])} seeds) | {cells} |")
    return "\n".join(rows)


def dualctx_table(data, names, cols=('mse_a', 'mse_b', 'total')):
    rows = []
    for name in names:
        if name not in data:
            rows.append(f"| {name} | — | — | — |")
            continue
        r = data[name]
        cells = " | ".join(
            fmt_mean_std(r[c]['mean'], r[c]['std']) for c in cols)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def main():
    t1 = load('results.json')
    b1 = load('baselines_t1.json')
    dc = load('dualctx.json')
    ptb = load('ptb.json')
    fat = load('fatigue.json')
    if t1:
        print("== TASK 1: multi-delay copy (MFN family) ==")
        print(task1_table(t1, ['mfn', 'mfnnofatigue', 'mfnnobidir',
                               'mfnnogates', 'mfnshareddecay']))
    if b1:
        print("\n== TASK 1: baselines ==")
        print(task1_table(b1, ['gru', 'lstm', 'mamba', 'xlstm']))
    if dc:
        print("\n== TASK 3: dual-context ==")
        print(dualctx_table(dc, ['mfn', 'mfnnobidir', 'gru', 'lstm']))
    if ptb:
        print("\n== TASK 2: PTB char BPC ==")
        for name, r in ptb.items():
            print(f"| {name} | best valid {r['best_valid_bpc']:.4f} | "
                  f"test {r['test_bpc']:.4f} | {r['params']} params |")
    if fat:
        print("\n== FATIGUE PROFILE ==")
        for k, v in fat.items():
            print(f"{k}: {v}")

    missing = [p for p, d in
               (('results.json', t1), ('baselines_t1.json', b1),
                ('dualctx.json', dc), ('ptb.json', ptb),
                ('fatigue.json', fat)) if d is None]
    if missing:
        print(f"\n[pending: {', '.join(missing)}]", file=sys.stderr)


if __name__ == '__main__':
    main()