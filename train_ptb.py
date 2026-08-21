import argparse
import json
import os
import time
import urllib.request

import torch
import torch.nn as nn

from mfn import MyelinFatigueNet

torch.set_num_threads(int(torch.get_num_threads()))

URLS = {
    'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/'
             'data/ptb.train.txt',
    'valid': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/'
             'data/ptb.valid.txt',
    'test': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/'
            'data/ptb.test.txt',
}


def get_data(root='data/ptb'):
    os.makedirs(root, exist_ok=True)
    files = {}
    for split, url in URLS.items():
        path = os.path.join(root, f'ptb.{split}.txt')
        if not os.path.exists(path):
            print(f'downloading {split}...', flush=True)
            urllib.request.urlretrieve(url, path)
        files[split] = path
    return files


def load_chars(paths):
    data = {}
    for split, path in paths.items():
        with open(path) as f:
            text = f.read()
        data[split] = list(text)
    vocab = sorted(set(sum((data[s] for s in data), start=[])))
    stoi = {c: i for i, c in enumerate(vocab)}
    return data, stoi


def batchify(seq, batch_size):
    n = (len(seq) // batch_size) * batch_size
    return seq[:n].view(batch_size, -1).t().contiguous()


class GRUPtb(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.emb = nn.Linear(V, H)
        self.cell = nn.GRUCell(H, H)
        self.readout = nn.Linear(H, V)
        self.H = H

    def step(self, x, h):
        h = self.cell(self.emb(x), h)
        return self.readout(h), h


def one_hot(x, V):
    return torch.nn.functional.one_hot(x, V).float()


def make_runner(model, V):
    def one_hot_step(xt, st, fn):
        return fn(one_hot(xt, V), st)

    if isinstance(model, MyelinFatigueNet):
        def init(B):
            return model.init_state(B, model.H)

        def step(xt, st):
            hl, hp, pl, pp = st
            hl, hp, pl, pp, y = model(one_hot(xt, V), hl, hp, pl, pp)
            return y, (hl, hp, pl, pp)
        return init, step

    def init(B):
        return torch.zeros(B, model.H)

    def step(xt, st):
        return model.step(one_hot(xt, V), st)
    return init, step


def train_model(model, train_seq, valid_seq, test_seq, V, n_epochs=5,
                bptt=256, batch=64, lr=1e-3, seed=0, tag='model'):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    cel = nn.CrossEntropyLoss()
    ntok = train_seq.shape[0]
    n_steps_per_epoch = (ntok - 1) // bptt
    init, step = make_runner(model, V)

    def evaluate(data):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            st = init(data.shape[1])
            for i in range(0, data.shape[0] - 1, bptt):
                x = data[i:i + bptt]
                y = data[i + 1:i + 1 + bptt]
                if x.shape[0] != bptt:
                    break
                logits_seq = []
                for xt in x:
                    o, st = step(xt, st)
                    logits_seq.append(o)
                logits = torch.stack(logits_seq)
                total += cel(logits.view(-1, logits.shape[-1]),
                             y.reshape(-1)).item() * x.numel()
                count += x.numel()
            model.train()
        return total / count

    best_bpc = float('inf')
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        total, count = 0.0, 0
        st = init(batch)
        for step_i in range(n_steps_per_epoch):
            i = step_i * bptt
            opt.zero_grad()
            x = train_seq[i:i + bptt]
            y = train_seq[i + 1:i + 1 + bptt]
            logits_seq = []
            for xt in x:
                o, st = step(xt, st)
                logits_seq.append(o)
            logits = torch.stack(logits_seq)
            loss = cel(logits.view(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if isinstance(st, torch.Tensor):
                st = st.detach()
            else:
                st = tuple(s.detach() for s in st)
            total += loss.item() * x.numel()
            count += x.numel()
        sched.step()
        train_bpc = total / count
        valid_bpc = evaluate(valid_seq)
        best_bpc = min(best_bpc, valid_bpc)
        print(f"[{tag}] epoch {epoch+1}/{n_epochs} train BPC={train_bpc:.4f} "
              f"valid BPC={valid_bpc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    test_bpc = evaluate(test_seq)
    print(f"[{tag}] best valid BPC={best_bpc:.4f} test BPC={test_bpc:.4f}",
          flush=True)
    return {'best_valid_bpc': best_bpc, 'test_bpc': test_bpc,
            'params': sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['mfn', 'gru'])
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--bptt', type=int, default=256)
    ap.add_argument('--out', type=str, default='ptb.json')
    ap.add_argument('--gru_h', type=int, default=170)
    ap.add_argument('--tag', type=str, default=None,
                    help='override result key (for variant runs)')
    args = ap.parse_args()

    files = get_data()
    data, stoi = load_chars(files)
    seqs = {}
    for split, chars in data.items():
        idx = torch.tensor([stoi[c] for c in chars], dtype=torch.long)
        seqs[split] = batchify(idx, args.batch)
    V = len(stoi)
    print(f'vocab size: {V}   train chars: {len(data["train"])}', flush=True)

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    for kind in args.models:
        if kind == 'mfn':
            m = MyelinFatigueNet(V, 64, V)
        elif kind == 'gru':
            m = GRUPtb(V, args.gru_h)
        else:
            raise ValueError(kind)
        r = train_model(m, seqs['train'], seqs['valid'], seqs['test'], V,
                        n_epochs=args.epochs, bptt=args.bptt, batch=args.batch,
                        tag=kind)
        results[args.tag or kind] = r
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()