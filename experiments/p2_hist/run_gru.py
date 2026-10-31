"""E-HIST neural sequence stand-in for sets E and G (H10 architecture contrast is NOT the question here; the
GRU is a diagnostic instrument: does whole-game history carry information the compact state misses?).

Fixed architecture, written before any outcome: per (game, offence) sequence of segments; input at step t =
compact state (start type 4, home 1, margin bucket 7, period 5, clutch 1 = 18 dense) + sum of learned
16-d embeddings of the 5 offence players + sum for the 5 defence players + 8-d one-hot of the PREVIOUS
segment's class (zeros at the first segment) [+ GL latent feature for set G]; GRU hidden 32; softmax over
8 classes. Nested state-only references in the SAME family: A_nn (no previous class, per-step MLP of width 32 instead
of the GRU, no memory) and F_nn (A_nn + GL), so history_signal_nn = A_nn - E_gru and history_residual_nn = F_nn - G_gru
compare like with like. Adam lr 2e-3, batch 64 sequences, 4 epochs, weight decay 1e-5, seed 0. Same prequential
months and training windows as run_hist.py; same per-event bits output (bits_E_gru, bits_G_gru), so the
summary can place the sequence model next to the frozen multinomial logit on identical test events.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.models.event import START, _hist_features, event_classes  # noqa: E402
from run_hist import OUT, TRAIN_DAYS, load_segments  # noqa: E402

torch.set_num_threads(4)
EMB, HID, EPOCHS, LR, BATCH = 16, 32, 4, 2e-3, 64


def dense_state(P):
    st = P["start_type"].map({s: i for i, s in enumerate(START)}).fillna(0).astype(int).to_numpy()
    mb = np.clip(np.round(P["off_margin_start"].to_numpy() / 6.0), -3, 3).astype(int) + 3
    per = np.minimum((P["start_elapsed_s"].to_numpy() // 720).astype(int), 4)
    clutch = (P["start_elapsed_s"].to_numpy() >= 2760) & (np.abs(P["off_margin_start"].to_numpy()) <= 5)
    return np.concatenate([np.eye(4)[st], P["off_is_home"].to_numpy().astype(float)[:, None], np.eye(7)[mb], np.eye(5)[per],
                           clutch.astype(float)[:, None]], 1).astype(np.float32)


class Seqs:
    """Sequences per (game, offence), ordered by possession_seq; player codes from the training vocabulary."""

    def __init__(self, P, vocab, with_gl, with_prev=True, shuffle_order=False, seed=0):
        P = P.sort_values(["game_id", "off_team_id", "possession_seq"], kind="stable")
        if shuffle_order:  # negative control: permute the segment order inside every (game, offence) sequence
            rng = np.random.default_rng(seed)
            P = P.assign(_k=rng.random(len(P))).sort_values(["game_id", "off_team_id", "_k"], kind="stable").drop(columns="_k")
        self.index = P.index.to_numpy()
        X = dense_state(P)
        y = event_classes(P)
        prev = np.zeros((len(P), 8), np.float32)
        H = _hist_features(P)
        gl = (10.0 * H["GL"].to_numpy(float)).astype(np.float32)[:, None]
        codes = lambda col: np.stack([[vocab.get(p, 0) for p in lu] for lu in P[col]]).astype(np.int64)  # noqa: E731
        off, deff = codes("off_lineup"), codes("def_lineup")
        self.seqs = []
        keys = (P["game_id"] + "|" + P["off_team_id"]).to_numpy()
        starts = np.r_[0, np.where(keys[1:] != keys[:-1])[0] + 1, len(P)]
        for a, b in zip(starts[:-1], starts[1:]):
            yy = y[a:b]
            pv = np.zeros((b - a, 8), np.float32)
            pv[1:, :] = np.eye(8, dtype=np.float32)[yy[:-1]]
            xx = np.concatenate([X[a:b]] + ([pv] if with_prev else []) + ([gl[a:b]] if with_gl else []), 1)
            self.seqs.append((torch.tensor(xx), torch.tensor(off[a:b]), torch.tensor(deff[a:b]), torch.tensor(yy), np.arange(a, b)))
        del prev


class GRUNet(nn.Module):
    """recurrent=True: GRU over the sequence (sets E / G). recurrent=False: the same embeddings and state through a
    per-step MLP of the same width, no previous-class input, no memory (sets A_nn / F_nn): the nested state-only
    reference inside the neural family."""

    def __init__(self, n_players, d_in, recurrent=True):
        super().__init__()
        self.recurrent = recurrent
        self.emb = nn.Embedding(n_players + 1, EMB, padding_idx=0)
        if recurrent:
            self.gru = nn.GRU(d_in + 2 * EMB, HID, batch_first=True)
        else:
            self.mlp = nn.Sequential(nn.Linear(d_in + 2 * EMB, HID), nn.ReLU())
        self.out = nn.Linear(HID, 8)

    def forward(self, x, off, deff):
        e = torch.cat([x, self.emb(off).sum(2), self.emb(deff).sum(2)], -1)
        h = self.gru(e)[0] if self.recurrent else self.mlp(e)
        return self.out(h)


def batches(seqs, idx, bs):
    for i in range(0, len(idx), bs):
        chunk = [seqs[j] for j in idx[i:i + bs]]
        L = max(len(c[3]) for c in chunk)
        x = torch.zeros(len(chunk), L, chunk[0][0].shape[1]); off = torch.zeros(len(chunk), L, 5, dtype=torch.long)
        deff = torch.zeros(len(chunk), L, 5, dtype=torch.long); y = torch.full((len(chunk), L), -100, dtype=torch.long)
        pos = []
        for k, c in enumerate(chunk):
            n = len(c[3]); x[k, :n] = c[0]; off[k, :n] = c[1]; deff[k, :n] = c[2]; y[k, :n] = c[3]; pos.append(c[4])
        yield x, off, deff, y, pos


def fit_eval(tr, te, with_gl, seed=0, recurrent=True, with_prev=None, shuffle_order=False):
    torch.manual_seed(seed); np.random.seed(seed)
    with_prev = recurrent if with_prev is None else with_prev
    players = sorted({p for col in ("off_lineup", "def_lineup") for lu in tr[col] for p in lu})
    vocab = {p: i + 1 for i, p in enumerate(players)}
    S_tr = Seqs(tr, vocab, with_gl, with_prev=with_prev, shuffle_order=shuffle_order, seed=seed)
    S_te = Seqs(te, vocab, with_gl, with_prev=with_prev, shuffle_order=shuffle_order, seed=seed + 1)
    d_in = S_tr.seqs[0][0].shape[1]
    net = GRUNet(len(players), d_in, recurrent=recurrent)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        idx = rng.permutation(len(S_tr.seqs))
        for x, off, deff, y, _ in batches(S_tr.seqs, idx, BATCH):
            opt.zero_grad()
            loss = lossf(net(x, off, deff).reshape(-1, 8), y.reshape(-1))
            loss.backward(); opt.step()
    net.eval()
    bits = np.zeros(len(te))
    with torch.no_grad():
        for x, off, deff, y, pos in batches(S_te.seqs, np.arange(len(S_te.seqs)), 256):
            lp = torch.log_softmax(net(x, off, deff), -1)
            for k, p in enumerate(pos):
                n = len(p)
                bits[p] = (-lp[k, :n].gather(1, y[k, :n, None]).squeeze(1) / np.log(2)).numpy()
    # bits are in Seqs order (sorted); map back to te's row order
    order = np.empty(len(te), int); order[np.arange(len(te))] = 0
    out = np.empty(len(te)); out[:] = np.nan
    pos_sorted = pd.Index(te.index).get_indexer(S_te.index)
    out[pos_sorted] = bits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", nargs="+", type=int, required=True)
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--controls", action="store_true", help="also fit E_shuffled (order permuted) and E_noprev (no previous-class input)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    bits_dir = os.path.join(OUT, "bits")
    os.makedirs(bits_dir, exist_ok=True)
    seasons_needed = sorted(set(range(min(a.test_seasons) - 3, max(a.test_seasons) + 1)) & set(range(1997, 2021)))
    P = load_segments(seasons_needed)
    months = sorted(P.loc[P["season"].isin(a.test_seasons), "month"].unique())
    if a.months:
        months = [m for m in months if m in set(a.months)]
    months = [m for i, m in enumerate(months) if i % a.every == a.offset]
    log = open(os.path.join(OUT, f"run_gru_{min(a.test_seasons)}_{max(a.test_seasons)}.log"), "a")
    for m in months:
        f = os.path.join(bits_dir, f"gru_{m}.parquet")
        if os.path.exists(f):
            continue
        te = P[(P["month"] == m) & (P["season"] >= 2000)]
        if not len(te):
            continue
        m_start = te["game_date"].min().normalize()
        tr = P[(P["game_date"] < m_start) & (P["game_date"] >= m_start - pd.Timedelta(days=TRAIN_DAYS))]
        if len(tr) < 20000:
            continue
        t0 = time.time()
        bA = fit_eval(tr, te, with_gl=False, recurrent=False)
        bF = fit_eval(tr, te, with_gl=True, recurrent=False)
        bE = fit_eval(tr, te, with_gl=False)
        bG = fit_eval(tr, te, with_gl=True)
        out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(), "season": te["season"].to_numpy(),
                            "month": m, "y": event_classes(te), "bits_A_nn": bA, "bits_F_nn": bF, "bits_E_gru": bE, "bits_G_gru": bG})
        if a.controls:
            out["bits_E_gru_shuffled"] = fit_eval(tr, te, with_gl=False, shuffle_order=True)
            out["bits_E_gru_noprev"] = fit_eval(tr, te, with_gl=False, with_prev=False)
        out.to_parquet(f, index=False)
        json.dump({"month": m, "n_train": int(len(tr)), "n_test": int(len(te)), "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(bits_dir, f"gru_{m}.json"), "w"))
        msg = f"{m} gru n_train={len(tr)} n_test={len(te)} A_nn={np.nanmean(bA):.4f} F_nn={np.nanmean(bF):.4f} E={np.nanmean(bE):.4f} G={np.nanmean(bG):.4f}" + (f" E_shuffled={out['bits_E_gru_shuffled'].mean():.4f} E_noprev={out['bits_E_gru_noprev'].mean():.4f}" if a.controls else "") + f" {time.time()-t0:.0f}s"
        print(msg, file=log, flush=True); print(msg, flush=True)


if __name__ == "__main__":
    main()
