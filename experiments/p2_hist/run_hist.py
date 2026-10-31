"""E-HIST driver (H3a / H3b): prequential-by-month nested feature sets on NBA nba_stats segments.

Frozen pieces used unchanged: hoopslab.models.event.EventDesign (blocks S, L1, L3, L10, EW, GL, NOISE),
MultinomialLogit (L2 = 1e-4), hoopslab.experiments.history.SETS, hoopslab.metrics.information.bits_per_event.
Protocol (docs/prereg/07: event-level models refit monthly, prequential; docs/prereg/01 E-HIST):
  * test unit = every segment of a development-season month (regular season + playoffs; preseason and
    all-star games excluded, as in P1);
  * training set for month m = every segment of games played in the 730 days before the first day of m
    (burn-in seasons 1997-1999 may serve as training history; nothing after 2020 exists here). The 2-year
    window is the compute compromise fixed before any outcome was seen;
  * one design vocabulary per month, from the training set;
  * per-event bits = -log2 P(actual class) on the test month, stored per segment; contrasts and the
    game-cluster bootstrap are computed afterwards over the pooled months (summarize_hist.py).
Model families: mnl (frozen multinomial logit) and gbm (LightGBM multiclass replication with fixed
parameters written here before any outcome: 31 leaves, 300 rounds, learning rate 0.05, min 50 per leaf,
L2 1.0, feature fraction 0.8, no early stopping). MNL optimiser: L-BFGS up to 1000 iterations (convergence flag stored per set).
Controls per month: NEG = sets E and G with the history/latent blocks permuted across games within
state cells (start type x margin bucket x period), in train and test; CANARY = set A plus the NEXT
segment's class (never a model input anywhere else).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.experiments.history import SETS  # noqa: E402
from hoopslab.metrics.information import bits_per_event  # noqa: E402
from hoopslab.models.event import EventDesign, MultinomialLogit, event_classes  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_hist")
PHASES = {"regular-season", "playoffs"}
TRAIN_DAYS = 730
L2 = 1e-4
MAXITER = 1000  # L-BFGS iterations; the frozen class defaults to 300, at which sets D/E/G did not converge on the timing month (1999-12)
GBM_PARAMS = {"objective": "multiclass", "num_class": 8, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50,
              "lambda_l2": 1.0, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 4,
              "seed": 0}
GBM_ROUNDS = 300
HIST_BLOCKS = ["L1", "L3", "L10", "EW", "GL"]


def load_segments(seasons):
    fr = []
    for S in seasons:
        f = os.path.join(OUT, f"segments_{S}.parquet")
        if os.path.exists(f):
            fr.append(pd.read_parquet(f))
    P = pd.concat(fr, ignore_index=True)
    P = P[P["season_type"].isin(PHASES)].copy()
    P["off_lineup"] = P["off_lineup"].map(tuple)
    P["def_lineup"] = P["def_lineup"].map(tuple)
    P["game_date"] = pd.to_datetime(P["game_date"])
    return P.sort_values(["game_date", "game_id", "possession_seq"], kind="stable").reset_index(drop=True)


def state_cells(P):
    mb = np.clip(np.round(P["off_margin_start"].to_numpy() / 6.0), -3, 3).astype(int)
    per = np.minimum((P["start_elapsed_s"].to_numpy() // 720).astype(int), 4)
    return pd.Series(P["start_type"].astype(str) + "|" + mb.astype(str) + "|" + per.astype(str), index=P.index)


def permute_within_cells(blocks, P, rng):
    """Permute the rows of the history/latent blocks across games within state cells (negative control)."""
    cells = state_cells(P).to_numpy()
    perm = np.arange(len(P))
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        perm[idx] = rng.permutation(idx)
    out = dict(blocks)
    for b in HIST_BLOCKS:
        out[b] = blocks[b][perm]
    return out


def canary_block(P):
    """One-hot of the NEXT segment's class for the same offence in the same game (positive control only)."""
    y = event_classes(P)
    df = pd.DataFrame({"g": P["game_id"].to_numpy(), "t": P["off_team_id"].to_numpy(), "s": P["possession_seq"].to_numpy(), "y": y},
                      index=P.index)
    df = df.sort_values(["g", "t", "s"], kind="stable")
    nxt = df.groupby(["g", "t"], sort=False)["y"].shift(-1).reindex(P.index).to_numpy()
    has = ~np.isnan(nxt)
    M = np.zeros((len(P), 9))
    M[has, nxt[has].astype(int)] = 1.0
    M[~has, 8] = 1.0
    return sp.csr_matrix(M)


def fit_predict(family, Xtr, ytr, Xte):
    if family == "mnl":
        m = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(Xtr, ytr)
        return m.predict_proba(Xte), bool(m.converged)
    import lightgbm as lgb
    ds = lgb.Dataset(sp.csr_matrix(Xtr, dtype=np.float32), label=ytr, free_raw_data=False)
    b = lgb.train(GBM_PARAMS, ds, num_boost_round=GBM_ROUNDS)
    return b.predict(sp.csr_matrix(Xte, dtype=np.float32)), True


def run_month(P_all, month, family, rng, with_controls=True):
    te = P_all[P_all["month"] == month]
    te = te[te["season"] >= 2000]
    if not len(te):
        return None
    m_start = te["game_date"].min().normalize()
    tr = P_all[(P_all["game_date"] < m_start) & (P_all["game_date"] >= m_start - pd.Timedelta(days=TRAIN_DAYS))]
    if len(tr) < 20000:
        return None
    t0 = time.time()
    design = EventDesign(tr)
    ytr, yte = event_classes(tr), event_classes(te)
    btr, bte = design.all_blocks(tr), design.all_blocks(te)
    out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(),
                        "season": te["season"].to_numpy(), "month": month, "y": yte})
    conv = {}
    for name, blocks in SETS.items():
        p, c = fit_predict(family, design.select(btr, blocks), ytr, design.select(bte, blocks))
        out[f"bits_{name}"] = bits_per_event(p, yte)
        conv[name] = c
    if with_controls:
        ptr, pte = permute_within_cells(btr, tr, rng), permute_within_cells(bte, te, rng)
        for name in ("E", "G"):
            p, _ = fit_predict(family, design.select(ptr, SETS[name]), ytr, design.select(pte, SETS[name]))
            out[f"bits_{name}_perm"] = bits_per_event(p, yte)
        ctr, cte = canary_block(tr), canary_block(te)
        XA_tr, XA_te = design.select(btr, SETS["A"]), design.select(bte, SETS["A"])
        p, _ = fit_predict(family, sp.hstack([XA_tr, ctr], format="csr"), ytr, sp.hstack([XA_te, cte], format="csr"))
        out["bits_CANARY"] = bits_per_event(p, yte)
    meta = {"month": month, "family": family, "n_train": int(len(tr)), "n_test": int(len(te)), "train_from": str(tr["game_date"].min().date()),
            "train_to": str(tr["game_date"].max().date()), "n_players_vocab": len(design.players), "n_teams": len(design.teams),
            "converged": conv, "seconds": round(time.time() - t0, 1)}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mnl", "gbm"], default="mnl")
    ap.add_argument("--test-seasons", nargs="+", type=int, required=True, help="development seasons whose months are tested")
    ap.add_argument("--months", nargs="*", default=None, help="restrict to these YYYY-MM test months")
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    bits_dir = os.path.join(OUT, "bits")
    os.makedirs(bits_dir, exist_ok=True)
    seasons_needed = sorted(set(range(min(a.test_seasons) - 3, max(a.test_seasons) + 1)) & set(range(1997, 2021)))
    P = load_segments(seasons_needed)
    months = sorted(P.loc[P["season"].isin(a.test_seasons), "month"].unique())
    if a.months:
        months = [m for m in months if m in set(a.months)]
    rng = np.random.default_rng(12345)
    log = open(os.path.join(OUT, f"run_{a.family}_{min(a.test_seasons)}_{max(a.test_seasons)}.log"), "a")
    for m in months:
        f = os.path.join(bits_dir, f"{a.family}_{m}.parquet")
        if os.path.exists(f):
            continue
        r = run_month(P, m, a.family, rng, with_controls=not a.no_controls)
        if r is None:
            print(m, "skipped (no test rows or too little training history)", file=log, flush=True)
            continue
        out, meta = r
        out.to_parquet(f, index=False)
        json.dump(meta, open(os.path.join(bits_dir, f"{a.family}_{m}.json"), "w"), indent=1)
        msg = f"{m} {a.family} n_train={meta['n_train']} n_test={meta['n_test']} " + \
              " ".join(f"{k}={out[f'bits_{k}'].mean():.4f}" for k in SETS) + f" {meta['seconds']}s"
        print(msg, file=log, flush=True)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
