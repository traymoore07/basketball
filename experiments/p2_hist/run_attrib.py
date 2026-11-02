"""H3b attribution step (docs/prereg/01 E-HIST, "omitted-slow-variable attribution"): add the slow variables one
at a time to the latent set and re-measure history_residual. Also measures the previous-segment CLASS block
(PC: 8 classes + none), which the frozen L1 block (previous points) does not carry, and whether the slow
variables absorb it.

Runs on a pre-specified subset of prequential months (every sixth test month of the development window, in
chronological order, index 0, 3, 6, ...) with the frozen multinomial logit; same training windows as run_hist.py.
Sets fitted per month (bits per event stored per segment):
    A, F, G (as the main run), E+PC, G+PC,
    F+v, G+v, G+v+PC for v in {fouls (def/off team fouls in period + bonus flag), stint (stint length, log1p),
    lineup (lineup-change flag), gap (seconds since previous segment, log1p)}, and F+all, G+all, G+all+PC.
Writes results/p2_hist/bits/attrib_{month}.parquet."""
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.experiments.history import SETS  # noqa: E402
from hoopslab.metrics.information import bits_per_event  # noqa: E402
from hoopslab.models.event import EventDesign, MultinomialLogit, event_classes  # noqa: E402
from run_hist import L2, MAXITER, OUT, TRAIN_DAYS, load_segments  # noqa: E402

SLOW = {"fouls": ["def_fouls_period", "off_fouls_period", "def_in_bonus"], "stint": ["stint_len_s"], "lineup": ["lineup_change"], "gap": ["secs_since_prev"]}


def load_with_slow(seasons):
    P = load_segments(seasons)
    sl = pd.concat([pd.read_parquet(os.path.join(OUT, f"slow_{S}.parquet")) for S in seasons if os.path.exists(os.path.join(OUT, f"slow_{S}.parquet"))],
                   ignore_index=True)
    P = P.merge(sl, on=["game_id", "possession_seq"], how="left")
    for c in ("def_fouls_period", "off_fouls_period", "def_in_bonus", "lineup_change"):
        P[c] = P[c].fillna(0)
    P["stint_len_s"] = np.log1p(P["stint_len_s"].fillna(0))
    P["secs_since_prev"] = np.log1p(P["secs_since_prev"].clip(lower=0).fillna(0))
    return P


def prevclass(Q):
    y = event_classes(Q)
    df = pd.DataFrame({"g": Q["game_id"].to_numpy(), "t": Q["off_team_id"].to_numpy(), "s": Q["possession_seq"].to_numpy(), "y": y}, index=Q.index)
    pv = df.sort_values(["g", "t", "s"], kind="stable").groupby(["g", "t"], sort=False)["y"].shift(1).reindex(Q.index).to_numpy()
    M = np.zeros((len(Q), 9))
    has = ~np.isnan(pv)
    M[has, pv[has].astype(int)] = 1
    M[~has, 8] = 1
    return sp.csr_matrix(M)


def slow_block(Q, cols):
    X = Q[cols].to_numpy(float)
    if "def_fouls_period" in cols:  # fouls as one-hot 0..6+ per team plus the bonus flag
        d = np.minimum(Q["def_fouls_period"].to_numpy(int), 6)
        o = np.minimum(Q["off_fouls_period"].to_numpy(int), 6)
        X = np.concatenate([np.eye(7)[d], np.eye(7)[o], Q[["def_in_bonus"]].to_numpy(float)], 1)
    return sp.csr_matrix(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", nargs="+", type=int, required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    bits_dir = os.path.join(OUT, "bits")
    seasons_needed = sorted(set(range(min(a.test_seasons) - 3, max(a.test_seasons) + 1)) & set(range(1997, 2021)))
    P = load_with_slow(seasons_needed)
    months = sorted(P.loc[P["season"].isin(a.test_seasons), "month"].unique())
    months = [m for i, m in enumerate(months) if i % a.every == a.offset]
    log = open(os.path.join(OUT, f"run_attrib_{min(a.test_seasons)}_{max(a.test_seasons)}.log"), "a")
    for m in months:
        f = os.path.join(bits_dir, f"attrib_{m}.parquet")
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
        d = EventDesign(tr)
        ytr, yte = event_classes(tr), event_classes(te)
        btr, bte = d.all_blocks(tr), d.all_blocks(te)
        btr["PC"], bte["PC"] = prevclass(tr), prevclass(te)
        for k, cols in SLOW.items():
            btr[k], bte[k] = slow_block(tr, cols), slow_block(te, cols)
        allslow = list(SLOW)
        sets = {"A": SETS["A"], "F": SETS["F"], "G": SETS["G"], "E_PC": SETS["E"] + ["PC"], "G_PC": SETS["G"] + ["PC"]}
        for k in SLOW:
            sets[f"F_{k}"] = SETS["F"] + [k]
            sets[f"G_{k}"] = SETS["G"] + [k]
        sets["F_all"] = SETS["F"] + allslow
        sets["G_all"] = SETS["G"] + allslow
        sets["G_all_PC"] = SETS["G"] + allslow + ["PC"]
        out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(), "season": te["season"].to_numpy(),
                            "month": m, "y": yte})
        conv = {}
        for name, blocks in sets.items():
            mdl = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(d.select(btr, blocks), ytr)
            out[f"bits_{name}"] = bits_per_event(mdl.predict_proba(d.select(bte, blocks)), yte)
            conv[name] = bool(mdl.converged)
        out.to_parquet(f, index=False)
        json.dump({"month": m, "n_train": int(len(tr)), "n_test": int(len(te)), "converged": conv, "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(bits_dir, f"attrib_{m}.json"), "w"), indent=1)
        msg = f"{m} attrib n_test={len(te)} " + " ".join(f"{k}={out[f'bits_{k}'].mean():.4f}" for k in ("A", "F", "G", "E_PC", "G_PC", "F_fouls", "G_fouls", "F_all", "G_all", "G_all_PC")) + f" {time.time()-t0:.0f}s"
        print(msg, file=log, flush=True)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
