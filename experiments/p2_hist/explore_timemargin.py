"""EXPLORATORY post-hoc diagnostic (not pre-registered; never enters a decision): is the history residual an
end-game regime that the frozen compact state cannot see?

The frozen state S carries the clock only as period + a clutch flag (last 4 minutes, |margin| <= 5) and the
margin only as 6-point buckets. The pooled history gain concentrates on the free-throw classes, which is what
deliberate late-game fouling produces. This script adds a TIME x MARGIN block (TM: seconds left in the period in
six bins crossed with a seven-level signed margin, for the fourth period and overtime only, plus the same clock
bins alone for periods 1-3) to the frozen sets and re-measures the history contrasts on the attribution months.
Sets per month: A, F, G, A_TM, F_TM, G_TM, F_all_TM, G_all_TM (the slow variables of run_attrib.py included).
Writes results/p2_hist/bits/explore_tm_{month}.parquet (+ .json)."""
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
from run_attrib import SLOW, load_with_slow, slow_block  # noqa: E402
from run_hist import L2, MAXITER, OUT, TRAIN_DAYS  # noqa: E402

CLOCK_BINS = [10, 30, 60, 120, 180]  # seconds left in the period -> 6 bins
MARGIN_BINS = [-8.5, -3.5, -0.5, 0.5, 3.5, 8.5]  # offence margin -> 7 levels


def tm_block(Q):
    per = Q["period"].to_numpy()
    el = Q["start_elapsed_s"].to_numpy()
    per_len = np.where(per <= 4, 720.0, 300.0)
    per_start = np.where(per <= 4, (per - 1) * 720.0, 2880.0 + (per - 5) * 300.0)
    left = np.clip(per_start + per_len - el, 0, None)
    cb = np.searchsorted(CLOCK_BINS, left, side="right")  # 0..5
    mb = np.searchsorted(MARGIN_BINS, Q["off_margin_start"].to_numpy(float), side="right")  # 0..6
    late = per >= 4
    cell = np.where(late, cb * 7 + mb, 42 + cb)  # 42 late cells + 6 early clock bins
    return sp.csr_matrix(np.eye(48)[cell])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    a = ap.parse_args()
    seasons_needed = sorted({(int(m[:4]) + (1 if int(m[5:]) >= 10 else 0)) - k for m in a.months for k in range(4)} & set(range(1997, 2021)))
    P = load_with_slow(seasons_needed)
    bits_dir = os.path.join(OUT, "bits")
    for m in a.months:
        f = os.path.join(bits_dir, f"explore_tm_{m}.parquet")
        if os.path.exists(f):
            continue
        te = P[(P["month"] == m) & (P["season"] >= 2000)]
        m_start = te["game_date"].min().normalize()
        tr = P[(P["game_date"] < m_start) & (P["game_date"] >= m_start - pd.Timedelta(days=TRAIN_DAYS))]
        t0 = time.time()
        d = EventDesign(tr)
        ytr, yte = event_classes(tr), event_classes(te)
        btr, bte = d.all_blocks(tr), d.all_blocks(te)
        btr["TM"], bte["TM"] = tm_block(tr), tm_block(te)
        for k, cols in SLOW.items():
            btr[k], bte[k] = slow_block(tr, cols), slow_block(te, cols)
        allslow = list(SLOW)
        sets = {"A": SETS["A"], "F": SETS["F"], "G": SETS["G"], "A_TM": SETS["A"] + ["TM"], "F_TM": SETS["F"] + ["TM"], "G_TM": SETS["G"] + ["TM"],
                "F_all_TM": SETS["F"] + allslow + ["TM"], "G_all_TM": SETS["G"] + allslow + ["TM"]}
        out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(), "month": m, "y": yte,
                            "late": (te["period"].to_numpy() >= 4)})
        conv = {}
        for name, blocks in sets.items():
            mdl = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(d.select(btr, blocks), ytr)
            out[f"bits_{name}"] = bits_per_event(mdl.predict_proba(d.select(bte, blocks)), yte)
            conv[name] = bool(mdl.converged)
        out.to_parquet(f, index=False)
        json.dump({"month": m, "n_train": int(len(tr)), "n_test": int(len(te)), "converged": conv, "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(bits_dir, f"explore_tm_{m}.json"), "w"), indent=1)
        msg = f"{m} explore_tm n_test={len(te)} " + " ".join(f"{k}={out[f'bits_{k}'].mean():.4f}" for k in sets) + f" {time.time()-t0:.0f}s"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
