"""Positive control (canary), replacing the weak first implementation in run_hist.py: a feature computed from the NEXT
segment of the GAME (either team): its start type one-hot and the change in the offence's margin between this segment's
start and the next segment's start (= this segment's points minus any opponent points, which here is simply this
segment's points because the next segment starts right after). This encodes the current outcome almost deterministically
and must give a LARGE gain. It is never an input to any model. Same months as the attribution subset (every sixth
test month); writes results/p2_hist/bits/canary2_{month}.parquet with bits_A and bits_A_CANARY2."""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.experiments.history import SETS
from hoopslab.metrics.information import bits_per_event
from hoopslab.models.event import START, EventDesign, MultinomialLogit, event_classes
from run_hist import L2, MAXITER, OUT, TRAIN_DAYS, load_segments

def canary2(Q):
    Q2 = Q.sort_values(["game_id", "possession_seq"], kind="stable")
    g = Q2.groupby("game_id", sort=False)
    nxt_start = g["start_type"].shift(-1)
    nxt_margin = g["off_margin_start"].shift(-1)
    nxt_off = g["off_team_id"].shift(-1)
    same = (nxt_off == Q2["off_team_id"])
    # margin seen from THIS offence at the next segment start: flip sign when the next offence is the other team
    m_next = np.where(same, nxt_margin, -nxt_margin)
    delta = pd.Series(m_next - Q2["off_margin_start"].to_numpy(), index=Q2.index)
    st = nxt_start.map({s: i + 1 for i, s in enumerate(START + ["after_dead_ball"])}).fillna(0).astype(int)
    M = np.zeros((len(Q2), 6 + 7))
    M[np.arange(len(Q2)), st.to_numpy()] = 1
    d = np.clip(delta.fillna(0).to_numpy().astype(int), -1, 5) + 1   # -1..5 -> 0..6
    M[np.arange(len(Q2)), 6 + d] = 1
    M[same.to_numpy(), 5] = 1  # next segment is the same offence (offensive rebound continuation)
    return sp.csr_matrix(pd.DataFrame(M, index=Q2.index).reindex(Q.index).to_numpy())

def main():
    seasons = list(range(2000, 2021))
    P = load_segments(list(range(1997, 2021)))
    months = sorted(P.loc[P["season"].isin(seasons), "month"].unique())
    months = [m for i, m in enumerate(months) if i % 6 == 0]
    for m in months:
        f = os.path.join(OUT, "bits", f"canary2_{m}.parquet")
        if os.path.exists(f):
            continue
        te = P[(P["month"] == m) & (P["season"] >= 2000)]
        ms = te["game_date"].min().normalize()
        tr = P[(P["game_date"] < ms) & (P["game_date"] >= ms - pd.Timedelta(days=TRAIN_DAYS))]
        if len(tr) < 20000 or not len(te):
            continue
        t0 = time.time()
        d = EventDesign(tr); ytr, yte = event_classes(tr), event_classes(te)
        XA_tr, XA_te = d.blocks(tr, SETS["A"]), d.blocks(te, SETS["A"])
        mA = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(XA_tr, ytr)
        mC = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(sp.hstack([XA_tr, canary2(tr)], format="csr"), ytr)
        out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(), "month": m, "y": yte,
                            "bits_A": bits_per_event(mA.predict_proba(XA_te), yte),
                            "bits_A_CANARY2": bits_per_event(mC.predict_proba(sp.hstack([XA_te, canary2(te)], format="csr")), yte)})
        out.to_parquet(f, index=False)
        print(m, "A", round(out.bits_A.mean(), 4), "A+canary2", round(out.bits_A_CANARY2.mean(), 4), f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
