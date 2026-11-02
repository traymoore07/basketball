"""Probe on the timing month only: (a) per-class decomposition of the neural history gain; (b) frozen logit with the
previous segment's CLASS one-hot (9 = none) added to set A and to set E. Diagnostic, not a pre-registered set."""
import os, sys, numpy as np, pandas as pd, scipy.sparse as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.models.event import EventDesign, MultinomialLogit, event_classes
from hoopslab.metrics.information import bits_per_event
from hoopslab.experiments.history import SETS
from run_hist import load_segments, OUT, TRAIN_DAYS
g = pd.read_parquet(os.path.join(OUT, "bits", "gru_1999-12.parquet"))
print("per-class mean bits (A_nn, E_gru, gain mb, share):")
for c, d in g.groupby("y"):
    print(c, round(d.bits_A_nn.mean(), 4), round(d.bits_E_gru.mean(), 4), round(1000 * (d.bits_A_nn - d.bits_E_gru).mean(), 1), round(len(d) / len(g), 3))
P = load_segments([1997, 1998, 1999, 2000]); m = "1999-12"
te = P[(P.month == m) & (P.season >= 2000)]; ms = te.game_date.min().normalize()
tr = P[(P.game_date < ms) & (P.game_date >= ms - pd.Timedelta(days=TRAIN_DAYS))]
def prevclass(Q):
    y = event_classes(Q); df = pd.DataFrame({"g": Q.game_id.to_numpy(), "t": Q.off_team_id.to_numpy(), "s": Q.possession_seq.to_numpy(), "y": y}, index=Q.index)
    pv = df.sort_values(["g", "t", "s"], kind="stable").groupby(["g", "t"], sort=False)["y"].shift(1).reindex(Q.index).to_numpy()
    M = np.zeros((len(Q), 9)); has = ~np.isnan(pv); M[has, pv[has].astype(int)] = 1; M[~has, 8] = 1; return sp.csr_matrix(M)
d = EventDesign(tr); ytr, yte = event_classes(tr), event_classes(te); btr, bte = d.all_blocks(tr), d.all_blocks(te)
btr["PC"], bte["PC"] = prevclass(tr), prevclass(te)
for name, blocks in {"A": SETS["A"], "A+PC": SETS["A"] + ["PC"], "E": SETS["E"], "E+PC": SETS["E"] + ["PC"]}.items():
    mdl = MultinomialLogit(l2=1e-4, maxiter=1000).fit(d.select(btr, blocks), ytr)
    b = bits_per_event(mdl.predict_proba(d.select(bte, blocks)), yte)
    print(name, round(b.mean(), 4), "converged", mdl.converged, flush=True)
