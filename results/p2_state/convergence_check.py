"""Timing-month convergence check (1999-12, not one of the 62 summary months): S6 and S6_H fitted with
maxiter 1000 (E-HIST protocol) and 2000; held-out mean bits compared. Decides the E-STATE maxiter before launch."""
import os, sys, time, json
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments", "p2_hist")); sys.path.insert(0, os.path.join(ROOT, "experiments", "p2_state"))
from hoopslab.metrics.information import bits_per_event
from hoopslab.models.event import EventDesign, MultinomialLogit, event_classes
from run_hist import L2, TRAIN_DAYS
from run_state import load_all, sets_for
from state_blocks import state_blocks
P, _ = load_all([1997, 1998, 1999, 2000])
te = P[(P["month"] == "1999-12") & (P["season"] >= 2000)]
m_start = te["game_date"].min().normalize()
tr = P[(P["game_date"] < m_start) & (P["game_date"] >= m_start - pd.Timedelta(days=TRAIN_DAYS))]
d = EventDesign(tr); ytr, yte = event_classes(tr), event_classes(te)
btr, bte = d.all_blocks(tr), d.all_blocks(te); btr.update(state_blocks(tr, ytr)); bte.update(state_blocks(te, yte))
sets = sets_for("mnl"); out = {}
for name in ("S6", "S6_H"):
    Xtr, Xte = d.select(btr, sets[name]), d.select(bte, sets[name])
    for it in (1000, 2000, 4000):
        t0 = time.time(); m = MultinomialLogit(l2=L2, maxiter=it).fit(Xtr, ytr)
        b = bits_per_event(m.predict_proba(Xte), yte).mean()
        out[f"{name}_{it}"] = {"bits": float(b), "converged": m.converged, "seconds": round(time.time() - t0)}
        print(name, it, out[f"{name}_{it}"], flush=True)
out["residual_mb"] = {it: 1000 * (out[f"S6_{it}"]["bits"] - out[f"S6_H_{it}"]["bits"]) for it in (1000, 2000, 4000)}
print(out["residual_mb"])
json.dump(out, open(os.path.join(ROOT, "results", "p2_state", "convergence_check.json"), "w"), indent=1)
