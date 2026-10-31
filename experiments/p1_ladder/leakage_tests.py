"""Leakage tests T1, T2, T3, T6, T7 (and T5) for every P1 model, on 20 random forecast times
per development season (docs/prereg/06: required before any result is reported).

All nine models are run through one composite forecaster so that each frozen detector
compares every model's output in one pass (the detectors take elementwise differences of
p_appear and cond[target]; concatenating the models' arrays keeps that exact).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from hoopslab import features  # noqa: E402
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter  # noqa: E402
from hoopslab.backtest import leakage as lk  # noqa: E402
from hoopslab.backtest.runner import make_requests, outcomes_table  # noqa: E402
from hoopslab.forecast import PlayerForecast  # noqa: E402
from hoopslab.models import baselines  # noqa: E402
from hoopslab.targets import CORE_TARGETS  # noqa: E402
import models as p1models  # noqa: E402
from exclusions import drop_unknown_team_games  # noqa: E402

PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}


class Composite:
    """Runs every ladder model; returns one PlayerForecast with stacked arrays."""
    name = "composite_L0_L4"

    def __init__(self):
        self.models = p1models.ladder_models()

    def fit(self, view):
        for m in self.models:
            m.fit(view)

    def predict(self, view, requests, targets):
        fs = [m.predict(view, requests, list(targets)) for m in self.models]
        cond = {t: np.concatenate([f.cond[t] for f in fs], axis=1) for t in targets}
        return PlayerForecast(self.name, view.t, fs[0].keys, np.stack([f.p_appear for f in fs], axis=1), cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p1_ladder"))
    a = ap.parse_args()
    t0 = time.time()
    adapter = CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A")
    tables = adapter.load_tables()
    store, _ = build_store(adapter, strict=True)
    g = store.full_table("GAME")
    g1 = g[(g["record_version"] == 1) & g["phase"].isin(PHASES)].copy()
    g1, dropped_phantom = drop_unknown_team_games(g1)  # DI-1
    g1["season"] = g1["season_id"].str.split("-").str[1].astype(int)
    outc = outcomes_table(store)
    res = {}
    for S in a.seasons:
        rng = np.random.default_rng(1000 + S)
        tips = sorted(g1.loc[g1["season"] == S, "scheduled_tip"].unique())
        pick = rng.choice(len(tips), size=min(a.n, len(tips)), replace=False)
        times = [pd.Timestamp(tips[i]) - pd.Timedelta(minutes=60) for i in sorted(pick)]
        fac = lambda s: Composite()  # noqa: E731
        r = {"n_times": len(times), "times": [str(t) for t in times],
             "di1_dropped_game_ids": dropped_phantom.loc[dropped_phantom["season_id"] == f"nba-{S}", "game_id"].tolist()}
        r["T1"] = lk.truncation_invariance(store, fac, times, targets=tuple(CORE_TARGETS))
        r["T2"] = lk.future_perturbation_invariance(store, fac, times, targets=tuple(CORE_TARGETS))
        r["T3"] = lk.id_relabel_invariance(tables, fac, times, targets=tuple(CORE_TARGETS))
        t6 = []
        for t in times:
            view = store.view(t)
            gg = view.table("GAME")
            games = gg[(gg["scheduled_tip"] == t + pd.Timedelta(minutes=60)) & (gg["status"] == "scheduled")]
            if len(games):
                t6.append(lk.population_check(make_requests(view, games), outc)["share_non_appearing"])
        r["T6"] = {"test": "T6_population", "share_non_appearing_min": float(min(t6)), "share_non_appearing_mean": float(np.mean(t6)),
                   "passed": bool(min(t6) > 0)}
        r["T7"] = lk.feature_recompute_test(store, features.player_history, times)
        r["passed"] = all(r[k]["passed"] for k in ("T1", "T2", "T3", "T6", "T7"))
        res[S] = r
        print(f"season {S}: " + " ".join(f"{k}={'PASS' if r[k]['passed'] else 'FAIL'}" for k in ("T1", "T2", "T3", "T6", "T7")),
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(a.out, f"leakage_{'_'.join(map(str, a.seasons))}.json"), "w"), indent=1, default=str)
    t5 = lk.static_access_scan([baselines, features, p1models])
    res["T5"] = t5
    json.dump(res, open(os.path.join(a.out, f"leakage_{'_'.join(map(str, a.seasons))}.json"), "w"), indent=1, default=str)
    print("T5", "PASS" if t5["passed"] else "FAIL", t5["hits"])


if __name__ == "__main__":
    main()
