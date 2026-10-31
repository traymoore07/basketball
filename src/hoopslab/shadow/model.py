"""L2-A-SHADOW-v0.1: the frozen P1 L2 (PlayerMean, half-life 5 appearances), mode A only.

The mathematics are NOT re-implemented here. The snapshot tables are loaded into the same AsOfStore
the P1 backtest used, a view at the information cutoff is taken, and the frozen class
hoopslab.models.baselines.PlayerMean(half_life=5.0) is fit and asked for its conditional pmfs. The
appearance probability the class also returns is discarded: this model forecasts
P(stat | the player appears) and nothing else.

Identity with P1 is tested by tests/test_shadow.py (synthetic store) and experiments/shadow/
replay_test.py (development-period games against the P1 forecasts on disk).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from ..asof import AsOfStore, to_utc
from ..models.baselines import PlayerMean
from ..targets import BINS, PLAYER_TARGETS
from .interface import FORECASTABLE_STATES, AvailabilityState, Snapshot

MODEL_VERSION = "L2-A-SHADOW-v0.1"
HALF_LIFE_APPEARANCES = 5.0
TARGETS = ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]
FORECAST_MODE = "A"  # conditional on appearance
LEAD_MINUTES = 60    # information cutoff = scheduled tip - 60 min (frozen forecast time)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# files whose bytes define the mathematics of this model (relative to the repository root)
MODEL_FILES = ["src/hoopslab/models/baselines.py", "src/hoopslab/features.py", "src/hoopslab/targets.py",
               "src/hoopslab/forecast.py", "src/hoopslab/asof.py", "src/hoopslab/shadow/model.py"]
PMF_CONSTRUCTION = {
    "MIN": "normal(mu = exponentially weighted mean of past minutes over appearances, half-life 5 appearances; sd = 6) "
           "discretised to integer minutes 0..70 (bins k-0.5..k+0.5, open ends)",
    "counts": "negative binomial(mean = exponentially weighted per-appearance mean, half-life 5; dispersion r per target "
              "fitted by method of moments on all visible appearances, clipped to [1.5, 500]) over 0..k_max, tail mass "
              "folded into the top bin",
    "unknown_player": "league pmf of the target over all visible appearances (+0.01 smoothing) when the player has no "
                      "visible appearance",
    "k_max": {t: PLAYER_TARGETS[t][0] for t in TARGETS},
}


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def model_implementation_hash() -> dict:
    """sha256 per model file and one combined identity (sha256 of the sorted 'path:hash' lines)."""
    root = repo_root()
    per = {}
    for f in MODEL_FILES:
        per[f] = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
    combined = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(per.items())).encode()).hexdigest()
    return {"files": per, "combined": combined}


def store_from_snapshot(snap: Snapshot) -> AsOfStore:
    """The snapshot tables as the AsOfStore the frozen model reads (GAME v1-style rows, PLAYER_GAME rows)."""
    g = snap.game.copy()
    g["record_version"] = 1
    g["knowledge_time_quality"] = "executor"
    g["source"] = snap.manifest.get("provider_id", "executor")
    pg = snap.player_game.copy()
    pg["status"] = np.where(pg["seconds"].to_numpy() > 0, "played", "dnp_coach")
    pg["started"] = False
    pg["record_version"] = 1
    pg["knowledge_time_quality"] = "executor"
    pg["source"] = snap.manifest.get("provider_id", "executor")
    return AsOfStore({"GAME": g, "PLAYER_GAME": pg})


def forecast(snap: Snapshot) -> dict:
    """Run the frozen L2 on the snapshot. Returns per-player conditional pmfs for the forecastable players.

    Output: {"players": [{game_id, player_id, team_id, availability_state, history_appearances, pmf: {target: list}}],
             "model_version", "half_life_appearances", "mode", "targets", "cutoff"}.
    Players with availability_state OUT are returned without a distribution.
    """
    store = store_from_snapshot(snap)
    t = to_utc(snap.cutoff)
    view = store.view(t)
    # every row the model can see is at or before the cutoff by construction of the store; assert anyway
    for name in ("GAME", "PLAYER_GAME"):
        kt = view.table(name)["knowledge_time"]
        assert len(kt) == 0 or kt.max() <= t, f"{name}: knowledge_time after the cutoff is visible"
    m = PlayerMean(half_life=HALF_LIFE_APPEARANCES)
    m.fit(view)
    el = snap.eligible.copy()
    el["availability_state"] = el["availability_state"].astype(str)
    fc_rows = el[el["availability_state"].isin({s.value for s in FORECASTABLE_STATES})].reset_index(drop=True)
    out = {"model_version": MODEL_VERSION, "half_life_appearances": HALF_LIFE_APPEARANCES, "mode": FORECAST_MODE,
           "interpretation": "P(stat | the player appears); no appearance probability is produced",
           "targets": TARGETS, "cutoff": str(t), "players": []}
    n_app = {pid: int(len(g)) for pid, g in m.groups.items()}
    if len(fc_rows):
        req = fc_rows[["game_id", "player_id", "team_id"]]
        f = m.predict(view, req, TARGETS)
        f.validate()
        for i, r in fc_rows.iterrows():
            out["players"].append({
                "game_id": r["game_id"], "player_id": r["player_id"], "team_id": r["team_id"],
                "availability_state": r["availability_state"], "status_source": str(r.get("status_source", "")),
                "status_time": str(r.get("status_time", "")),
                "history_appearances": n_app.get(r["player_id"], 0),
                "history_fallback": "league_pmf" if n_app.get(r["player_id"], 0) == 0 else "player",
                "pmf": {tg: f.cond[tg][i].astype(np.float64).tolist() for tg in TARGETS},
            })
    for _, r in el[el["availability_state"] == AvailabilityState.OUT.value].iterrows():
        out["players"].append({"game_id": r["game_id"], "player_id": r["player_id"], "team_id": r["team_id"],
                               "availability_state": "OUT", "status_source": str(r.get("status_source", "")),
                               "status_time": str(r.get("status_time", "")), "history_appearances": n_app.get(r["player_id"], 0),
                               "history_fallback": None, "pmf": None})
    out["dispersion_r"] = {tg: float(m.r[tg]) for tg in TARGETS if tg in m.r}
    out["league_appearances_in_history"] = int(len(m.app))
    return out


def derived_quantities(pmf: list[float], target: str) -> dict:
    """Mean, quantiles and threshold probabilities computed directly from the pmf (no model logic)."""
    p = np.asarray(pmf, dtype=np.float64)
    p = p / p.sum()
    ks = np.arange(len(p))
    F = np.cumsum(p)
    q = {f"p{int(round(100 * qq))}": int((F < qq - 1e-12).sum()) for qq in QUANTILES}
    thr = {f"P(X>={e})": float(p[ks >= e].sum()) for e in BINS.get(target, [])}
    return {"mean": float(p @ ks), "median": q["p50"], **q, "threshold_probabilities": thr}
