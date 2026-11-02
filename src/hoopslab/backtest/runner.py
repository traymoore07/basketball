"""Rolling-origin ("living through history") backtest.

For each forecast time t (default: scheduled tip - 60 min):
    view = store.view(t)            # only records with knowledge_time <= t
    requests = as-of roster of every team playing at t + lead
    for each model: model.fit(view); forecast = model.predict(view, requests)
Outcomes are joined only afterwards, from the full store, by the scorer.

The request population is the as-of ROSTER. It is never "players who played",
since that would select on the outcome (leakage checklist L-E1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..asof import AsOfStore, to_utc
from ..features import roster_at
from ..forecast import PlayerForecast
from ..metrics import scoring as sc
from ..targets import BINS, CORE_TARGETS, PLAYER_TARGETS, player_outcomes


class LockboxError(RuntimeError):
    pass


@dataclass
class BacktestSpec:
    start: pd.Timestamp
    end: pd.Timestamp
    lead: pd.Timedelta = pd.Timedelta(minutes=60)
    targets: list = field(default_factory=lambda: list(CORE_TARGETS))
    lockbox: list = field(default_factory=list)       # [(start, end)] sealed windows
    unseal_manifest_hash: str | None = None            # must match registry freeze to score lockbox

    def in_lockbox(self, t) -> bool:
        return any(to_utc(a) <= t <= to_utc(b) for a, b in self.lockbox)


def forecast_points(store: AsOfStore, spec: BacktestSpec):
    """Yield (t, games_df). Candidate tips come from the schedule as known at `spec.start`
    minus one day; each point re-reads the schedule as of t (reschedules respected)."""
    start, end = to_utc(spec.start), to_utc(spec.end)
    g0 = store.view(start - pd.Timedelta(days=1)).table("GAME")
    tips = sorted(set(g0.loc[(g0["scheduled_tip"] >= start) & (g0["scheduled_tip"] <= end), "scheduled_tip"]))
    for tip in tips:
        t = tip - spec.lead
        g = store.view(t).table("GAME")
        games = g[(g["scheduled_tip"] == tip) & (g["status"] == "scheduled")]
        if len(games):
            yield t, games


def make_requests(view, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, gm in games.iterrows():
        r = roster_at(view, [gm["home_team_id"], gm["away_team_id"]], gm["scheduled_tip"])
        r = r.assign(game_id=gm["game_id"], tip=gm["scheduled_tip"],
                     is_home=r["team_id"] == gm["home_team_id"])
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def run_backtest(store: AsOfStore, spec: BacktestSpec, models: list, verbose: bool = False,
                 frozen_hash: str | None = None):
    """Run all models over the evaluation window. Returns {model_name: [PlayerForecast]}."""
    out = {m.name: [] for m in models}
    requests_log = []
    for t, games in forecast_points(store, spec):
        if spec.in_lockbox(t):
            if spec.unseal_manifest_hash is None or spec.unseal_manifest_hash != frozen_hash:
                raise LockboxError(f"forecast time {t} is in a sealed lockbox window; "
                                   "unseal requires the frozen registry manifest hash")
        view = store.view(t)
        req = make_requests(view, games)
        req["forecast_time"] = t
        requests_log.append(req)
        for m in models:
            m.fit(view)
            f = m.predict(view, req, spec.targets)
            f.validate()
            f.meta = {"accessed": sorted(view.accessed)}
            out[m.name].append(f)
        if verbose:
            print(t, len(req))
    return out, pd.concat(requests_log, ignore_index=True) if requests_log else pd.DataFrame()


def outcomes_table(store: AsOfStore) -> pd.DataFrame:
    pg = store.full_table("PLAYER_GAME").sort_values("knowledge_time").drop_duplicates(
        ["game_id", "player_id"], keep="last")
    return player_outcomes(pg)


def score_forecasts(store: AsOfStore, forecasts: dict, targets: list, modes=("U", "A"),
                    seed: int = 0, tail_thresholds: dict | None = None) -> pd.DataFrame:
    """Long table of per-unit losses: one row per (model, target, mode, player-game)."""
    rng = np.random.default_rng(seed)
    outc = outcomes_table(store)
    tail_thresholds = tail_thresholds or {}
    rows = []
    for model, flist in forecasts.items():
        for f in flist:
            keys = f.keys.merge(outc, on=["game_id", "player_id"], how="left", suffixes=("", "_o"))
            appeared = keys["appeared"].fillna(False).to_numpy().astype(bool)
            for t in targets:
                y_all = keys[t].fillna(0).to_numpy().astype(int)
                for mode in modes:
                    if mode == "U":
                        pmf, mask = f.unconditional(t), np.ones(len(keys), bool)
                    elif mode == "A":
                        pmf, mask = f.cond[t], appeared
                    else:
                        continue
                    if mask.sum() == 0:
                        continue
                    P, y = pmf[mask], y_all[mask]
                    d = {
                        "model": model, "target": t, "mode": mode,
                        "game_id": keys["game_id"].to_numpy()[mask],
                        "player_id": keys["player_id"].to_numpy()[mask],
                        "team_id": keys["team_id"].to_numpy()[mask],
                        "forecast_time": f.forecast_time, "cluster": f.forecast_time.floor("D"),
                        "y": y, "log": sc.log_score(P, y), "crps": sc.crps_pmf(P, y),
                        "pit": sc.randomized_pit(P, y, rng),
                        "ae_median": sc.abs_error_of_median(P, y), "se_mean": sc.sq_error_of_mean(P, y),
                        "mean": sc.pmf_mean(P),
                    }
                    if t in BINS:
                        d["rps"] = sc.rps(sc.bin_pmf(P, BINS[t]), sc.bin_outcome(y, BINS[t]))
                    if t in tail_thresholds:
                        d["tw_crps"] = sc.tw_crps_upper(P, y, tail_thresholds[t])
                    rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def paired(scores: pd.DataFrame, ref: str, cand: str, target: str, mode: str, metric: str):
    """Align two models' losses on identical units."""
    a = scores[(scores.model == ref) & (scores.target == target) & (scores["mode"] == mode)]
    b = scores[(scores.model == cand) & (scores.target == target) & (scores["mode"] == mode)]
    k = ["game_id", "player_id"]
    m = a[k + ["cluster", metric]].merge(b[k + [metric]], on=k, suffixes=("_ref", "_cand"))
    return m
