"""P1 ladder: rolling-origin backtest of L0-L4 for ONE NBA season (nba_stats family).

Replicates hoopslab.backtest.runner.run_backtest exactly (forecast_points, make_requests,
fit + predict at every forecast time, lockbox guard) with three documented additions:
  * chunked enumeration of forecast points (forecast_points reads the schedule once, as known at
    chunk start - 1 day; the chunk length is derived from the measured GAME v1 publication lag so
    that every game in a chunk is already published at that snapshot: chunk_days = lag_days - 1);
  * no forecasts for preseason games (P1 phases only);
  * DI-1 structural exclusion (experiments/p1_ladder/exclusions.py).
A coverage assertion at the end verifies that every P1-phase game inside the season window (minus
DI-1 rows) received requests; the run fails loudly otherwise.
Writes requests, forecasts (float32 pmfs), per-unit scores and as-of strata.

Run 1 (2026-10-29/30) used fixed 7-day chunks and silently skipped every 7th game day (C11 bugfix).
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
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter  # noqa: E402
from hoopslab.backtest.runner import BacktestSpec, LockboxError, forecast_points, make_requests, score_forecasts  # noqa: E402
from hoopslab.backtest.strata import compute_strata  # noqa: E402
from hoopslab.targets import CORE_TARGETS  # noqa: E402
from models import ladder_models  # noqa: E402
from exclusions import drop_unknown_team_games  # noqa: E402

TAIL = {"PTS": 25, "PRA": 40, "REB": 10, "AST": 8}
PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}
LEAD = pd.Timedelta(minutes=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p1_ladder"))
    ap.add_argument("--dry-run", action="store_true", help="enumerate forecast points and requests only (no models, no scores)")
    a = ap.parse_args()
    S, out = a.season, a.out
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(ROOT, "experiments", "p1_ladder", "config.json")))
    pre = json.load(open(os.path.join(out, "preflight.json")))
    assert pre["all_passed"], "preflight did not pass"
    log = open(os.path.join(out, f"log_{S}{'_dry' if a.dry_run else ''}.txt"), "a")

    def P(*x):
        print(*x, file=log, flush=True)
        print(*x, flush=True)

    t0 = time.time()
    store, rep = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A"), strict=True)
    P(f"season {S}: store built in {time.time()-t0:.0f}s rows={rep['rows']}")
    seas = store.full_table("SEASON").set_index("season_id")
    gfull = store.full_table("GAME")
    g1 = gfull[gfull["record_version"] == 1]
    lags = (g1["scheduled_tip"] - g1["knowledge_time"]).unique()
    assert len(lags) == 1, f"GAME v1 publication lag is not a single value: {lags}"
    lag_days = int(pd.Timedelta(lags[0]).days)
    chunk_days = lag_days - 1
    assert chunk_days >= 1
    P(f"season {S}: GAME v1 lag={lag_days}d -> chunk length {chunk_days}d")

    def win(s):
        r = seas.loc[f"nba-{s}"]
        return (pd.Timestamp(r["start_date"], tz="UTC"),
                pd.Timestamp(r["end_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=59))

    lockbox = [win(s) for s in cfg["windows"]["lockbox"]]
    start, end = win(S)
    models = [] if a.dry_run else ladder_models()
    targets = list(CORE_TARGETS)
    req_log, fc_store = [], {m.name: [] for m in models}
    accessed = {m.name: set() for m in models}
    score_parts, n_days = 0, 0
    excluded = []
    c0 = start
    while c0 <= end:
        c1 = min(c0 + pd.Timedelta(days=chunk_days - 1, hours=23, minutes=59), end)
        spec = BacktestSpec(start=c0, end=c1, targets=targets, lockbox=lockbox)
        chunk_fc = {m.name: [] for m in models}
        for t, games in forecast_points(store, spec):
            if spec.in_lockbox(t):
                raise LockboxError(f"forecast time {t} is inside a sealed lockbox window")
            games = games[games["phase"].isin(PHASES)]
            games, dropped = drop_unknown_team_games(games)
            if len(dropped):
                excluded.extend({"forecast_time": str(t), "game_id": g, "reason": "DI-1 unknown team placeholder"}
                                for g in dropped["game_id"])
                P(f"  {t}: excluded {len(dropped)} unknown-team game(s) (DI-1)")
            if not len(games):
                continue
            view = store.view(t)
            req = make_requests(view, games)
            if not len(req):
                raise RuntimeError(f"empty request set at {t} for games {games['game_id'].tolist()} (not DI-1)")
            req["forecast_time"] = t
            req_log.append(req)
            for m in models:
                m.fit(view)
                f = m.predict(view, req, targets)
                f.validate()
                f.meta = {"accessed": sorted(view.accessed)}
                accessed[m.name] |= set(view.accessed)
                chunk_fc[m.name].append(f)
                fc_store[m.name].append((f.p_appear.astype(np.float32), {k: v.astype(np.float32) for k, v in f.cond.items()}))
            n_days += 1
            if n_days % 10 == 0:
                P(f"  {t} games={len(games)} req={len(req)} days={n_days} elapsed={time.time()-t0:.0f}s")
        if models and any(chunk_fc.values()):
            sc = score_forecasts(store, chunk_fc, targets, modes=("U", "A"), seed=0, tail_thresholds=TAIL)
            sc["season"] = S
            sc.to_parquet(os.path.join(out, f"scores_{S}_part{score_parts:03d}.parquet"), index=False)
            score_parts += 1
        c0 = c1 + pd.Timedelta(minutes=1)
    req_all = pd.concat(req_log, ignore_index=True)
    g = gfull.drop_duplicates("game_id", keep="last").set_index("game_id")
    req_all["phase"] = req_all["game_id"].map(g["phase"])
    req_all["season"] = S

    # coverage assertion: every P1-phase v1 game whose forecast time lies in the window, minus DI-1
    # rows, must have requests (reschedules: a game is matched by game_id, not by tip)
    gs = g1[(g1["season_id"] == f"nba-{S}") & g1["phase"].isin(PHASES)]
    gs, _ = drop_unknown_team_games(gs)
    ft = gs["scheduled_tip"] - LEAD
    expected = set(gs.loc[(ft >= start) & (ft <= end), "game_id"])
    got = set(req_all["game_id"])
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    P(f"season {S}: coverage expected={len(expected)} got={len(got)} missing={len(missing)} extra={len(extra)}")
    if missing:
        raise RuntimeError(f"coverage assertion failed: {len(missing)} game(s) never forecast, e.g. {missing[:5]}")

    tag = "_dry" if a.dry_run else ""
    req_all.to_parquet(os.path.join(out, f"requests_{S}{tag}.parquet"), index=False)
    if not a.dry_run:
        for name, lst in fc_store.items():
            arr = {"p_appear": np.concatenate([p for p, _ in lst])}
            for tg in targets:
                arr[f"cond_{tg}"] = np.concatenate([c[tg] for _, c in lst])
            np.savez_compressed(os.path.join(out, f"forecasts_{S}_{name}.npz"), **arr)
        P(f"season {S}: forecasts done days={n_days} requests={len(req_all)} elapsed={time.time()-t0:.0f}s; strata...")
        strata = compute_strata(store, req_all)
        strata["season"] = S
        strata.to_parquet(os.path.join(out, f"strata_{S}.parquet"), index=False)
    json.dump({"season": S, "days": n_days, "requests": int(len(req_all)), "score_parts": score_parts,
               "seconds": round(time.time() - t0), "accessed_tables": {k: sorted(v) for k, v in accessed.items()},
               "models": [m.name for m in models], "excluded_games": excluded,
               "game_v1_lag_days": lag_days, "chunk_days": chunk_days,
               "coverage": {"expected_games": len(expected), "forecast_games": len(got), "missing": missing, "extra": extra},
               "dry_run": a.dry_run},
              open(os.path.join(out, f"done_{S}{tag}.json"), "w"), indent=1)
    P(f"season {S}: DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
