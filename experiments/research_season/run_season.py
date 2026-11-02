"""R1 SEASON-CARRYOVER: prequential forecasts of the 44 frozen grid points (hoopslab.models.carryover) for ONE
NBA season (nba_stats, level A), mode A only, with the as-of context columns the frozen spec uses for selection
strata (research_season/01_carryover_spec.md, C25).

The forecast protocol is P1's (experiments/p1_ladder/run_season.py): rolling origin, chunked enumeration by
the GAME v1 publication lag, P1 phases only, DI-1 exclusion, T-60 cutoff, as-of roster requests, coverage
assertion. Scores are compact: per (grid point, target, unit) CRPS, log score, randomised PIT, outcome and
forecast mean (float32), one parquet per season. Burn-in seasons 1998 and 1999 are forecast for selection
only (P1 convention) and never reported.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "p1_ladder"))
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter  # noqa: E402
from hoopslab.backtest.runner import BacktestSpec, LockboxError, forecast_points, make_requests, outcomes_table  # noqa: E402
from hoopslab.backtest.strata import compute_strata  # noqa: E402
from hoopslab.metrics import scoring as sc  # noqa: E402
from hoopslab.models.carryover import GRID_IDS, TARGETS, SeasonAwareMean, season_ordinal  # noqa: E402
from exclusions import drop_unknown_team_games  # noqa: E402

OUT = os.path.join(ROOT, "results", "research_season")
PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}
LEAD = pd.Timedelta(minutes=60)
SEL_ONLY = {1998, 1999}


def score_compact(forecasts: dict, outc: pd.DataFrame, rng) -> pd.DataFrame:
    """Mode A rows (appeared units) for every grid point: crps, log, pit, y, mean."""
    rows = []
    for gid, f in forecasts.items():
        keys = f.keys.merge(outc, on=["game_id", "player_id"], how="left", suffixes=("", "_o"))
        appeared = keys["appeared"].fillna(False).to_numpy().astype(bool)
        if not appeared.any():
            continue
        for t in TARGETS:
            P = f.cond[t][appeared]
            y = keys[t].fillna(0).to_numpy().astype(int)[appeared]
            rows.append(pd.DataFrame({"model": gid, "target": t, "game_id": keys["game_id"].to_numpy()[appeared],
                                      "player_id": keys["player_id"].to_numpy()[appeared], "y": y.astype(np.int16),
                                      "crps": sc.crps_pmf(P, y).astype(np.float32), "log": sc.log_score(P, y).astype(np.float32),
                                      "pit": sc.randomized_pit(P, y, rng).astype(np.float32), "mean": sc.pmf_mean(P).astype(np.float32)}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-days", type=int, default=None, help="smoke test: stop after this many forecast days (never a result)")
    a = ap.parse_args()
    S, out = a.season, a.out
    if S > 2020:
        raise PermissionError("development seasons only (<= 2020)")
    os.makedirs(out, exist_ok=True)
    pre = json.load(open(os.path.join(out, "preflight.json")))
    assert pre["all_passed"], "preflight did not pass"
    log = open(os.path.join(out, f"log_{S}.txt"), "a")

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
    chunk_days = int(pd.Timedelta(lags[0]).days) - 1
    assert chunk_days >= 1

    def win(s):
        r = seas.loc[f"nba-{s}"]
        return (pd.Timestamp(r["start_date"], tz="UTC"), pd.Timestamp(r["end_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=59))

    lockbox = [win(s) for s in (2021, 2022)]
    start, end = win(S)
    outc = outcomes_table(store)
    rng = np.random.default_rng(0)
    model = SeasonAwareMean()
    req_log, ctx_log, excluded = [], [], []
    n_days, parts = 0, 0
    game_season = gfull.drop_duplicates("game_id", keep="last").set_index("game_id")["season_id"]
    # as-of team game index: completed (final, v2) games of the team in this season known before t
    gv2 = gfull[gfull["record_version"] == 2][["game_id", "home_team_id", "away_team_id", "season_id", "knowledge_time", "scheduled_tip"]]
    c0 = start
    while c0 <= end:
        c1 = min(c0 + pd.Timedelta(days=chunk_days - 1, hours=23, minutes=59), end)
        spec = BacktestSpec(start=c0, end=c1, targets=TARGETS, lockbox=lockbox)
        chunk_scores = []
        for t, games in forecast_points(store, spec):
            if spec.in_lockbox(t):
                raise LockboxError(f"forecast time {t} is inside a sealed lockbox window")
            games = games[games["phase"].isin(PHASES)]
            games, dropped = drop_unknown_team_games(games)
            if len(dropped):
                excluded.extend({"forecast_time": str(t), "game_id": g, "reason": "DI-1 unknown team placeholder"} for g in dropped["game_id"])
            if not len(games):
                continue
            view = store.view(t)
            req = make_requests(view, games)
            if not len(req):
                raise RuntimeError(f"empty request set at {t}")
            req["forecast_time"] = t
            cur = season_ordinal(game_season[games["game_id"].iloc[0]])
            tip = req["tip"].iloc[0]
            model.fit(view, cur, req, tip)
            fc = model.predict(view, req, TARGETS)
            for f in fc.values():
                f.validate()
            chunk_scores.append(score_compact(fc, outc, rng))
            # context columns per request (as-of)
            ctx = model.ctx.reset_index()
            done = gv2[(gv2["knowledge_time"] <= t) & (gv2["season_id"] == f"nba-{cur}")]
            tg = pd.concat([done["home_team_id"], done["away_team_id"]]).value_counts()
            ctx["team_game_index"] = ctx["team_id"].map(tg).fillna(0).astype(int) + 1   # tonight's game number for the team
            ctx["forecast_time"] = t
            ctx = req[["forecast_time", "game_id", "player_id", "team_id"]].merge(ctx, on=["forecast_time", "player_id", "team_id"], how="left")
            ctx_log.append(ctx)
            req_log.append(req)
            n_days += 1
            if n_days % 10 == 0:
                P(f"  {t} games={len(games)} req={len(req)} days={n_days} elapsed={time.time()-t0:.0f}s")
            if a.max_days and n_days >= a.max_days:
                break
        if chunk_scores:
            scp = pd.concat(chunk_scores, ignore_index=True)
            scp["model"] = scp["model"].astype(pd.CategoricalDtype(GRID_IDS))
            scp["target"] = scp["target"].astype(pd.CategoricalDtype(TARGETS))
            scp.to_parquet(os.path.join(out, f"scores_{S}_part{parts:03d}.parquet"), index=False)
            parts += 1
        if a.max_days and n_days >= a.max_days:
            break
        c0 = c1 + pd.Timedelta(minutes=1)
    req_all = pd.concat(req_log, ignore_index=True)
    ctx_all = pd.concat(ctx_log, ignore_index=True)
    g = gfull.drop_duplicates("game_id", keep="last").set_index("game_id")
    req_all["phase"] = req_all["game_id"].map(g["phase"])
    req_all["season"] = S
    ctx_all["season"] = S
    gs = g1[(g1["season_id"] == f"nba-{S}") & g1["phase"].isin(PHASES)]
    gs, _ = drop_unknown_team_games(gs)
    ft = gs["scheduled_tip"] - LEAD
    expected = set(gs.loc[(ft >= start) & (ft <= end), "game_id"])
    got = set(req_all["game_id"])
    missing = sorted(expected - got)
    P(f"season {S}: coverage expected={len(expected)} got={len(got)} missing={len(missing)}")
    if missing and not a.max_days:
        raise RuntimeError(f"coverage assertion failed: {len(missing)} game(s) never forecast, e.g. {missing[:5]}")
    tag = "_smoke" if a.max_days else ""
    req_all.to_parquet(os.path.join(out, f"requests_{S}{tag}.parquet"), index=False)
    ctx_all.to_parquet(os.path.join(out, f"context_{S}{tag}.parquet"), index=False)
    if not a.max_days:
        strata = compute_strata(store, req_all)
        strata["season"] = S
        strata.to_parquet(os.path.join(out, f"strata_{S}.parquet"), index=False)
    json.dump({"season": S, "selection_only": S in SEL_ONLY, "days": n_days, "requests": int(len(req_all)), "score_parts": parts,
               "seconds": round(time.time() - t0), "grid": GRID_IDS, "targets": TARGETS, "excluded_games": excluded,
               "coverage": {"expected_games": len(expected), "forecast_games": len(got), "missing": missing}, "smoke": bool(a.max_days)},
              open(os.path.join(out, f"done_{S}{tag}.json"), "w"), indent=1)
    P(f"season {S}: DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
