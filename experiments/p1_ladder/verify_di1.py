"""DI-1 verification (structural exclusion of unknown-team placeholder games).

Writes results/p1_ladder/di1_verification.json with:
  * source level: every nba_stats schedule row in corpus v1.0.2 whose home/away team is '0';
  * store level: rows meeting the frozen rule (either team id == 'nba:nba_stats:0') by season,
    phase and whether the row lies inside the season forecast window used by run_season.py;
  * per finished season: whether any request in requests_{S}.parquet involves such a game
    (must be 0 everywhere), and which driver version the season ran under.
The rule is structural (placeholder team identity => no valid roster); it never looks at a forecast
or a score. A season that ran under the pre-patch driver is provably identical to a post-patch run
iff its forecast population contains zero qualifying rows (the only code difference is the
exclusion branch and an empty-request guard that cannot fire when requests are non-empty).
"""
import datetime
import json
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "p1_ladder")
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import DEFAULT_DB, CorpusV102Adapter  # noqa: E402
from exclusions import UNKNOWN_TEAM, drop_unknown_team_games  # noqa: E402

PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}
res = {"rule": f"drop a GAME row iff home_team_id == '{UNKNOWN_TEAM}' or away_team_id == '{UNKNOWN_TEAM}'",
       "basis": "structural invalidity (placeholder team identity, no roster spell can exist); never forecast performance"}

con = duckdb.connect(DEFAULT_DB, read_only=True)
src = con.execute("""SELECT season, season_type, source_game_id, game_date, home_team_id, away_team_id,
    (SELECT count(*) FROM player_game_box p WHERE p.canonical_game_id = g.canonical_game_id) AS player_box_rows
    FROM games g WHERE league='nba' AND source='nba_stats' AND (home_team_id='0' OR away_team_id='0')
    ORDER BY season, game_date""").df()
con.close()
res["source_rows_team_0"] = src.astype(str).to_dict("records")
res["source_rows_team_0_count"] = int(len(src))

store, _ = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A"), strict=True)
g = store.full_table("GAME")
seas = store.full_table("SEASON").set_index("season_id")
g1 = g[g["record_version"] == 1].copy()
g1["season"] = g1["season_id"].str.split("-").str[1].astype(int)
_, bad = drop_unknown_team_games(g1)
rows = []
for _, r in bad.iterrows():
    s = seas.loc[r["season_id"]]
    start = pd.Timestamp(s["start_date"], tz="UTC")
    end = pd.Timestamp(s["end_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    t = r["scheduled_tip"] - pd.Timedelta(minutes=60)
    rows.append({"game_id": r["game_id"], "season": int(r["season"]), "phase": r["phase"], "tip": str(r["scheduled_tip"]),
                 "p1_phase": r["phase"] in PHASES, "forecast_time_in_season_window": bool(start <= t <= end),
                 "would_be_forecast": bool(r["phase"] in PHASES and start <= t <= end)})
res["store_rows_meeting_rule"] = rows
res["store_rows_meeting_rule_count"] = len(rows)
res["rows_that_would_enter_p1_forecast_population"] = [x for x in rows if x["would_be_forecast"]]

edit = os.path.getmtime(os.path.join(ROOT, "experiments", "p1_ladder", "exclusions.py"))
per = {}
for S in range(1998, 2021):
    f = os.path.join(OUT, f"done_{S}.json")
    if not os.path.exists(f):
        per[S] = {"status": "not finished"}
        continue
    d = json.load(open(f))
    start = os.path.getmtime(f) - d["seconds"]
    req = pd.read_parquet(os.path.join(OUT, f"requests_{S}.parquet"))
    gm = g1.set_index("game_id")
    hit = req[(req["team_id"] == UNKNOWN_TEAM) | req["game_id"].map(gm["home_team_id"]).eq(UNKNOWN_TEAM)
              | req["game_id"].map(gm["away_team_id"]).eq(UNKNOWN_TEAM)]
    qualifying_in_pop = [x for x in rows if x["season"] == S and x["would_be_forecast"]]
    per[S] = {"driver": "post-patch (exclusion rule executed)" if start > edit else "pre-patch",
              "process_start_local": datetime.datetime.fromtimestamp(start).isoformat(timespec="minutes"),
              "requests": int(len(req)), "requests_involving_unknown_team": int(len(hit)),
              "qualifying_rows_in_forecast_population": len(qualifying_in_pop),
              "excluded_games_logged": d.get("excluded_games", "n/a (pre-patch driver)"),
              "identical_under_both_drivers": bool(len(qualifying_in_pop) == 0 and len(req) > 0)}
res["per_season"] = per
res["all_finished_seasons_consistent"] = all(
    v.get("status") != "not finished" and (v["identical_under_both_drivers"] or v["driver"].startswith("post-patch")) and
    v["requests_involving_unknown_team"] == 0 for v in per.values())
res["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
json.dump(res, open(os.path.join(OUT, "di1_verification.json"), "w", encoding="utf-8"), indent=1, default=str)
print("source rows:", res["source_rows_team_0_count"], "store rows meeting rule:", len(rows),
      "would enter P1 population:", len(res["rows_that_would_enter_p1_forecast_population"]))
for S, v in per.items():
    print(S, v)
print("ALL FINISHED SEASONS CONSISTENT:", res["all_finished_seasons_consistent"])
