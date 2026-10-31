"""P1 diagnostic mission (2026-10-30): population accounting for DI-2 (the 2004 step in requests per game).

Counts only (structural / data-quality class), plus the mean forecast p_appear per model against the
realised appearance share (a diagnostic on already-exposed P1 forecasts). Nothing here changes any
model, population or threshold; it reads the frozen adapter tables, the P1 requests/forecasts and the
corpus box-score table.

Writes results/p1_diag/population_accounting.json and .md.
"""
import json
import os
import sys

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import DEFAULT_DB, CorpusV102Adapter  # noqa: E402

P1 = os.path.join(ROOT, "results", "p1_ladder")
OUT = os.path.join(ROOT, "results", "p1_diag")
os.makedirs(OUT, exist_ok=True)
PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}
MODELS = ["L0_climatology", "L1_player_mean", "L2_recent_hl5", "L3_minutes_rate", "L4_hier_shrink"]
SEASONS = list(range(1998, 2021))
res = {}

# ---- A. corpus source: player_game_box rows per season --------------------------------------
con = duckdb.connect(DEFAULT_DB, read_only=True)
src = con.execute("""
    WITH b AS (
      SELECT season, canonical_game_id, team_id, canonical_player_id, did_not_play, TRY_CAST(minutes AS DOUBLE) AS minutes, TRY_CAST(secs_played AS DOUBLE) AS secs_played,
             coalesce(TRY_CAST(points AS DOUBLE),0)+coalesce(TRY_CAST(fgm AS DOUBLE),0)+coalesce(TRY_CAST(fga AS DOUBLE),0)+coalesce(TRY_CAST(fg3m AS DOUBLE),0)+coalesce(TRY_CAST(fg3a AS DOUBLE),0)+coalesce(TRY_CAST(ftm AS DOUBLE),0)+coalesce(TRY_CAST(fta AS DOUBLE),0)+coalesce(TRY_CAST(oreb AS DOUBLE),0)+coalesce(TRY_CAST(dreb AS DOUBLE),0)+coalesce(TRY_CAST(ast AS DOUBLE),0)+coalesce(TRY_CAST(stl AS DOUBLE),0)+coalesce(TRY_CAST(blk AS DOUBLE),0)+coalesce(TRY_CAST(tov AS DOUBLE),0)+coalesce(TRY_CAST(pf AS DOUBLE),0) AS stats
      FROM player_game_box WHERE league='nba' AND source='nba_stats' AND canonical_player_id IS NOT NULL AND team_id IS NOT NULL)
    SELECT season, count(*) AS rows, count(DISTINCT canonical_game_id) AS games,
           sum(CASE WHEN lower(cast(did_not_play AS varchar))='true' THEN 1 ELSE 0 END) AS dnp_flag_true,
           sum(CASE WHEN minutes IS NULL AND secs_played IS NULL THEN 1 ELSE 0 END) AS minutes_null,
           sum(CASE WHEN (minutes IS NULL AND secs_played IS NULL) AND stats=0 THEN 1 ELSE 0 END) AS minutes_null_zero_stats,
           sum(CASE WHEN coalesce(secs_played, minutes*60) > 0 THEN 1 ELSE 0 END) AS secs_positive,
           sum(CASE WHEN coalesce(secs_played, minutes*60) = 0 THEN 1 ELSE 0 END) AS secs_zero,
           count(*) * 1.0 / count(DISTINCT (canonical_game_id, team_id)) AS rows_per_team_game
    FROM b GROUP BY season ORDER BY season""").df()
con.close()
res["A_source_player_game_box_by_season"] = src.astype(object).to_dict("records")

# ---- B. adapter PLAYER_GAME status mix and ROSTER spells -----------------------------------
adapter = CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A")
tables = adapter.load_tables()
store, _ = build_store(adapter, strict=True)
pg = tables["PLAYER_GAME"]
g = store.full_table("GAME")
g1 = g[g["record_version"] == 1].drop_duplicates("game_id").set_index("game_id")
pg = pg.assign(season=pg["game_id"].map(g1["season_id"]).str.split("-").str[1].astype("Int64"),
               phase=pg["game_id"].map(g1["phase"]))
pgp = pg[pg["phase"].isin(PHASES)]
mix = pgp.groupby(["season", "status"]).size().unstack(fill_value=0)
tg = pgp.drop_duplicates(["game_id", "team_id"]).groupby("season").size()
B = {}
for s in mix.index:
    row = {k: int(v) for k, v in mix.loc[s].items()}
    row["team_games"] = int(tg.get(s, 0))
    row["rows_per_team_game"] = round(float(mix.loc[s].sum() / max(tg.get(s, 1), 1)), 2)
    row["played_per_team_game"] = round(float(mix.loc[s].get("played", 0) / max(tg.get(s, 1), 1)), 2)
    B[int(s)] = row
res["B_adapter_player_game_status_by_season"] = B

# ---- C. as-of roster size at mid-season dates ----------------------------------------------
seas = store.full_table("SEASON").set_index("season_id")
C = {}
for S in SEASONS:
    d = pd.Timestamp(f"{S}-01-15 12:00", tz="UTC")
    view = store.view(d)
    r = view.table("ROSTER")
    vt = r["valid_to"]
    live = r[(r["valid_from"] <= d) & (vt.isna() | (vt > d))]
    per_team = live.groupby("team_id")["player_id"].nunique()
    # composition: status of each rostered player's most recent box row before d
    h = view.table("PLAYER_GAME")
    h = h.assign(tip=h["game_id"].map(g1["scheduled_tip"]))
    last = h.sort_values(["player_id", "tip"]).drop_duplicates("player_id", keep="last").set_index("player_id")["status"]
    comp = live["player_id"].map(last).fillna("none").value_counts()
    C[S] = {"date": str(d.date()), "teams": int(len(per_team)), "mean_roster": round(float(per_team.mean()), 2),
            "min_roster": int(per_team.min()), "max_roster": int(per_team.max()),
            "last_box_status_of_rostered": {k: int(v) for k, v in comp.items()}}
res["C_asof_roster_mid_season"] = C

# ---- D/E. requests: outcome composition and forecast p_appear per season -------------------
outc = pg.set_index(["game_id", "player_id"])["status"]
D = {}
for S in SEASONS:
    f = os.path.join(P1, f"requests_{S}.parquet")
    if not os.path.exists(f):
        continue
    req = pd.read_parquet(f)
    st = pd.Series(list(zip(req["game_id"], req["player_id"]))).map(outc).fillna("no_box_row")
    n = len(req)
    games = req["game_id"].nunique()
    d = {"requests": int(n), "games": int(games), "req_per_game": round(n / games, 2),
         "outcome_mix": {k: round(float(v) / n, 4) for k, v in st.value_counts().items()},
         "appeared_share": round(float((st == "played").mean()), 4)}
    for m in MODELS:
        fz = os.path.join(P1, f"forecasts_{S}_{m}.npz")
        if os.path.exists(fz):
            pa = np.load(fz)["p_appear"]
            d[f"mean_p_appear_{m}"] = round(float(pa.mean()), 4) if len(pa) == n else f"len mismatch {len(pa)} vs {n}"
    D[S] = d
res["D_requests_outcome_mix_and_p_appear_by_season"] = D

# ---- F. reconciliation on representative game days before / after the step -----------------
def reconcile(day):
    d0 = pd.Timestamp(day, tz="UTC")
    gg = g1[(g1["scheduled_tip"] >= d0) & (g1["scheduled_tip"] < d0 + pd.Timedelta(days=7)) & g1["phase"].isin(PHASES)]
    if not len(gg):
        return {"day": day, "note": "no game in the following 7 days"}
    tip = gg["scheduled_tip"].min()
    gg = gg[gg["scheduled_tip"] == tip]
    t = tip - pd.Timedelta(minutes=60)
    view = store.view(t)
    r = view.table("ROSTER")
    vt = r["valid_to"]
    S = int(gg["season_id"].iloc[0].split("-")[1])
    req = pd.read_parquet(os.path.join(P1, f"requests_{S}.parquet"))
    req = req[req["game_id"].isin(gg.index)]
    rows = []
    for gid, gm in gg.iterrows():
        for side, tm in (("home", gm["home_team_id"]), ("away", gm["away_team_id"])):
            live = r[(r["team_id"] == tm) & (r["valid_from"] <= tip) & (vt.isna() | (vt > tip))]
            rostered = set(live["player_id"])
            rq = set(req.loc[(req["game_id"] == gid) & (req["team_id"] == tm), "player_id"])
            box = pg[(pg["game_id"] == gid) & (pg["team_id"] == tm)]
            bst = box.set_index("player_id")["status"]
            rows.append({"game_id": gid, "team": tm, "side": side, "rostered_asof": len(rostered), "requests": len(rq),
                         "rostered_not_requested": len(rostered - rq), "requested_not_rostered": len(rq - rostered),
                         "box_rows": int(len(box)), "box_played": int((bst == "played").sum()),
                         "box_dnp_coach": int((bst == "dnp_coach").sum()), "box_inactive": int((bst == "inactive").sum()),
                         "requested_played": int(sum(bst.get(p) == "played" for p in rq)),
                         "requested_dnp_or_inactive": int(sum(bst.get(p) in ("dnp_coach", "inactive") for p in rq)),
                         "requested_no_box_row": int(sum(p not in bst.index for p in rq)),
                         "box_player_not_requested": int(sum(p not in rq for p in bst.index))})
    df = pd.DataFrame(rows)
    tot = df.drop(columns=["game_id", "team", "side"]).sum().to_dict()
    return {"day": day, "season": S, "forecast_time": str(t), "games": int(len(gg)), "teams": int(len(df)),
            "totals": {k: int(v) for k, v in tot.items()}, "per_team": df.astype(object).to_dict("records")}

res["F_reconciliation"] = [reconcile(d) for d in ("2003-01-15", "2003-03-15", "2004-01-15", "2004-03-15", "2003-11-01", "2003-10-28")]

json.dump(res, open(os.path.join(OUT, "population_accounting.json"), "w"), indent=1, default=str)

# ---- markdown -------------------------------------------------------------------------------
L = ["# DI-2 population accounting (counts only)", ""]
L += ["## A. corpus player_game_box (nba_stats) per season", "",
      "| season | rows | games | dnp_flag_true | minutes_null | minutes_null & zero stats | secs>0 | secs=0 | rows/team-game |", "|---|---|---|---|---|---|---|---|---|"]
for r_ in res["A_source_player_game_box_by_season"]:
    L.append(f"| {r_['season']} | {r_['rows']} | {r_['games']} | {r_['dnp_flag_true']} | {r_['minutes_null']} | {r_['minutes_null_zero_stats']} | {r_['secs_positive']} | {r_['secs_zero']} | {float(r_['rows_per_team_game']):.2f} |")
L += ["", "## B. adapter PLAYER_GAME status per season (P1 phases)", "", "| season | played | dnp_coach | inactive | team-games | rows/team-game | played/team-game |", "|---|---|---|---|---|---|---|"]
for s, r_ in B.items():
    L.append(f"| {s} | {r_.get('played',0)} | {r_.get('dnp_coach',0)} | {r_.get('inactive',0)} | {r_['team_games']} | {r_['rows_per_team_game']} | {r_['played_per_team_game']} |")
L += ["", "## C. as-of roster size on Jan 15 (spells covering the date)", "", "| season | teams | mean | min | max | last box status of rostered players |", "|---|---|---|---|---|---|"]
for s, r_ in C.items():
    L.append(f"| {s} | {r_['teams']} | {r_['mean_roster']} | {r_['min_roster']} | {r_['max_roster']} | {r_['last_box_status_of_rostered']} |")
L += ["", "## D. requests: outcome mix and mean forecast p_appear", "", "| season | requests | req/game | played | dnp_coach | inactive | no box row | p_appear L0 | L1 | L2 | L3 | L4 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s, r_ in D.items():
    om = r_["outcome_mix"]
    L.append(f"| {s} | {r_['requests']} | {r_['req_per_game']} | {om.get('played',0):.3f} | {om.get('dnp_coach',0):.3f} | {om.get('inactive',0):.3f} | {om.get('no_box_row',0):.3f} | "
             + " | ".join(str(r_.get(f'mean_p_appear_{m}', 'n/a')) for m in MODELS) + " |")
L += ["", "## F. reconciliation on representative game days", ""]
for rc in res["F_reconciliation"]:
    if "totals" not in rc:
        L.append(f"- {rc}")
        continue
    L.append(f"- **{rc['day']}** (season {rc['season']}, forecast time {rc['forecast_time']}, {rc['games']} games, {rc['teams']} teams): " +
             ", ".join(f"{k}={v}" for k, v in rc["totals"].items()))
open(os.path.join(OUT, "population_accounting.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
