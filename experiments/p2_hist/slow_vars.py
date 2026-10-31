"""Pre-registered slow / game-level variables for the H3b attribution step (docs/prereg/01 E-HIST: "add the
slow variables one at a time: stint length / fatigue proxy, fouls, bonus, timeouts remaining, lineup-change
indicator"). All are as-of the segment's first event (nothing from the segment itself or later).

Per segment (keyed by game_id, possession_seq):
    def_fouls_period   team fouls charged to the DEFENCE in the current period before the segment (personal, shooting,
                       loose-ball, clear-path, flagrant, away-from-play; not offensive, not technical)
    off_fouls_period   same for the offence
    def_in_bonus       def_fouls_period >= 5 (>= 4 in overtime): the offence shoots free throws on the next non-shooting foul
    off_timeouts_used  timeouts the offence has used so far in the game; def_timeouts_used likewise
    lineup_change      1 if the offence lineup differs from the offence's previous segment lineup in this game (0 at its first)
    secs_since_prev    game seconds since the offence's previous segment started (NaN at its first; stored as -1)
    stint_len_s        seconds the current offence lineup has been on the floor (since its last lineup change; proxy for fatigue)
Writes results/p2_hist/slow_{S}.parquet."""
import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.adapters.corpus_v102 import DEFAULT_DB  # noqa: E402
from segments import OUT, clock_s, elapsed  # noqa: E402

TEAM_FOUL_SUBTYPES = ("Personal", "Shooting", "Loose Ball", "Clear Path", "Flagrant", "Away From Play", "Personal Block", "Shooting Block",
                      "Personal Take", "Inbound", "Double Personal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    a = ap.parse_args()
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    for S in a.seasons:
        if S > 2020:
            raise PermissionError("<= 2020 only")
        seg = pd.read_parquet(os.path.join(OUT, f"segments_{S}.parquet"))
        ev = con.execute(f"""SELECT source_game_id, sequence, period, game_clock, team_id, player_id, event_family, event_subtype
                             FROM play_by_play WHERE league='nba' AND source='nba_stats' AND season={S}
                               AND event_family IN ('foul','timeout') ORDER BY source_game_id, sequence""").df()
        ev["clock_s"] = ev["game_clock"].map(clock_s)
        ev["elapsed"] = [elapsed(p, c) if not np.isnan(c) else np.nan for p, c in zip(ev["period"], ev["clock_s"])]
        ev["team"] = "nba:nba_stats:" + ev["team_id"].astype("string").fillna("")
        ev["sub"] = ev["event_subtype"].astype("string").fillna("")
        is_tf = (ev["event_family"] == "foul") & ev["sub"].map(lambda s: any(s.startswith(k) for k in TEAM_FOUL_SUBTYPES))
        fouls = ev[is_tf & ev["elapsed"].notna()].copy()
        fouls["gid"] = "nba:nba_stats:" + fouls["source_game_id"].astype(str)
        tos = ev[(ev["event_family"] == "timeout") & ev["elapsed"].notna()].copy()
        tos["gid"] = "nba:nba_stats:" + tos["source_game_id"].astype(str)
        seg = seg.sort_values(["game_id", "start_elapsed_s", "possession_seq"], kind="stable").reset_index(drop=True)
        out = pd.DataFrame({"game_id": seg["game_id"], "possession_seq": seg["possession_seq"]})
        # fouls in the current period before the segment start, per team
        def count_before(events, team_col):
            res = np.zeros(len(seg), int)
            for gid, d in seg.groupby("game_id", sort=False):
                f = events[events["gid"] == gid]
                if not len(f):
                    continue
                for team, dd in d.groupby(team_col, sort=False):
                    ft = f[f["team"] == team]
                    for per, ddd in dd.groupby("period", sort=False):
                        fp = np.sort(ft[ft["period"] == per]["elapsed"].to_numpy().copy())
                        res[ddd.index] = np.searchsorted(fp, ddd["start_elapsed_s"].to_numpy(), side="left")
            return res
        out["def_fouls_period"] = count_before(fouls, "def_team_id")
        out["off_fouls_period"] = count_before(fouls, "off_team_id")
        out["def_in_bonus"] = (out["def_fouls_period"] >= np.where(seg["period"] > 4, 4, 5)).astype(int)
        def timeouts_before(team_col):
            res = np.zeros(len(seg), int)
            for gid, d in seg.groupby("game_id", sort=False):
                t = tos[tos["gid"] == gid]
                for team, dd in d.groupby(team_col, sort=False):
                    tt = np.sort(t[t["team"] == team]["elapsed"].to_numpy())
                    res[dd.index] = np.searchsorted(tt, dd["start_elapsed_s"].to_numpy(), side="left")
            return res
        out["off_timeouts_used"] = timeouts_before("off_team_id")
        out["def_timeouts_used"] = timeouts_before("def_team_id")
        g = seg.groupby(["game_id", "off_team_id"], sort=False)
        prev_lu = g["off_lineup"].shift(1)
        same = [None if p is None or (isinstance(p, float)) else tuple(p) == tuple(c) for p, c in zip(prev_lu, seg["off_lineup"])]
        out["lineup_change"] = [0 if s is None else int(not s) for s in same]
        prev_t = g["start_elapsed_s"].shift(1)
        out["secs_since_prev"] = (seg["start_elapsed_s"] - prev_t).fillna(-1).to_numpy()
        # stint length: time since the offence lineup last changed (within the game)
        stint_start = np.zeros(len(seg))
        for (gid, team), d in g:
            last = d["start_elapsed_s"].iloc[0]
            prev = None
            for i, (lu, t0) in enumerate(zip(d["off_lineup"], d["start_elapsed_s"])):
                tl = tuple(lu)
                if prev is not None and tl != prev:
                    last = t0
                stint_start[d.index[i]] = last
                prev = tl
        out["stint_len_s"] = seg["start_elapsed_s"].to_numpy() - stint_start
        out.to_parquet(os.path.join(OUT, f"slow_{S}.parquet"), index=False)
        print(S, len(out), "bonus share", round(out["def_in_bonus"].mean(), 3), "lineup_change", round(out["lineup_change"].mean(), 3),
              "mean def fouls", round(out["def_fouls_period"].mean(), 2), flush=True)


if __name__ == "__main__":
    main()
