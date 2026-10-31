"""E-HIST segment parser: possession SEGMENTS (the frozen unit) from the corpus play-by-play, inside
each supplied nba_stats possession (M18: the supplied possessions are whole possessions with
offensive-rebound continuations folded in; hoopslab's own parser over the events was deferred to P2).

Unit (docs/prereg/01, E-HIST): a possession segment; an offensive rebound starts a new segment of the
same possession (start_type after_oreb). Terminal outcome, 8 classes (hoopslab.models.event):
    0 TO | 1 FT trip 0 made | 2 FT trip 1 made | 3 FT trip 2+ made | 4 2PA miss | 5 2PA make | 6 3PA miss | 7 3PA make
Rules fixed before any model was fit (documented decisions, not tuned):
  * a segment's terminal is the first of: a field-goal attempt by the offence, a free-throw trip by the
    offence that is not preceded by a field-goal attempt in the segment, a turnover by the offence;
  * and-one free throws after a made field goal stay in the same segment: the class is the make, the
    made free throws add to the segment's points;
  * a free-throw trip following a MISSED field-goal attempt in the same segment (shooting foul where the
    source still logged the miss) replaces the miss as the terminal (class by made free throws);
  * technical free throws add points but never define a terminal; flagrant / clear-path trips count as
    free-throw trips;
  * an offensive rebound (team or player, after a missed attempt or a missed final free throw) closes the
    current segment and the next offensive action opens a new segment with start_type after_oreb; a
    second attempt by the offence without any rebound row in between (tip-ins) is treated the same way;
  * a possession whose last segment has no terminal (period end, dead-ball end, source gap) loses that
    segment; it is counted, never guessed.
Start type of the first segment from the source start_reason: OffMadeShot -> after_make, OffMissedShot ->
after_dreb, OffLiveBallTurnover -> after_live_turnover, OffDeadball / OffTimeout / other -> after_dead_ball
(the baseline category of the frozen design). Margin and elapsed time at the segment's first event.
Lineups are the supplied possession lineups (held fixed across the possession's segments).

Validation gate per season (written next to the parquet): share of possessions whose parsed points equal
the supplied points and whose parsed field-goal attempts equal src_fg2a + src_fg3a.

Only seasons <= 2020 can be parsed (burn-in 1997-1999 + development 2000-2020); the lockbox and
holdout are refused here.
"""
import argparse
import json
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.adapters.corpus_v102 import DEFAULT_DB  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_hist")
START_MAP = {"OffMadeShot": "after_make", "OffMissedShot": "after_dreb", "OffLiveBallTurnover": "after_live_turnover"}
FT_RE = re.compile(r"(\d) of (\d)")
PERIOD_S, OT_S = 720.0, 300.0


def clock_s(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v)
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return np.nan


def elapsed(period, clock):
    p = int(period)
    before = min(p - 1, 4) * PERIOD_S + max(p - 5, 0) * OT_S
    plen = PERIOD_S if p <= 4 else OT_S
    return before + (plen - clock)


def parse_season(season: int, con) -> tuple[pd.DataFrame, dict]:
    if season > 2020:
        raise PermissionError("segments are built for burn-in/development seasons only (<= 2020)")
    poss = con.execute(f"""
        SELECT source_game_id, possession_number, offense_team_id, defense_team_id, start_sequence, end_sequence, points,
               is_turnover, start_reason, players_on_floor, defense_players_on_floor, src_fg2a, src_fg3a, src_fta, src_ftm, src_oreb,
               ambiguous_boundary
        FROM possessions WHERE league='nba' AND source='nba_stats' AND season={season} AND coalesce(count_as_possession, TRUE)
          AND offense_team_id IS NOT NULL AND start_sequence IS NOT NULL AND end_sequence IS NOT NULL
        ORDER BY source_game_id, possession_number""").df()
    ev = con.execute(f"""
        SELECT source_game_id, sequence, period, game_clock, team_id, player_id, event_family, event_subtype, description,
               home_score, away_score, shot_value
        FROM play_by_play WHERE league='nba' AND source='nba_stats' AND season={season}
        ORDER BY source_game_id, sequence""").df()
    games = con.execute(f"""
        SELECT source_game_id, game_date, season_type, home_team_id, away_team_id FROM games
        WHERE league='nba' AND source='nba_stats' AND season={season}""").df().drop_duplicates("source_game_id").set_index("source_game_id")
    ev["home_score"] = pd.to_numeric(ev["home_score"], errors="coerce")
    ev["away_score"] = pd.to_numeric(ev["away_score"], errors="coerce")
    ev[["home_score", "away_score"]] = ev.groupby("source_game_id")[["home_score", "away_score"]].ffill().fillna(0)
    ev["clock_s"] = ev["game_clock"].map(clock_s)
    ev["team"] = ev["team_id"].astype("string").fillna("")
    ev["pid"] = ev["player_id"].astype("string").fillna("")
    ev["fam"] = ev["event_family"].astype("string").fillna("")
    ev["sub"] = ev["event_subtype"].astype("string").fillna("")
    ev["desc"] = ev["description"].astype("string").fillna("")
    ev["sv"] = pd.to_numeric(ev["shot_value"], errors="coerce").fillna(0).astype(int)
    rows, stats = [], {"possessions": int(len(poss)), "segments": 0, "dropped_no_terminal": 0, "points_match": 0, "fga_match": 0,
                       "games_missing": 0, "possessions_without_events": 0}
    by_game = {g: d for g, d in ev.groupby("source_game_id", sort=False)}
    for gid, P in poss.groupby("source_game_id", sort=False):
        if gid not in games.index or gid not in by_game:
            stats["games_missing"] += 1
            continue
        g = games.loc[gid]
        E = by_game[gid]
        seq = E["sequence"].to_numpy()
        fam, team, pid, sub, desc = E["fam"].to_numpy(), E["team"].to_numpy(), E["pid"].to_numpy(), E["sub"].to_numpy(), E["desc"].to_numpy()
        hs, as_, per, clk, sv = E["home_score"].to_numpy(), E["away_score"].to_numpy(), E["period"].to_numpy(), E["clock_s"].to_numpy(), E["sv"].to_numpy()
        home = str(g["home_team_id"])
        for r in P.itertuples(index=False):
            O = str(r.offense_team_id)
            i0, i1 = np.searchsorted(seq, r.start_sequence, side="left"), np.searchsorted(seq, r.end_sequence, side="right")
            if i1 <= i0:
                stats["possessions_without_events"] += 1
                continue
            off_home = O == home
            segs = []
            cur = None          # dict for the open segment
            next_start = START_MAP.get(str(r.start_reason), "after_dead_ball")
            pending_miss = False
            k = 0

            def open_seg(i):
                nonlocal cur, k
                m_before = (hs[i - 1] - as_[i - 1]) if i > 0 else 0.0
                cur = {"k": k, "start_type": next_start, "i": i, "event_code": -1, "made": 0, "ftm": 0, "fta": 0, "points": 0,
                       "margin": m_before if off_home else -m_before,
                       "elapsed": elapsed(per[i], clk[i]) if not np.isnan(clk[i]) else np.nan, "closed": False}
                k += 1

            def close_seg():
                nonlocal cur, next_start
                if cur is not None:
                    if cur["event_code"] >= 0:
                        segs.append(cur)
                    else:
                        stats["dropped_no_terminal"] += 1
                cur = None
                next_start = "after_oreb"

            for i in range(i0, i1):
                f = fam[i]
                if f in ("shot_made", "shot_missed"):
                    if team[i] != O:
                        continue
                    if cur is not None and cur["event_code"] >= 0:
                        close_seg()                      # second attempt without a rebound row: tip-in / gap
                    if cur is None:
                        open_seg(i)
                    made = f == "shot_made"
                    cur["event_code"] = 3 if sv[i] == 3 else 2
                    cur["made"] = int(made)
                    cur["points"] += (3 if sv[i] == 3 else 2) if made else 0
                    pending_miss = not made
                elif f == "free_throw":
                    if team[i] != O:
                        continue
                    if "Technical" in sub[i]:
                        if cur is not None:
                            cur["points"] += 0 if desc[i].startswith("MISS") else 1
                        continue
                    if cur is None:
                        open_seg(i)
                    made = not desc[i].startswith("MISS")
                    m = FT_RE.search(desc[i])
                    n_of, of = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
                    if cur["event_code"] in (-1, 1) or (cur["event_code"] >= 2 and cur["made"] == 0):
                        if cur["event_code"] != 1:       # trip starts (also replaces a logged miss)
                            cur["event_code"], cur["made"], cur["ftm"], cur["fta"] = 1, 0, 0, 0
                        cur["fta"] += 1
                        cur["ftm"] += int(made)
                    cur["points"] += int(made)
                    if n_of == of:
                        pending_miss = not made
                        if any(k in sub[i] for k in ("Flagrant", "Clear Path", "Away From Play")) and made:
                            close_seg(); next_start = "after_dead_ball"   # the fouled team retains possession
                elif f == "rebound":
                    # a rebound row acts only when a miss is pending (missed attempt or missed FINAL free throw).
                    # nba_stats also logs a dead-ball team rebound after a missed non-final free throw and logs the
                    # rebound of a put-back AFTER the made tip shot; both must not split the segment (rule fixed after the
                    # timing-month controls exposed the split, see docs/p2/01).
                    rteam = team[i] if team[i] not in ("", "0") else pid[i]
                    if cur is not None and cur["event_code"] >= 0 and pending_miss:
                        if rteam == O:
                            close_seg()                  # offensive rebound: next action opens after_oreb
                        else:
                            close_seg()
                            next_start = "possession_over"
                        pending_miss = False
                elif f == "turnover":
                    if team[i] != O:
                        continue
                    if cur is not None and cur["event_code"] >= 0:
                        close_seg()
                    if cur is None:
                        open_seg(i)
                    cur["event_code"] = 0
                    cur["points"] += 0
                    close_seg()
                    next_start = "possession_over"
            close_seg()
            fga = sum(1 for s in segs if s["event_code"] >= 2)
            pts = sum(s["points"] for s in segs)
            stats["points_match"] += int(pts == int(r.points if r.points == r.points else 0))
            stats["fga_match"] += int(fga == int((r.src_fg2a or 0) + (r.src_fg3a or 0)))
            off_lu = tuple(f"nba:{x}" for x in list(r.players_on_floor)) if r.players_on_floor is not None else None
            def_lu = tuple(f"nba:{x}" for x in list(r.defense_players_on_floor)) if r.defense_players_on_floor is not None else None
            D = str(r.defense_team_id) if r.defense_team_id is not None and str(r.defense_team_id) != "None" else (str(g["away_team_id"]) if off_home else home)
            for s in segs:
                if s["start_type"] == "possession_over" or np.isnan(s["elapsed"]) or off_lu is None or def_lu is None or len(off_lu) != 5 or len(def_lu) != 5:
                    stats["dropped_no_terminal"] += 1
                    continue
                rows.append((f"nba:nba_stats:{gid}", int(r.possession_number) * 10 + s["k"], s["k"] + 1, f"nba:nba_stats:{O}", f"nba:nba_stats:{D}",
                             off_home, float(s["elapsed"]), s["start_type"], int(s["event_code"]), int(s["made"]), int(min(s["ftm"], 2)),
                             int(s["points"]), int(s["margin"]), off_lu, def_lu, int(per[s["i"]]), str(g["game_date"])[:10], str(g["season_type"])))
            stats["segments"] += len(segs)
    cols = ["game_id", "possession_seq", "segment", "off_team_id", "def_team_id", "off_is_home", "start_elapsed_s", "start_type", "event_code",
            "made", "ftm", "points", "off_margin_start", "off_lineup", "def_lineup", "period", "game_date", "season_type"]
    P = pd.DataFrame(rows, columns=cols)
    P["season"] = season
    P["month"] = P["game_date"].str[:7]
    stats["segments_kept"] = int(len(P))
    stats["points_match_rate"] = round(stats["points_match"] / max(stats["possessions"], 1), 4)
    stats["fga_match_rate"] = round(stats["fga_match"] / max(stats["possessions"], 1), 4)
    stats["class_shares"] = {int(k): round(float(v), 4) for k, v in
                             (P.assign(c=np.select([P.event_code == 0, P.event_code == 1, P.event_code == 2, P.event_code == 3],
                                                   [0, 1 + P.ftm, 4 + P.made, 6 + P.made]))["c"].value_counts(normalize=True).sort_index()).items()}
    stats["start_type_shares"] = {k: round(float(v), 4) for k, v in P["start_type"].value_counts(normalize=True).items()}
    return P, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    for S in a.seasons:
        t0 = time.time()
        P, st = parse_season(S, con)
        P.to_parquet(os.path.join(OUT, f"segments_{S}.parquet"), index=False)
        st["seconds"] = round(time.time() - t0)
        json.dump(st, open(os.path.join(OUT, f"segments_{S}_validation.json"), "w"), indent=1)
        print(S, json.dumps({k: v for k, v in st.items() if k not in ("class_shares", "start_type_shares")}), flush=True)
        print("   classes", st["class_shares"], "starts", st["start_type_shares"], flush=True)


if __name__ == "__main__":
    main()
