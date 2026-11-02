"""Onboarding audit, exposure classes S1 (structural) and S2 (data quality) only.

Reads the researcher-facing release (BASKETBALL_CORPUS_v1.0.2 isolated package) READ-ONLY
and writes structural facts to results/onboarding/. Nothing here estimates any quantity a
hypothesis is about: null shares, coverage, key uniqueness, internal-consistency checks.

    python experiments/onboarding/audit_corpus_s1s2.py [--db PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import duckdb
import pandas as pd

DEFAULT_DB = (r"C:\Users\Tray\Documents\LLM AI Models\Kimi\bball_corpus\release"
              r"\BASKETBALL_CORPUS_v1.0.2\researcher_isolated\researcher_isolated_v1.0.2.duckdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "onboarding")

# contract-critical fields per corpus table: null share per (league, source, season)
FIELDS = {
    "games": ["game_date_time_utc", "game_date", "status", "neutral_site", "venue_id", "home_team_id",
              "away_team_id", "season_type", "game_type", "overtime_count"],
    "player_game_box": ["canonical_player_id", "player_id", "team_id", "starter", "did_not_play",
                        "minutes", "secs_played", "points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                        "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus", "position"],
    "team_game_box": ["canonical_team_id", "is_home", "final_score", "fgm", "fga", "fg3m", "fg3a", "ftm",
                      "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "technical_fouls"],
    "play_by_play": ["period", "game_clock", "game_clock_seconds_remaining", "shot_clock", "wall_clock",
                     "team_id", "player_id", "secondary_player_id", "event_type", "description",
                     "home_score", "away_score", "x", "y", "shot_distance", "shot_value", "made",
                     "is_assisted", "is_turnover", "is_foul", "is_rebound", "is_substitution",
                     "home_team_id", "away_team_id", "event_family"],
    "possessions": ["offense_team_id", "defense_team_id", "canonical_offense_team_id", "period",
                    "start_sequence", "end_sequence", "start_clock", "end_clock", "points",
                    "players_on_floor", "defense_players_on_floor", "start_reason", "end_reason",
                    "possession_boundary_reason", "ambiguous_boundary", "points_reconciled",
                    "offense_team_confidence", "is_turnover", "n_shots", "count_as_possession",
                    "offensive_rebound_continuation", "is_second_chance"],
    "substitutions": ["team_id", "player_in_id", "player_out_id", "canonical_player_in_id",
                      "canonical_player_out_id", "period", "game_clock", "sequence", "parse_confidence",
                      "parse_method"],
    "shots": ["x", "y", "shot_distance", "shot_value", "made", "shot_type", "assisted",
              "assisting_player_id", "blocked", "canonical_player_id", "canonical_team_id", "period",
              "game_clock"],
    "rosters": ["canonical_player_id", "canonical_team_id", "team_id", "player_id", "roster_grain",
                "starter", "status", "position", "jersey", "canonical_game_id"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    con.execute("SET memory_limit='6GB'")
    con.execute("PRAGMA threads=4")
    t0 = time.time()

    # ------------------------------------------------------------ 1. null shares
    frames = []
    for tbl, cols in FIELDS.items():
        present = {r[0] for r in con.execute(f"DESCRIBE {tbl}").fetchall()}
        cols = [c for c in cols if c in present]
        sel = ", ".join(f"count({c}) AS \"{c}\"" for c in cols)
        df = con.execute(f"SELECT league, source, season, count(*) AS n_rows, {sel} FROM {tbl} "
                         f"GROUP BY 1,2,3").df()
        long = df.melt(id_vars=["league", "source", "season", "n_rows"], var_name="field",
                       value_name="n_filled")
        long["pct_filled"] = (100.0 * long["n_filled"] / long["n_rows"]).round(2)
        long.insert(0, "table", tbl)
        frames.append(long)
        print(f"  null shares: {tbl:16} {len(df):4} partitions  {time.time()-t0:6.0f}s", flush=True)
    cov = pd.concat(frames, ignore_index=True)
    cov.to_csv(os.path.join(OUT, "field_coverage_by_league_source_season.csv"), index=False)

    # ------------------------------------------------------------ 2. key uniqueness (S1)
    keys = {
        "games": ["league", "source", "source_game_id"],
        "player_game_box": ["league", "source", "source_game_id", "player_id"],
        "team_game_box": ["league", "source", "source_game_id", "team_id"],
        "play_by_play": ["league", "source", "canonical_game_id", "sequence"],
        "possessions": ["league", "source", "canonical_game_id", "possession_number"],
    }
    uniq = []
    for tbl, k in keys.items():
        r = con.execute(f"SELECT count(*) n, count(DISTINCT ({', '.join(k)})) d FROM {tbl}").fetchone()
        uniq.append(dict(table=tbl, key=k, rows=r[0], distinct=r[1], duplicates=r[0] - r[1]))
        print(f"  key uniqueness: {tbl:16} dup={r[0]-r[1]:,}  {time.time()-t0:6.0f}s", flush=True)

    # ------------------------------------------------------------ 3. S2 consistency checks
    s2 = {}
    # 3a. PLAYER_GAME must contain non-appearing rostered players (did_not_play / 0 seconds)
    s2["dnp_share"] = con.execute("""
        SELECT league, source, season, count(*) n,
               count(*) FILTER (WHERE did_not_play IN ('true','True','1')) dnp_flagged,
               count(*) FILTER (WHERE TRY_CAST(secs_played AS DOUBLE) = 0
                                   OR TRY_CAST(minutes_decimal AS DOUBLE) = 0) zero_minutes,
               count(*) FILTER (WHERE secs_played IS NULL AND minutes IS NULL AND minutes_decimal IS NULL) minutes_null
        FROM player_game_box GROUP BY 1,2,3 ORDER BY 1,2,3""").df().to_dict("records")
    # 3b. sum of player points = team points; sum of player seconds vs expected (quality_flags carries both)
    s2["quality_flag_checks"] = con.execute("""
        SELECT check_name, severity, league, count(*) n, sum(n_affected) n_affected, sum(n_total) n_total
        FROM quality_flags WHERE check_name IS NOT NULL GROUP BY 1,2,3 ORDER BY 1,2,3""").df().to_dict("records")
    s2["pbp_vs_official_score"] = con.execute("""
        SELECT league, pbp_source AS source, season, count(*) games,
               count(*) FILTER (WHERE "match") AS n_matched, round(100.0*count(*) FILTER (WHERE "match")/count(*),2) pct
        FROM quality_flags WHERE "match" IS NOT NULL GROUP BY 1,2,3 ORDER BY 1,2,3""").df().to_dict("records")
    s2["player_points_vs_team"] = con.execute("""
        SELECT league, team_source AS source, count(*) team_games,
               count(*) FILTER (WHERE diff = 0) exact, round(100.0*count(*) FILTER (WHERE diff = 0)/count(*),2) pct_exact
        FROM quality_flags WHERE diff IS NOT NULL AND team_points IS NOT NULL GROUP BY 1,2 ORDER BY 1,2""").df().to_dict("records")
    s2["player_minutes_vs_expected"] = con.execute("""
        SELECT league, source, count(*) team_games,
               count(*) FILTER (WHERE abs(sum_player_minutes - expected_team_minutes) <= 1) within_1min,
               round(100.0*count(*) FILTER (WHERE abs(sum_player_minutes - expected_team_minutes) <= 1)/count(*),2) pct
        FROM quality_flags WHERE sum_player_minutes IS NOT NULL AND expected_team_minutes IS NOT NULL
        GROUP BY 1,2 ORDER BY 1,2""").df().to_dict("records")
    # 3c. possessions per team-game vs box-score estimate (FGA - OREB + TOV + 0.44 FTA), derived parser
    s2["possessions_vs_box_estimate"] = con.execute("""
        WITH p AS (
          SELECT league, source, season, canonical_game_id, offense_team_id,
                 count(*) FILTER (WHERE count_as_possession) n_poss
          FROM possessions GROUP BY 1,2,3,4,5),
        b AS (
          SELECT league, source, season, canonical_game_id, team_id,
                 fga - oreb + tov + 0.44*fta AS est
          FROM team_game_box WHERE fga IS NOT NULL AND oreb IS NOT NULL AND tov IS NOT NULL AND fta IS NOT NULL),
        j AS (SELECT p.league, p.source, p.season, p.n_poss, b.est
              FROM p JOIN b ON p.league=b.league AND p.source=b.source AND p.canonical_game_id=b.canonical_game_id
                            AND p.offense_team_id=b.team_id)
        SELECT league, source, count(*) team_games,
               round(100.0*count(*) FILTER (WHERE abs(n_poss - est) <= 3)/count(*),2) AS pct_within_3,
               round(100.0*count(*) FILTER (WHERE abs(n_poss - est) <= 6)/count(*),2) pct_within_6,
               round(median(n_poss - est),2) median_diff
        FROM j GROUP BY 1,2 ORDER BY 1,2""").df().to_dict("records")
    print(f"  S2 checks done  {time.time()-t0:6.0f}s", flush=True)
    # 3d. lineup reconstruction QC by league x era (corpus's own QC, class shares)
    s2["lineup_qc_by_league_season"] = con.execute("""
        SELECT league, source, season, lineup_class, count(*) games,
               round(avg(qc_five_on_court),4) mean_five_on_court, round(avg(lineup_coverage_pct),2) mean_coverage_pct
        FROM lineup_reconstruction_qc GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""").df().to_dict("records")
    s2["coverage_tiers"] = con.execute("""
        SELECT league, grain, season, n_games, box_pct, pbp_basic_pct, pbp_coord_pct, lineup_full_pct,
               lineup_quality_full_games, tracking_pct
        FROM coverage_tiers ORDER BY 1,2,3""").df().to_dict("records")
    s2["substitution_parse_confidence"] = con.execute("""
        SELECT league, source, parse_confidence, parse_method, count(*) n FROM substitutions
        GROUP BY 1,2,3,4 ORDER BY 1,2,5 DESC""").df().to_dict("records")
    s2["possession_versions"] = con.execute("""
        SELECT league, source, possession_origin, parser_version, possession_rule, count(*) n,
               round(100.0*count(*) FILTER (WHERE ambiguous_boundary)/count(*),2) pct_ambiguous,
               round(100.0*count(*) FILTER (WHERE points_reconciled)/count(*),2) pct_points_reconciled,
               round(100.0*count(*) FILTER (WHERE players_on_floor IS NOT NULL)/count(*),2) pct_with_lineup
        FROM possessions GROUP BY 1,2,3,4,5 ORDER BY 1,2,3""").df().to_dict("records")
    with open(os.path.join(OUT, "s2_checks.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(key_uniqueness=uniq, **s2), fh, indent=1, default=str)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
