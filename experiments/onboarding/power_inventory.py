"""Structural sample-size / power inventory using COUNTS ONLY (exposure S1).

Counts, per league and family, of the units the frozen experiments and strata need:
games, player-games, players, rookies, in-season team changes, returns from absence,
possessions, substitutions, events. No outcome statistic is computed.

    python experiments/onboarding/power_inventory.py
"""
import json
import os

import duckdb
import pandas as pd

DB = (r"C:\Users\Tray\Documents\LLM AI Models\Kimi\bball_corpus\release\BASKETBALL_CORPUS_v1.0.2"
      r"\researcher_isolated\researcher_isolated_v1.0.2.duckdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "onboarding", "power_inventory.json")

con = duckdb.connect(DB, read_only=True)
con.execute("SET memory_limit='6GB'")
out = {}
fams = [("nba", "espn"), ("nba", "nba_stats"), ("wnba", "espn"), ("ncaa_men", "espn"), ("ncaa_men", "ncaa"),
        ("ncaa_women", "espn"), ("ncaa_women", "ncaa")]
for lg, src in fams:
    k = f"{lg}/{src}"
    r = {}
    r["games_with_box_by_season"] = con.execute(f"""
        SELECT season, count(DISTINCT source_game_id) games FROM team_game_box
        WHERE league='{lg}' AND source='{src}' AND season <= 2022 GROUP BY 1 ORDER BY 1""").df().set_index("season")["games"].to_dict()
    r["player_games"] = con.execute(f"SELECT count(*) FROM player_game_box WHERE league='{lg}' AND source='{src}' AND season <= 2022 AND canonical_player_id IS NOT NULL").fetchone()[0]
    r["players"] = con.execute(f"SELECT count(DISTINCT canonical_player_id) FROM player_game_box WHERE league='{lg}' AND source='{src}' AND season <= 2022").fetchone()[0]
    # rookies: first corpus appearance in a season after the family's first season (left-censoring excluded)
    r["rookies_by_season"] = con.execute(f"""
        WITH f AS (SELECT canonical_player_id, min(season) s0 FROM player_game_box WHERE league='{lg}' AND source='{src}' AND canonical_player_id IS NOT NULL GROUP BY 1),
             m AS (SELECT min(season) s FROM player_game_box WHERE league='{lg}' AND source='{src}')
        SELECT s0 season, count(*) n FROM f, m WHERE s0 > m.s AND s0 <= 2022 GROUP BY 1 ORDER BY 1""").df().set_index("season")["n"].to_dict()
    # in-season team changes (own_move proxy): player-seasons with box rows for > 1 team
    r["player_seasons_with_team_change"] = con.execute(f"""
        SELECT count(*) FROM (SELECT canonical_player_id, season FROM player_game_box WHERE league='{lg}' AND source='{src}' AND season <= 2022
          AND canonical_player_id IS NOT NULL AND team_id IS NOT NULL GROUP BY 1,2 HAVING count(DISTINCT team_id) > 1)""").fetchone()[0]
    # returns from absence: player appearances preceded by >= 5 of the team's games without the player (same season)
    r["returns_after_5_team_games_absent"] = con.execute(f"""
        WITH tg AS (SELECT DISTINCT league, source, season, team_id, source_game_id, g.game_date
                    FROM team_game_box t JOIN games g USING (league, source, source_game_id, season)
                    WHERE league='{lg}' AND source='{src}' AND season <= 2022 AND team_id IS NOT NULL),
             tgn AS (SELECT *, row_number() OVER (PARTITION BY season, team_id ORDER BY game_date, source_game_id) gidx FROM tg),
             app AS (SELECT p.canonical_player_id, p.season, p.team_id, tgn.gidx
                     FROM player_game_box p JOIN tgn ON p.league=tgn.league AND p.source=tgn.source AND p.season=tgn.season AND p.team_id=tgn.team_id AND p.source_game_id=tgn.source_game_id
                     WHERE p.league='{lg}' AND p.source='{src}' AND p.season <= 2022 AND p.canonical_player_id IS NOT NULL
                       AND (TRY_CAST(p.minutes AS DOUBLE) > 0 OR TRY_CAST(p.secs_played AS DOUBLE) > 0)),
             lag AS (SELECT *, gidx - lag(gidx) OVER (PARTITION BY canonical_player_id, season, team_id ORDER BY gidx) gap FROM app)
        SELECT count(*) FROM lag WHERE gap >= 6""").fetchone()[0]
    r["possessions"] = con.execute(f"SELECT count(*) FROM possessions WHERE league='{lg}' AND source='{src}' AND season <= 2022").fetchone()[0]
    r["substitutions"] = con.execute(f"SELECT count(*) FROM substitutions WHERE league='{lg}' AND source='{src}' AND season <= 2022").fetchone()[0]
    r["events"] = con.execute(f"SELECT count(*) FROM play_by_play WHERE league='{lg}' AND source='{src}' AND season <= 2022").fetchone()[0]
    r["seasons_with_box"] = sorted(r["games_with_box_by_season"])
    out[k] = r
    print(k, {kk: (v if not isinstance(v, dict) else f"{len(v)} seasons") for kk, v in r.items()}, flush=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=int)
print("wrote", OUT)
