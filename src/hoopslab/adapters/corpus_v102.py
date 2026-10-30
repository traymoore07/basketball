"""Read-only adapter: BASKETBALL_CORPUS researcher release v1.0.2 -> hoopslab data contract.

Source: the physically isolated researcher package
    release/BASKETBALL_CORPUS_v1.0.2/researcher_isolated/researcher_isolated_v1.0.2.duckdb
opened READ-ONLY. Never the unrestricted Bronze/Silver trees. The package holds no
holdout (season >= 2023) in-game row, so the adapter cannot expose one; holdout
schedule rows (pre-game columns only) are served as GAME version 1 (status scheduled)
with no outcome version, which is exactly what the lockbox guard expects.

One adapter instance = one league x one source family x a season range. The corpus
keeps every source family separately (canonical ids are PER SOURCE and PER LEAGUE),
so a family is never mixed inside one store. Which family serves which level is
documented in docs/onboarding/01_corpus_compatibility_audit.md.

Knowledge-time provenance (the corpus records effective time only; see its
temporal_semantics table). Every rule below errs LATE, per the contract's fallback
rules (contract/data_contract.yaml -> fallback_rules):

  GAME v1 (schedule)         tip - SCHEDULE_LEAD_DAYS, imputed_conservative, status scheduled
  GAME v2 (final)            tip + POST_GAME_HOURS,   imputed_conservative, status final
  TEAM_GAME, PLAYER_GAME,
  EVENT, POSSESSION, SHOT,
  SUBSTITUTION, LINEUP_STINT tip + POST_GAME_HOURS,   imputed_conservative
  STARTING_LINEUP            tip + POST_GAME_HOURS (REALISED starters; no announcement
                             exists, so it is never visible pre-game: contract 11 §4)
  ROSTER                     spells built from game-day roster / box appearances:
                             a spell becomes known POST_GAME_HOURS after the first game
                             on which the player appears for the team (contract rule
                             roster_membership_season_only, made stricter); it is
                             closed when the player appears for another team, or
                             after ROSTER_GAP_GAMES team games without the player
                             (known after that game)
  PLAYER, TEAM, RULESET,
  SEASON                     season start (RULESET: minus one day), bounded

Identifiers are namespaced by league and family so that the corpus's per-league id
collisions (cp:espn:10 is both an NBA and a WNBA player) cannot merge two people.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import DataAdapter

DEFAULT_DB = os.environ.get(
    "HOOPSLAB_CORPUS_DB",
    r"C:\Users\Tray\Documents\LLM AI Models\Kimi\bball_corpus\release\BASKETBALL_CORPUS_v1.0.2"
    r"\researcher_isolated\researcher_isolated_v1.0.2.duckdb")

SCHEDULE_LEAD_DAYS = 7
POST_GAME_HOURS = 4
ROSTER_GAP_GAMES = 20          # team games without the player before the spell closes
HOLDOUT_FROM_SEASON = 2023          # corpus SPLITS.json; GAME rows only beyond this

LEAGUE_ENUM = {"nba": "NBA", "wnba": "WNBA", "ncaa_men": "NCAAM", "ncaa_women": "NCAAW"}
# A season with fewer played box rows carrying minutes than this is excluded from the
# Level-A tables (seconds is a required outcome; NULL would silently become
# "did not appear"). Recorded in provenance; the audit lists the affected seasons.
MIN_MINUTES_COVERAGE = 0.95

FAMILY_SOURCE = {"espn": "espn", "nba_stats": "nba_stats", "wnba_stats": "wnba_stats", "ncaa": "ncaa"}


@dataclass
class CorpusV102Adapter(DataAdapter):
    league: str
    family: str = "espn"
    seasons: list[int] | None = None          # END years; None = every research/validation season
    level: str = "A"                          # A | B | C : how much to load
    db_path: str = DEFAULT_DB
    min_minutes_coverage: float = MIN_MINUTES_COVERAGE
    include_holdout_schedule: bool = True
    name: str = field(init=False)
    _excluded_seasons: list = field(default_factory=list, init=False)
    _meta: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert self.league in LEAGUE_ENUM, self.league
        assert self.family in FAMILY_SOURCE, self.family
        assert self.level in ("A", "B", "C")
        self.name = f"corpus_v102:{self.league}:{self.family}:{self.level}"

    # ------------------------------------------------------------------ helpers
    def _con(self):
        import duckdb
        return duckdb.connect(self.db_path, read_only=True)

    def _q(self, con, sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    def _season_filter(self, col: str = "season") -> str:
        if self.seasons:
            return f"{col} IN ({', '.join(str(int(s)) for s in self.seasons)})"
        return f"{col} < {HOLDOUT_FROM_SEASON}"

    def _pid(self, s: pd.Series) -> pd.Series:
        return (self.league + ":" + s.astype("string")).astype(object)

    def _tid(self, s: pd.Series) -> pd.Series:
        return (self.league + ":" + self.family + ":" + s.astype("string")).astype(object)

    def _gid(self, s: pd.Series) -> pd.Series:
        """game key: the family's own source_game_id, namespaced (unique within a family)."""
        return (self.league + ":" + self.family + ":" + s.astype("string")).astype(object)

    @staticmethod
    def _common(df: pd.DataFrame, kt: pd.Series, quality: str, source: str, version: int | None = None):
        df["knowledge_time"] = pd.to_datetime(kt, utc=True)
        df["knowledge_time_quality"] = quality
        df["source"] = source
        if version is not None:
            df["record_version"] = version
        return df

    # ------------------------------------------------------------------ tables
    def load_tables(self) -> dict[str, pd.DataFrame]:
        con = self._con()
        try:
            self._meta = dict(self._q(con, "SELECT k, v FROM isolated_release_meta").values.tolist())
            seasons = self._q(con, f"SELECT league, season, season_start, season_end, ruleset_id FROM seasons "
                                   f"WHERE league = '{self.league}' ORDER BY season")
            rules = self._ruleset(con, seasons)
            games_raw = self._games_raw(con)
            pg_raw = self._player_box_raw(con)
            excluded = self._minutes_exclusion(pg_raw)
            self._excluded_seasons = excluded
            if excluded:
                games_raw = games_raw[~games_raw["season"].isin(excluded)]
                pg_raw = pg_raw[~pg_raw["season"].isin(excluded)]
            tables = {}
            tables["RULESET"] = rules
            tables["SEASON"] = self._season_table(seasons)
            tables["GAME"] = self._game(games_raw)
            tip = games_raw.drop_duplicates("game_id").set_index("game_id")["scheduled_tip"]
            if not len(games_raw):
                raise ValueError(f"{self.name}: no games after exclusions (seasons excluded for minutes coverage: {excluded})")
            tables["TEAM_GAME"] = self._team_game(con, tip, excluded, pg_raw)
            tables["PLAYER_GAME"] = self._player_game(pg_raw, tip)
            tables["ROSTER"] = self._roster(con, tables["PLAYER_GAME"], tip)
            tables["TEAM"] = self._team(tables["GAME"], tables["TEAM_GAME"], seasons)
            tables["PLAYER"] = self._player(con, tables["PLAYER_GAME"], seasons)
            if self.level in ("B", "C"):
                tables["EVENT"] = self._event(con, tip, rules, games_raw)
                tables["POSSESSION"] = self._possession(con, tip, tables["EVENT"], games_raw)
                tables["SHOT"] = self._shot_from_events(tables["EVENT"])
            if self.level == "C":
                tables["SUBSTITUTION"] = self._substitution(con, tip, rules, games_raw)
                st = self._lineup_stint(con, tip, rules, games_raw)
                if len(st):
                    tables["LINEUP_STINT"] = st
                    if not tables["PLAYER_GAME"]["started"].any():
                        # sources without a starter flag (nba_stats, ncaa): the opening lineup
                        # stint of each team IS the realised starting five; transcribe it into
                        # PLAYER_GAME.started (an outcome field) as the corpus's own lineups say
                        first = st.sort_values(["game_id", "team_id", "start_elapsed_s", "stint_seq"]).drop_duplicates(["game_id", "team_id"])
                        opener = {(g, t): set(p) for g, t, p in zip(first["game_id"], first["team_id"], first["player_ids"])}
                        pg = tables["PLAYER_GAME"]
                        pg["started"] = [pid in opener.get((g, t), ()) for g, t, pid in zip(pg["game_id"], pg["team_id"], pg["player_id"])]
                        self._meta["starters_from_opening_lineup_stint"] = int(pg["started"].sum())
                tables["STARTING_LINEUP"] = self._starting_lineup(tables["PLAYER_GAME"])
            return tables
        finally:
            con.close()

    # ---- RULESET / SEASON ----------------------------------------------------
    def _ruleset(self, con, seasons: pd.DataFrame) -> pd.DataFrame:
        g = self._q(con, f"""
            SELECT ruleset_id, season, regulation_periods, regulation_period_minutes, overtime_period_minutes,
                   shot_clock_seconds, off_rebound_reset_seconds, three_point_arc_ft, three_point_corner_ft,
                   bonus_penalty_foul_number, bonus_rule, foul_out_limit
            FROM rule_era_geometry WHERE league = '{self.league}' AND regulation_periods IS NOT NULL""")
        g = g.drop_duplicates("ruleset_id").merge(seasons[["season", "season_start", "season_end"]], on="season", how="inner")
        out = pd.DataFrame({
            "ruleset_id": g["ruleset_id"],
            "league": LEAGUE_ENUM[self.league],
            "effective_from": pd.to_datetime(g["season_start"]).dt.date,
            "effective_to": pd.to_datetime(g["season_end"]).dt.date,
            "periods": g["regulation_periods"].astype(int),
            "period_length_s": (g["regulation_period_minutes"] * 60).astype(int),
            # overtime: the corpus has no verified value for some NCAA seasons; 5 minutes is
            # the rule in every league in scope (corpus rule_eras, area overtime_rules)
            "overtime_length_s": (g["overtime_period_minutes"].fillna(5) * 60).astype(int),
            "shot_clock_s": g["shot_clock_seconds"].astype("Int64"),
            "shot_clock_offensive_rebound_reset_s": g["off_rebound_reset_seconds"].astype("Int64"),
            "three_point_distance_m": (g["three_point_arc_ft"] * 0.3048).round(3),
            "three_point_corner_distance_m": (g["three_point_corner_ft"] * 0.3048).round(3),
            "personal_foul_limit": g["foul_out_limit"].astype("Int64"),
            "team_foul_bonus_rule": g["bonus_rule"].astype("string"),
            "notes": "values from corpus rule_era_geometry (research-sourced, per season)",
        })
        # required fields the corpus leaves NULL for some seasons: fill from the league's
        # modal value and say so (structural, not an estimate of anything a hypothesis is about)
        # Personal-foul limit where the corpus geometry has no value at all (WNBA, NCAAW):
        # NCAAW -> 5 from the corpus's own rule_eras record (DQ at 5 fouls, verified, no later
        # change recorded); WNBA -> 6 from the WNBA rulebook, which the corpus cites for team
        # fouls but did not extract the personal limit from. The WNBA value is the one value in
        # this adapter not taken from the corpus; logged as a data-contract mismatch.
        NOT_IN_CORPUS = {"ncaa_women": (5, "rule_eras ncaa_women 1918 record: DQ at 5 fouls"),
                         "wnba": (6, "NOT IN CORPUS: WNBA rulebook personal-foul limit (adapter default, flagged)")}
        for c, default in (("shot_clock_s", None), ("personal_foul_limit", NOT_IN_CORPUS.get(self.league, (None, ""))[0])):
            if out[c].isna().any():
                mode = out[c].dropna().mode()
                fill = int(mode.iloc[0]) if len(mode) else default
                why = f"league mode {fill}" if len(mode) else NOT_IN_CORPUS.get(self.league, (None, "no value"))[1]
                if fill is not None:
                    out.loc[out[c].isna(), "notes"] = out["notes"] + f"; {c} filled: {why}"
                    out[c] = out[c].fillna(fill)
        out["shot_clock_s"] = out["shot_clock_s"].astype(int)
        out["personal_foul_limit"] = out["personal_foul_limit"].astype(int)
        out["three_point_distance_m"] = out["three_point_distance_m"].fillna(out["three_point_distance_m"].median())
        out["team_foul_bonus_rule"] = out["team_foul_bonus_rule"].fillna("unknown").astype(object)
        kt = pd.to_datetime(g["season_start"], utc=True) - pd.Timedelta(days=1)
        return self._common(out, kt, "bounded", "corpus_v102:rule_era_geometry")

    def _season_table(self, seasons: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "league": LEAGUE_ENUM[self.league],
            "season_id": self.league + "-" + seasons["season"].astype(str),
            "phase": "regular",
            "start_date": pd.to_datetime(seasons["season_start"]).dt.date,
            "end_date": pd.to_datetime(seasons["season_end"]).dt.date,
        })
        kt = pd.to_datetime(seasons["season_start"], utc=True) - pd.Timedelta(days=90)
        return self._common(out, kt, "bounded", "corpus_v102:seasons")

    # ---- GAME ----------------------------------------------------------------
    def _games_raw(self, con) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        upper = "" if self.include_holdout_schedule else f" AND season < {HOLDOUT_FROM_SEASON}"
        sf = self._season_filter() if self.seasons else "TRUE"
        g = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, source_game_id, season, game_date, game_date_time_utc, season_type, game_type,
                   home_team_id, away_team_id, neutral_site, venue_id, status, overtime_count
            FROM games WHERE league = '{self.league}' AND source = '{src}' AND ({sf}){upper}""")
        if g["home_team_id"].isna().all():
            # the ncaa source schedule carries no team ids; its team box does (is_home flag)
            tb = self._q(con, f"""
                SELECT source_game_id, max(CASE WHEN is_home THEN team_id END) home_team_id,
                       max(CASE WHEN NOT is_home THEN team_id END) away_team_id
                FROM team_game_box WHERE league = '{self.league}' AND source = '{src}' AND team_id IS NOT NULL
                GROUP BY 1""")
            g = g.drop(columns=["home_team_id", "away_team_id"]).merge(tb, on="source_game_id", how="left")
            self._meta["game_teams_from_team_box"] = True
        g = g[g["home_team_id"].notna() & g["away_team_id"].notna()]
        g = g[~g["season_type"].astype(str).str.lower().isin(["all-star"])]
        g = g[~g["game_type"].astype(str).isin(["ALLSTAR"])]
        # scheduled tip: exact instant where the source carries one; else date-only ->
        # noon UTC on the game date, flagged (structural fallback, documented in the audit)
        tip_exact = pd.to_datetime(g["game_date_time_utc"], utc=True, errors="coerce")
        tip_date = pd.to_datetime(g["game_date"], errors="coerce").dt.tz_localize("UTC") + pd.Timedelta(hours=12)
        g["scheduled_tip"] = tip_exact.fillna(tip_date)
        g["tip_quality"] = np.where(tip_exact.notna(), "exact", "date_only")
        g = g[g["scheduled_tip"].notna()]
        g["game_id"] = self._gid(g["source_game_id"])
        return g.reset_index(drop=True)

    def _phase(self, g: pd.DataFrame) -> pd.Series:
        st = g["season_type"].astype(str).str.lower()
        gt = g["game_type"].astype(str).str.upper()
        phase = pd.Series("regular", index=g.index, dtype=object)
        if self.league in ("nba", "wnba"):
            phase[st.isin(["3", "playoffs"])] = "postseason"
            phase[st.isin(["1", "preseason"])] = "preseason"
            phase[st.isin(["5", "play-in"])] = "play_in"
            phase[st.isin(["nba-cup"])] = "in_season_tournament"
        else:
            phase[st.isin(["3"])] = "national_tournament"
            phase[(st == "2") & gt.isin(["TRNMNT", "CC", "CONFERENCE CHAMPIONSHIP"])] = "conference_tournament"
            phase[gt.isin(["EXH"])] = "preseason"
        return phase

    def _game(self, g: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame({
            "game_id": g["game_id"],
            "league": LEAGUE_ENUM[self.league],
            "season_id": self.league + "-" + g["season"].astype(str),
            "phase": self._phase(g),
            "ruleset_id": "rs:" + self.league + ":" + g["season"].astype(str),
            "scheduled_tip": g["scheduled_tip"],
            "home_team_id": self._tid(g["home_team_id"]),
            "away_team_id": self._tid(g["away_team_id"]),
            "neutral_site": g["neutral_site"].astype(str).str.lower().eq("true"),
            "venue_id": g["venue_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{self.family}:{v}"),
            "tip_quality": g["tip_quality"],
        })
        v1 = base.copy()
        v1["status"] = "scheduled"
        v1["periods_played"] = pd.array([None] * len(v1), dtype="Int64")
        v1 = self._common(v1, base["scheduled_tip"] - pd.Timedelta(days=SCHEDULE_LEAD_DAYS),
                          "imputed_conservative", f"corpus_v102:games:{self.family}", version=1)
        played = g["season"] < HOLDOUT_FROM_SEASON
        v2 = base[played].copy()
        v2["status"] = "final"
        ot = pd.to_numeric(g.loc[played, "overtime_count"], errors="coerce").fillna(0).astype(int)
        v2["periods_played"] = pd.array((ot + (4 if self.league in ("nba",) else 0)).astype(int), dtype="Int64")
        v2 = self._common(v2, base.loc[played, "scheduled_tip"] + pd.Timedelta(hours=POST_GAME_HOURS),
                          "imputed_conservative", f"corpus_v102:games:{self.family}", version=2)
        # periods_played is only meaningful where regulation periods are known per season;
        # it is informational (outcome) and left as NULL for leagues whose period count varies
        if self.league != "nba":
            v2["periods_played"] = pd.array([None] * len(v2), dtype="Int64")
        return pd.concat([v1, v2], ignore_index=True)

    # ---- boxes ---------------------------------------------------------------
    def _player_box_raw(self, con) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        return self._q(con, f"""
            SELECT canonical_game_id, source_game_id, canonical_player_id, team_id, season, starter, did_not_play,
                   minutes, secs_played, points, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, plus_minus
            FROM player_game_box WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()}
              AND canonical_player_id IS NOT NULL AND team_id IS NOT NULL""")

    @staticmethod
    def _activity(pg: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """(minutes_known, played_evidence): a row shows play when it has minutes or any
        positive counting stat. Sources without a DNP flag (nba_stats, ncaa) leave minutes
        NULL on DNP rows with zero stats, which must not count as missing minutes."""
        mins = pd.to_numeric(pg["minutes"], errors="coerce").fillna(pd.to_numeric(pg["secs_played"], errors="coerce"))
        known = mins.notna()
        stats = sum(pd.to_numeric(pg[c], errors="coerce").fillna(0) for c in
                    ("points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"))
        played = known | (stats > 0)
        return known, played

    def _minutes_exclusion(self, pg: pd.DataFrame) -> list[int]:
        """Seasons where rows with evidence of play lack minutes: structural coverage."""
        known, played = self._activity(pg)
        sub = pg[played]
        cov = known[played].groupby(sub["season"]).mean()
        return sorted(int(s) for s, c in cov.items() if c < self.min_minutes_coverage)

    def _player_game(self, pg: pd.DataFrame, tip: pd.Series) -> pd.DataFrame:
        pg = pg.copy()
        pg["game_id"] = self._gid(pg["source_game_id"])
        pg = pg[pg["game_id"].isin(tip.index)]
        secs = pd.to_numeric(pg["secs_played"], errors="coerce")
        mins = pd.to_numeric(pg["minutes"], errors="coerce")
        seconds = secs.fillna(mins * 60)
        known, played = self._activity(pg)
        dnp = pg["did_not_play"].astype(str).str.lower().eq("true") | ~played
        # a row with evidence of play but no minutes cannot satisfy the contract (seconds is a
        # required outcome); it is dropped and counted, never imputed
        unknown_played = played & ~known & ~dnp
        self._meta["player_game_rows_dropped_unknown_minutes"] = int(unknown_played.sum())
        pg = pg[~unknown_played]; seconds = seconds[~unknown_played]; dnp = dnp[~unknown_played]
        seconds = seconds.where(~dnp, 0).fillna(0)
        num = {c: pd.to_numeric(pg[c], errors="coerce").fillna(0).astype(int) for c in
               ("points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")}
        started = pg["starter"].astype(str).str.lower().eq("true")
        status = np.where(seconds > 0, "played", np.where(dnp, "dnp_coach", "inactive"))
        out = pd.DataFrame({
            "game_id": pg["game_id"], "player_id": self._pid(pg["canonical_player_id"]),
            "team_id": self._tid(pg["team_id"]), "status": status, "started": started,
            "seconds": seconds.round().astype(int), "pts": num["points"], "fgm": num["fgm"], "fga": num["fga"],
            "fg3m": num["fg3m"], "fg3a": num["fg3a"], "ftm": num["ftm"], "fta": num["fta"], "oreb": num["oreb"],
            "dreb": num["dreb"], "ast": num["ast"], "stl": num["stl"], "blk": num["blk"], "tov": num["tov"],
            "pf": num["pf"], "plus_minus": pd.to_numeric(pg["plus_minus"], errors="coerce").astype("Int64"),
        })
        # the corpus has a handful of duplicated (game, player) rows (team changes recorded twice);
        # keep the row with the most seconds, deterministically
        out = out.sort_values(["game_id", "player_id", "seconds"], kind="stable").drop_duplicates(["game_id", "player_id"], keep="last")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:player_game_box:{self.family}", version=1).reset_index(drop=True)

    def _team_game(self, con, tip: pd.Series, excluded: list[int], pg_raw: pd.DataFrame | None = None) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        t = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, team_id, is_home, final_score, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf
            FROM team_game_box WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()}
              AND team_id IS NOT NULL""")
        t["game_id"] = self._gid(t["source_game_id"])
        t = t[t["game_id"].isin(tip.index)]
        num = {c: pd.to_numeric(t[c], errors="coerce") for c in ("final_score", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")}
        out = pd.DataFrame({"game_id": t["game_id"], "team_id": self._tid(t["team_id"]), "is_home": t["is_home"].astype(bool)})
        for c, k in (("pts", "final_score"), ("fgm", "fgm"), ("fga", "fga"), ("fg3m", "fg3m"), ("fg3a", "fg3a"), ("ftm", "ftm"),
                     ("fta", "fta"), ("oreb", "oreb"), ("dreb", "dreb"), ("ast", "ast"), ("stl", "stl"), ("blk", "blk"), ("tov", "tov"), ("pf", "pf")):
            out[c] = num[k]
        # the ncaa source carries no steals / fouls at team level: fill them with the sum of
        # the players' box rows for that (game, team) -- a transcription of the same box,
        # recorded in provenance; any remaining NULL in a required count drops the row
        req = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
        if pg_raw is not None and len(pg_raw):
            agg = pg_raw.assign(game_id=self._gid(pg_raw["source_game_id"]), team_id=self._tid(pg_raw["team_id"]))
            for c in ("stl", "pf", "blk", "ast", "tov", "oreb", "dreb"):
                if out[c].isna().any():
                    s = pd.to_numeric(agg[c], errors="coerce").groupby([agg["game_id"], agg["team_id"]]).sum(min_count=1)
                    key = pd.MultiIndex.from_arrays([out["game_id"], out["team_id"]])
                    filled = pd.Series(s.reindex(key).to_numpy(), index=out.index)
                    n = int((out[c].isna() & filled.notna()).sum())
                    if n:
                        self._meta[f"team_game_{c}_filled_from_player_sums"] = n
                    out[c] = out[c].fillna(filled)
        complete = out[req].notna().all(axis=1)
        self._meta["team_game_rows_dropped_incomplete"] = int((~complete).sum())
        out = out[complete].copy()
        for c in req:
            out[c] = out[c].astype(int)
        out = out.drop_duplicates(["game_id", "team_id"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:team_game_box:{self.family}", version=1).reset_index(drop=True)

    # ---- ROSTER / TEAM / PLAYER ---------------------------------------------
    def _roster(self, con, pg: pd.DataFrame, tip: pd.Series) -> pd.DataFrame:
        """Membership spells from game-day appearances (box rows incl. DNP).

        A spell opens at the first game on which the player appears for the team
        (known POST_GAME_HOURS later). It closes, as a second version, when the player
        appears for another team (known after that game), or after the team has played
        ROSTER_GAP_GAMES consecutive games without the player (known after that game).
        Counting team games rather than days keeps spells open across the offseason,
        which an absence-in-days rule would close, emptying opening-night populations.
        """
        obs = pg[["game_id", "player_id", "team_id"]].copy()
        obs["tip"] = obs["game_id"].map(tip)
        # every game each team played, in order (from the appearances themselves)
        team_games = obs.drop_duplicates(["team_id", "game_id"])[["team_id", "game_id", "tip"]].sort_values(["team_id", "tip"], kind="stable")
        team_sched = {t: g["tip"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
                      for t, g in team_games.groupby("team_id", sort=False)}
        obs = obs.sort_values(["player_id", "tip"], kind="stable")
        rows = []
        post = pd.Timedelta(hours=POST_GAME_HOURS)

        def gap_close(team, last_tip):
            """tip of the team's ROSTER_GAP_GAMES-th game after last_tip, or None."""
            sched = team_sched[team]
            i = np.searchsorted(sched, last_tip.value, side="right")
            j = i + ROSTER_GAP_GAMES - 1
            return pd.Timestamp(int(sched[j]), unit="ns", tz="UTC") if j < len(sched) else None

        for pid, grp in obs.groupby("player_id", sort=False):
            cur_team, start, last = None, None, None
            for tid, t in zip(grp["team_id"], grp["tip"]):
                if cur_team is None:
                    cur_team, start, last = tid, t, t
                    continue
                gc = gap_close(cur_team, last)
                if gc is not None and gc < t:
                    rows.append((pid, cur_team, start, None, start + post, 1))
                    rows.append((pid, cur_team, start, gc, gc + post, 2))
                    cur_team, start, last = tid, t, t
                elif tid != cur_team:
                    rows.append((pid, cur_team, start, None, start + post, 1))
                    rows.append((pid, cur_team, start, t, t + post, 2))
                    cur_team, start, last = tid, t, t
                else:
                    last = t
            rows.append((pid, cur_team, start, None, start + post, 1))
            gc = gap_close(cur_team, last)
            if gc is not None:
                rows.append((pid, cur_team, start, gc, gc + post, 2))
        out = pd.DataFrame(rows, columns=["player_id", "team_id", "valid_from", "valid_to", "knowledge_time", "record_version"])
        out["valid_from"] = pd.to_datetime(out["valid_from"], utc=True)
        out["valid_to"] = pd.to_datetime(out["valid_to"], utc=True)
        out["transaction_type"] = "other"
        out["knowledge_time"] = pd.to_datetime(out["knowledge_time"], utc=True)
        out["knowledge_time_quality"] = "imputed_conservative"
        out["source"] = f"corpus_v102:appearances:{self.family}"
        return out.reset_index(drop=True)

    def _team(self, game: pd.DataFrame, tg: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
        ids = pd.unique(pd.concat([game["home_team_id"], game["away_team_id"], tg["team_id"]]))
        first = pd.concat([game[["home_team_id", "scheduled_tip"]].rename(columns={"home_team_id": "team_id"}),
                           game[["away_team_id", "scheduled_tip"]].rename(columns={"away_team_id": "team_id"})]).groupby("team_id")["scheduled_tip"].min()
        out = pd.DataFrame({"team_id": ids})
        out["franchise_id"] = out["team_id"]
        out["league"] = LEAGUE_ENUM[self.league]
        out["valid_from"] = out["team_id"].map(first).dt.date
        kt = pd.to_datetime(out["team_id"].map(first), utc=True) - pd.Timedelta(days=90)
        return self._common(out, kt, "bounded", f"corpus_v102:games:{self.family}")

    def _player(self, con, pg: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
        p = self._q(con, f"SELECT canonical_player_id, height_inches, weight, position, first_season FROM players WHERE league = '{self.league}'")
        p["player_id"] = self._pid(p["canonical_player_id"])
        p = p[p["player_id"].isin(pg["player_id"].unique())].drop_duplicates("player_id")
        out = pd.DataFrame({
            "player_id": p["player_id"],
            "height_cm": (pd.to_numeric(p["height_inches"], errors="coerce") * 2.54).round(1),
            "weight_kg": (pd.to_numeric(p["weight"], errors="coerce") * 0.45359).round(1),
            "listed_position": p["position"].astype(object),
        })
        start = seasons.set_index("season")["season_start"]
        kt = pd.to_datetime(p["first_season"].map(start), utc=True)
        kt = kt.fillna(pd.Timestamp("1990-01-01", tz="UTC"))
        return self._common(out, kt, "bounded", "corpus_v102:players", version=1).reset_index(drop=True)

    # ---- Level B: EVENT / POSSESSION / SHOT ----------------------------------
    @staticmethod
    def _clock_to_s(s: pd.Series) -> pd.Series:
        """'MM:SS', 'M:SS', 'PT12M00.00S' -> seconds remaining (float)."""
        x = s.astype("string")
        iso = x.str.extract(r"PT(\d+)M([\d.]+)S")
        mmss = x.str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
        out = pd.to_numeric(iso[0], errors="coerce") * 60 + pd.to_numeric(iso[1], errors="coerce")
        out = out.fillna(pd.to_numeric(mmss[0], errors="coerce") * 60 + pd.to_numeric(mmss[1], errors="coerce"))
        return out.astype(float)

    def _elapsed(self, period: pd.Series, clock_remaining: pd.Series, season: pd.Series, rules: pd.DataFrame) -> pd.Series:
        r = rules.set_index("ruleset_id")
        rid = "rs:" + self.league + ":" + season.astype(str)
        nper = rid.map(r["periods"]).astype(float)
        plen = rid.map(r["period_length_s"]).astype(float)
        otlen = rid.map(r["overtime_length_s"]).astype(float)
        p = period.astype(float)
        reg = np.minimum(p - 1, nper) * plen
        ot = np.maximum(p - 1 - nper, 0) * otlen
        cur_len = np.where(p <= nper, plen, otlen)
        return reg + ot + (cur_len - clock_remaining)

    def _event(self, con, tip: pd.Series, rules: pd.DataFrame, games_raw: pd.DataFrame) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        e = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, season, sequence, period, game_clock, team_id, player_id, secondary_player_id,
                   tertiary_player_id, event_type, event_subtype, event_family, description, home_score, away_score, x, y,
                   x_raw, y_raw, shot_distance, shot_value, made, is_substitution
            FROM play_by_play WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()}""")
        e["game_id"] = self._gid(e["source_game_id"])
        e = e[e["game_id"].isin(tip.index)].copy()
        fam = e["event_family"].astype("string").fillna("")
        etype = pd.Series("other", index=e.index, dtype=object)
        m = {"shot_made": "shot", "shot_missed": "shot", "free_throw": "free_throw", "rebound": "rebound",
             "turnover": "turnover", "foul": "foul", "substitution": "substitution", "timeout": "timeout",
             "jump_ball": "jump_ball", "period_boundary": "period_end"}
        for k, v in m.items():
            etype[fam == k] = v
        desc = e["event_type"].astype("string").fillna("").str.lower()
        etype[(fam == "period_boundary") & desc.str.contains("start", na=False)] = "period_start"
        clock = self._clock_to_s(e["game_clock"])
        # shot value. ESPN's score_value is the points SCORED (0 on a miss), so the attempt
        # value of a miss is read from the source's own event wording ("three point ...").
        sv = pd.to_numeric(e["shot_value"], errors="coerce")
        is_shot = fam.isin(["shot_made", "shot_missed"])
        text = (e["event_type"].astype("string").fillna("") + " " + e["description"].astype("string").fillna("")).str.lower()
        three = text.str.contains("three point|3pt|three-point|3-pt|three pt", regex=True, na=False)
        inferred = np.where(three, 3, 2)
        sv = sv.where(~(is_shot & (sv.fillna(0) == 0)), inferred)
        sv = sv.where(is_shot | sv.notna(), 0)
        # coordinates: ESPN publishes court-centred feet (x, y); the stats families publish
        # basket-origin tenths of a foot (x_legacy, y_legacy -> x_raw, y_raw) plus a distance
        x_src = pd.to_numeric(e["x"], errors="coerce")
        y_src = pd.to_numeric(e["y"], errors="coerce")
        if x_src.notna().sum() == 0:
            x_src = pd.to_numeric(e["x_raw"], errors="coerce") / 10.0
            y_src = pd.to_numeric(e["y_raw"], errors="coerce") / 10.0
            coord_frame = "basket_origin_ft"
        else:
            coord_frame = "court_centred_ft_baskets_at_x_pm41.75"
        self._meta["event_coordinate_frame"] = coord_frame
        out = pd.DataFrame({
            "game_id": e["game_id"], "event_seq": e["sequence"].astype(int), "period": e["period"].astype(int),
            "clock_remaining_s": clock, "elapsed_s": self._elapsed(e["period"], clock, e["season"], rules),
            "event_type": etype, "event_subtype": fam.where(fam.isin(["steal", "block", "assist"]), e["event_subtype"].astype("string")).astype(object).where(lambda s: s.notna(), None),
            "team_id": e["team_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{self.family}:{v}"),
            "player1_id": e["player_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "player2_id": e["secondary_player_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "player3_id": e["tertiary_player_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "shot_made": np.where(fam == "shot_made", True, np.where(fam == "shot_missed", False, None)),
            "shot_value": sv.astype("Int64"),
            "shot_x": x_src, "shot_y": y_src,
            "shot_distance_src": pd.to_numeric(e["shot_distance"], errors="coerce"),
            "score_home": pd.to_numeric(e["home_score"], errors="coerce").astype("Int64"),
            "score_away": pd.to_numeric(e["away_score"], errors="coerce").astype("Int64"),
        })
        out["shot_made"] = out["shot_made"].astype(object)
        # required: score columns. The corpus carries them on every ESPN row; where a source leaves
        # them NULL the row cannot satisfy the contract and is dropped (count in provenance)
        ok = out["score_home"].notna() & out["score_away"].notna() & out["clock_remaining_s"].notna()
        self._meta["event_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy()
        out["score_home"] = out["score_home"].astype(int)
        out["score_away"] = out["score_away"].astype(int)
        out = out.sort_values(["game_id", "event_seq"], kind="stable").drop_duplicates(["game_id", "event_seq"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:play_by_play:{self.family}").reset_index(drop=True)

    def _possession(self, con, tip: pd.Series, event: pd.DataFrame, games_raw: pd.DataFrame) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        p = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, possession_number, offense_team_id, defense_team_id, period, start_sequence, end_sequence,
                   start_clock, end_clock, points, is_turnover, is_second_chance, count_as_possession, start_reason, end_reason,
                   possession_boundary_reason, players_on_floor, defense_players_on_floor, possession_origin
            FROM possessions WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()}
              AND coalesce(count_as_possession, TRUE) AND offense_team_id IS NOT NULL""")
        p["game_id"] = self._gid(p["source_game_id"])
        p = p[p["game_id"].isin(tip.index)].copy()
        sr = p["start_reason"].astype("string").str.lower().fillna("")
        br = p["possession_boundary_reason"].astype("string").str.lower().fillna("")
        start = pd.Series("after_dead_ball", index=p.index, dtype=object)
        start[sr.str.contains("rebound")] = "after_dreb"
        start[sr.str.contains("jump_ball|period_start|jumpball")] = "period_start"
        start[sr.str.contains("turnover|steal")] = "after_live_turnover"
        start[sr.str.contains("made|clean_boundary")] = "after_make"
        start[sr.str.contains("free_throw|free throw")] = "after_ft"
        start[p["is_second_chance"].astype(bool)] = "after_oreb"
        end = pd.Series("other", index=p.index, dtype=object)
        end[br == "made_fg"] = "made_fg"
        end[br == "defensive_rebound"] = "missed_fg_dreb"
        end[br == "turnover"] = "turnover_live"
        end[br == "made_final_ft"] = "ft_trip"
        end[br == "period_end"] = "period_end"
        start_s = self._clock_to_s(p["start_clock"])
        end_s = self._clock_to_s(p["end_clock"])
        # margin at the start: score on the first event of the possession
        p = p[p["start_sequence"].notna()].copy()
        # the derived parser (ESPN families) records the offence only; the defence is the
        # game's other team, read from the schedule
        gh = games_raw.drop_duplicates("game_id").set_index("game_id")
        home = p["game_id"].map(gh["home_team_id"]).astype("string")
        away = p["game_id"].map(gh["away_team_id"]).astype("string")
        off = p["offense_team_id"].astype("string")
        other = np.where(off == home, away, np.where(off == away, home, None))
        p["defense_team_id"] = p["defense_team_id"].astype("string").fillna(pd.Series(other, index=p.index, dtype="string"))
        ev = event[["game_id", "event_seq", "score_home", "score_away", "elapsed_s"]].rename(columns={"event_seq": "start_sequence"})
        p["start_sequence"] = p["start_sequence"].astype("int64")
        sc = p[["game_id", "start_sequence"]].merge(ev, on=["game_id", "start_sequence"], how="left")
        sc.index = p.index
        margin = (sc["score_home"].to_numpy(dtype=float) - sc["score_away"].to_numpy(dtype=float))
        out = pd.DataFrame({
            "game_id": p["game_id"], "possession_seq": p["possession_number"].astype(int),
            "segment": np.where(p["is_second_chance"].astype(bool), 2, 1),
            "off_team_id": self._tid(p["offense_team_id"]),
            "def_team_id": p["defense_team_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{self.family}:{v}"),
            "start_elapsed_s": sc["elapsed_s"].to_numpy(), "duration_s": (start_s - end_s).to_numpy(),
            "start_type": start, "end_type": end,
            "points": pd.to_numeric(p["points"], errors="coerce").fillna(0).astype(int),
            "off_margin_start": pd.Series(margin, index=p.index),
            "off_lineup": p["players_on_floor"].map(self._lineup_tuple),
            "def_lineup": p["defense_players_on_floor"].map(self._lineup_tuple),
        })
        ok = out["start_elapsed_s"].notna() & out["def_team_id"].notna() & out["off_margin_start"].notna() & out["duration_s"].notna()
        self._meta["possession_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy()
        out["off_margin_start"] = out["off_margin_start"].astype(int)
        out = out.drop_duplicates(["game_id", "possession_seq", "segment"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:possessions:{self.family}").reset_index(drop=True)

    def _lineup_tuple(self, v):
        """list/array of native player ids -> tuple of namespaced ids; any null -> None."""
        if v is None or v is pd.NA:
            return None
        if isinstance(v, float):
            return None
        try:
            ids = [x for x in v if x is not None and x is not pd.NA]
        except TypeError:
            return None
        return tuple(f"{self.league}:{x}" for x in ids) if ids else None

    def _shot_from_events(self, event: pd.DataFrame) -> pd.DataFrame:
        """SHOT as a view over EVENT: the corpus's own `shots` table carries no event key
        and (for ESPN) no outcome, while every ESPN shot event carries x/y and the
        transcribed outcome. Coordinates are the source's frame (ESPN: feet, court-centred,
        baskets at x = +/-41.75); the stats families carry no x/y in the researcher
        release, so their SHOT rows are those with a non-null distance only."""
        e = event[event["event_type"] == "shot"]
        x, y = e["shot_x"], e["shot_y"]
        if self._meta.get("event_coordinate_frame") == "basket_origin_ft":
            dist_xy = np.sqrt(x ** 2 + y ** 2)
        else:
            dist_xy = np.sqrt((x.abs() - 41.75) ** 2 + y ** 2)
        dist = e["shot_distance_src"].fillna(dist_xy) if "shot_distance_src" in e else dist_xy
        out = pd.DataFrame({
            "game_id": e["game_id"], "event_seq": e["event_seq"], "shooter_id": e["player1_id"],
            "x": x, "y": y, "distance_m": (dist * 0.3048).round(2), "shot_value": e["shot_value"].astype("Int64"),
            "action_type": e["event_subtype"], "made": e["shot_made"],
            "knowledge_time": e["knowledge_time"], "knowledge_time_quality": e["knowledge_time_quality"], "source": e["source"],
        })
        ok = out["shooter_id"].notna() & out["x"].notna() & out["y"].notna() & out["distance_m"].notna() & out["made"].notna() & (out["shot_value"].fillna(0) > 0)
        self._meta["shot_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy()
        out["shot_value"] = out["shot_value"].astype(int)
        out["made"] = out["made"].astype(bool)
        return out.reset_index(drop=True)

    def _shot(self, con, tip: pd.Series, event: pd.DataFrame | None = None) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        s = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, sequence, canonical_player_id, x, y, shot_distance, shot_value, made, shot_type,
                   assisting_player_id, blocking_player_id
            FROM shots WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()} AND canonical_player_id IS NOT NULL""")
        s["game_id"] = self._gid(s["source_game_id"])
        s = s[s["game_id"].isin(tip.index)].copy()
        x = pd.to_numeric(s["x"], errors="coerce"); y = pd.to_numeric(s["y"], errors="coerce")
        dist_ft = pd.to_numeric(s["shot_distance"], errors="coerce")
        # ESPN: x/y in feet on a court-centred frame; distance from the basket at (±41.75, 0)
        dist_xy = np.sqrt((x.abs() - 41.75) ** 2 + y ** 2)
        made = s["made"].astype("string").str.lower().map({"true": True, "false": False})
        if made.isna().all() and event is not None and len(event):
            # ESPN's shot table carries no made flag; the play-by-play row of the same
            # (game, sequence) does (shot outcome transcribed from scoring_play)
            ev = event[["game_id", "event_seq", "shot_made"]].rename(columns={"event_seq": "sequence"})
            s["sequence"] = pd.to_numeric(s["sequence"], errors="coerce").astype("Int64")
            j = s[["game_id", "sequence"]].merge(ev, on=["game_id", "sequence"], how="left")
            made = pd.Series(j["shot_made"].to_numpy(), index=s.index)
        out = pd.DataFrame({
            "game_id": s["game_id"], "event_seq": pd.to_numeric(s["sequence"], errors="coerce").astype("Int64"),
            "shooter_id": self._pid(s["canonical_player_id"]), "x": x, "y": y,
            "distance_m": (dist_ft.fillna(dist_xy) * 0.3048).round(2),
            "shot_value": pd.to_numeric(s["shot_value"], errors="coerce").astype("Int64"),
            "action_type": s["shot_type"].astype(object), "made": made.astype(object),
            "assisted_by_id": s["assisting_player_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "blocked_by_id": s["blocking_player_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
        })
        ok = out["event_seq"].notna() & out["x"].notna() & out["y"].notna() & out["distance_m"].notna() & out["shot_value"].notna() & out["made"].notna()
        self._meta["shot_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy()
        out["event_seq"] = out["event_seq"].astype(int); out["shot_value"] = out["shot_value"].astype(int)
        out = out.drop_duplicates(["game_id", "event_seq"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:shots:{self.family}").reset_index(drop=True)

    # ---- Level C ---------------------------------------------------------------
    def _substitution(self, con, tip: pd.Series, rules: pd.DataFrame, games_raw: pd.DataFrame) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        s = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, season, sequence, period, game_clock, team_id, canonical_player_in_id, canonical_player_out_id, parse_confidence
            FROM substitutions WHERE league = '{self.league}' AND source = '{src}' AND {self._season_filter()}""")
        s["game_id"] = self._gid(s["source_game_id"])
        s = s[s["game_id"].isin(tip.index)].copy()
        clock = self._clock_to_s(s["game_clock"])
        out = pd.DataFrame({
            "game_id": s["game_id"], "event_seq": pd.to_numeric(s["sequence"], errors="coerce").astype("Int64"),
            "elapsed_s": self._elapsed(s["period"], clock, s["season"], rules),
            "team_id": s["team_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{self.family}:{v}"),
            "player_in_id": s["canonical_player_in_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "player_out_id": s["canonical_player_out_id"].map(lambda v: None if pd.isna(v) else f"{self.league}:{v}"),
            "timing_quality": "exact",
        })
        ok = out[["event_seq", "elapsed_s", "team_id", "player_in_id", "player_out_id"]].notna().all(axis=1)
        self._meta["substitution_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy(); out["event_seq"] = out["event_seq"].astype(int)
        out = out.drop_duplicates(["game_id", "event_seq"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:substitutions:{self.family}").reset_index(drop=True)

    def _lineup_stint(self, con, tip: pd.Series, rules: pd.DataFrame, games_raw: pd.DataFrame) -> pd.DataFrame:
        src = FAMILY_SOURCE[self.family]
        if src == "espn":
            return pd.DataFrame()
        s = self._q(con, f"""
            SELECT canonical_game_id, source_game_id, season, team_id, period, stint_index, start_clock, end_clock, n_players_on_floor,
                   canonical_player_1, canonical_player_2, canonical_player_3, canonical_player_4, canonical_player_5, possessions, points_for, points_against
            FROM lineups WHERE league = '{self.league}' AND source = '{src}' AND lineup_grain = 'stint' AND {self._season_filter()}""")
        if not len(s):
            return pd.DataFrame()
        s["game_id"] = self._gid(s["source_game_id"])
        s = s[s["game_id"].isin(tip.index)].copy()
        st = self._elapsed(s["period"], self._clock_to_s(s["start_clock"]), s["season"], rules)
        en = self._elapsed(s["period"], self._clock_to_s(s["end_clock"]), s["season"], rules)
        ids = s[[f"canonical_player_{i}" for i in range(1, 6)]]
        out = pd.DataFrame({
            "game_id": s["game_id"], "team_id": self._tid(s["team_id"]),
            "stint_seq": pd.to_numeric(s["stint_index"], errors="coerce").astype("Int64"),
            "start_elapsed_s": st, "end_elapsed_s": en,
            "player_ids": [tuple(f"{self.league}:{v}" for v in row if pd.notna(v)) for row in ids.to_numpy()],
            "off_possessions": pd.to_numeric(s["possessions"], errors="coerce").astype("Int64"),
            "pts_for": pd.to_numeric(s["points_for"], errors="coerce").astype("Int64"),
            "pts_against": pd.to_numeric(s["points_against"], errors="coerce").astype("Int64"),
        })
        ok = out["stint_seq"].notna() & out["start_elapsed_s"].notna() & out["end_elapsed_s"].notna() & (out["player_ids"].map(len) == 5)
        self._meta["lineup_stint_rows_dropped_incomplete"] = int((~ok).sum())
        out = out[ok].copy(); out["stint_seq"] = out["stint_seq"].astype(int)
        out = out.drop_duplicates(["game_id", "team_id", "stint_seq"], keep="first")
        kt = out["game_id"].map(tip) + pd.Timedelta(hours=POST_GAME_HOURS)
        return self._common(out, kt, "imputed_conservative", f"corpus_v102:lineups:{self.family}").reset_index(drop=True)

    def _starting_lineup(self, pg: pd.DataFrame) -> pd.DataFrame:
        st = pg[pg["started"].astype(bool)][["game_id", "team_id", "player_id", "knowledge_time"]].copy()
        st["knowledge_time_quality"] = "imputed_conservative"
        st["source"] = f"corpus_v102:realised_starters:{self.family}"
        return st.reset_index(drop=True)

    # ------------------------------------------------------------------ provenance
    def provenance(self) -> dict:
        return {
            "adapter": self.name, "corpus_package": self._meta.get("package"), "corpus_built_at_utc": self._meta.get("built_at_utc"),
            "db_path": self.db_path, "league": self.league, "family": self.family, "seasons": self.seasons, "level": self.level,
            "knowledge_time": {
                "GAME v1": f"scheduled_tip - {SCHEDULE_LEAD_DAYS} d (imputed_conservative): the corpus holds the final schedule only",
                "GAME v2 / TEAM_GAME / PLAYER_GAME / EVENT / POSSESSION / SHOT / SUBSTITUTION / LINEUP_STINT":
                    f"scheduled_tip + {POST_GAME_HOURS} h (contract fallback box_score_or_event_unknown)",
                "STARTING_LINEUP": "REALISED starters at tip + 4 h; no announcement exists, never visible pre-game",
                "ROSTER": f"spells from game-day appearances; known {POST_GAME_HOURS} h after the first appearance; closed on a team change or after {ROSTER_GAP_GAMES} team games without the player",
                "PLAYER / TEAM / SEASON / RULESET": "season start (bounded); PLAYER bio is the corpus's rule-resolved single version (no versions exist)",
                "scheduled_tip": "exact UTC instant where the source carries one (ESPN); date-only sources get 12:00 UTC on the game date (tip_quality column)",
            },
            "excluded_seasons_low_minutes_coverage": self._excluded_seasons,
            "rows_dropped": {k: v for k, v in self._meta.items() if k.endswith("_dropped_incomplete")},
            "not_available_in_release": ["INJURY_REPORT", "COACH_TENURE", "OFFICIAL_ASSIGNMENT", "MARKET_SNAPSHOT", "VENUE.tz",
                                         "PERSON_LINK (cross-league)", "MATCHUP", "TRACKING_FRAME", "transaction announcements"],
        }
