"""Provider-agnostic input contract for the prospective shadow lane (docs/shadow/01_input_contract.md).

The executor supplies a SNAPSHOT: a set of plain tables (pandas frames or CSV/parquet files) for one
information cutoff. Nothing in this module reads any data source; it only defines and validates the
shapes. The research repository therefore depends on no website or API.

Tables (column -> dtype; every timestamp is UTC; knowledge_time is when the executor's data path
first held the row, never later than the cutoff for rows that enter a forecast):

GAME        game_id, league, season_id, phase, scheduled_tip, home_team_id, away_team_id, status,
            knowledge_time            (one row per game known at the cutoff; the forecast games plus
                                       every game that appears in the appearance history)
PLAYER_GAME game_id, player_id, team_id, seconds, pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb,
            ast, stl, blk, tov, pf, knowledge_time
                                      (completed games only; one row per (game, player); the
                                       executor's whole history window, league-wide, see HISTORY_POLICY)
ELIGIBLE    game_id, player_id, team_id, availability_state, status_source, status_time
                                      (the players the executor wants conditional forecasts for, with the
                                       availability state known at the cutoff; OUT players may be listed
                                       and receive no distribution)

availability_state is one of AvailabilityState. The model never converts a state into an appearance
probability: mode A forecasts are P(stat | the player appears) for every listed player, and the state
only decides which prospective evaluation population the forecast belongs to (docs/shadow/01, scoring
policy).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd


class AvailabilityState(str, Enum):
    CONFIRMED_ACTIVE = "CONFIRMED_ACTIVE"      # publicly confirmed active by the cutoff
    EXPECTED_ACTIVE = "EXPECTED_ACTIVE"        # expected to play, not formally confirmed
    QUESTIONABLE_UNKNOWN = "QUESTIONABLE_UNKNOWN"  # listed questionable / doubtful / no information
    OUT = "OUT"                                # ruled out by the cutoff: listed, no distribution


FORECASTABLE_STATES = {AvailabilityState.CONFIRMED_ACTIVE, AvailabilityState.EXPECTED_ACTIVE,
                       AvailabilityState.QUESTIONABLE_UNKNOWN}

GAME_COLUMNS = ["game_id", "league", "season_id", "phase", "scheduled_tip", "home_team_id", "away_team_id", "status",
                "knowledge_time"]
PLAYER_GAME_COLUMNS = ["game_id", "player_id", "team_id", "seconds", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                       "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "knowledge_time"]
ELIGIBLE_COLUMNS = ["game_id", "player_id", "team_id", "availability_state", "status_source", "status_time"]
STAT_COLUMNS = ["seconds", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]

HISTORY_POLICY = ("league-wide completed player-games with knowledge_time <= cutoff, from the executor's history start "
                  "(the P1 reference history starts 1997-10-01; the league pmf fallback and the negative-binomial "
                  "dispersion are fitted on everything supplied, so the history start is part of data_snapshot_id)")


@dataclass
class Snapshot:
    """Validated, cutoff-truncated input tables plus the manifest that identifies them."""
    cutoff: pd.Timestamp
    game: pd.DataFrame
    player_game: pd.DataFrame
    eligible: pd.DataFrame
    manifest: dict


class SnapshotProvider(Protocol):
    """What an executor implements. Each method returns rows as the executor's data path knew them at
    `cutoff`; rows with knowledge_time > cutoff are dropped again by the snapshot builder (and counted)."""

    provider_id: str

    def schedule(self, cutoff: pd.Timestamp) -> pd.DataFrame: ...          # GAME rows (forecast games + history games)

    def appearance_history(self, cutoff: pd.Timestamp) -> pd.DataFrame: ...  # PLAYER_GAME rows, completed games only

    def eligible_players(self, cutoff: pd.Timestamp, game_ids: list[str]) -> pd.DataFrame: ...  # ELIGIBLE rows


class OutcomeProvider(Protocol):
    """Used ONLY by the scoring step, after the game is final. Never passed to the forecaster."""

    def final_box_score(self, game_id: str) -> pd.DataFrame: ...  # PLAYER_GAME rows of the finished game (+ result_time)


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def validate_tables(game: pd.DataFrame, player_game: pd.DataFrame, eligible: pd.DataFrame) -> list[str]:
    """Return a list of contract violations (empty = valid). Pure checks, no repair."""
    errs = []
    for name, df, cols in (("GAME", game, GAME_COLUMNS), ("PLAYER_GAME", player_game, PLAYER_GAME_COLUMNS),
                           ("ELIGIBLE", eligible, ELIGIBLE_COLUMNS)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errs.append(f"{name}: missing columns {missing}")
    if errs:
        return errs
    if game["game_id"].duplicated().any():
        errs.append("GAME: duplicate game_id (supply one row per game as known at the cutoff)")
    if player_game.duplicated(["game_id", "player_id"]).any():
        errs.append("PLAYER_GAME: duplicate (game_id, player_id) rows; de-duplicate deterministically before supplying")
    if eligible.duplicated(["game_id", "player_id"]).any():
        errs.append("ELIGIBLE: duplicate (game_id, player_id)")
    bad_state = set(eligible["availability_state"].astype(str)) - {s.value for s in AvailabilityState}
    if bad_state:
        errs.append(f"ELIGIBLE: unknown availability_state values {sorted(bad_state)}")
    if not set(eligible["game_id"]).issubset(set(game["game_id"])):
        errs.append("ELIGIBLE: game_id not present in GAME")
    if not set(player_game["game_id"]).issubset(set(game["game_id"])):
        errs.append("PLAYER_GAME: game_id not present in GAME (every history game needs its GAME row for the tip time)")
    for c in STAT_COLUMNS:
        if player_game[c].isna().any():
            errs.append(f"PLAYER_GAME: NULL in {c}; a completed game must carry every stat (never impute)")
    if (player_game["seconds"] < 0).any():
        errs.append("PLAYER_GAME: negative seconds")
    for name, df in (("GAME", game), ("PLAYER_GAME", player_game)):
        if df["knowledge_time"].isna().any():
            errs.append(f"{name}: NULL knowledge_time")
    return errs
