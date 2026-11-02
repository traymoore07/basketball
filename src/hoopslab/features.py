"""As-of feature helpers. Every function takes an AsOfView, so nothing unknown at
view.t can enter a feature. (A test recomputes features on a physically truncated
store and requires identical output.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import AsOfView


def game_schedule(view: AsOfView) -> pd.DataFrame:
    g = view.table("GAME")
    return g[["game_id", "scheduled_tip", "home_team_id", "away_team_id", "season_id", "status"]]


def player_history(view: AsOfView) -> pd.DataFrame:
    """Visible player-game box scores joined to tip time, sorted by (player, tip)."""
    if "player_history" in view.cache:
        return view.cache["player_history"]
    pg = view.table("PLAYER_GAME")
    g = view.table("GAME")[["game_id", "scheduled_tip", "season_id"]].drop_duplicates("game_id", keep="last")
    h = pg.merge(g, on="game_id", how="left")
    h = h.sort_values(["player_id", "scheduled_tip"], kind="stable").reset_index(drop=True)
    view.cache["player_history"] = h
    return h


def roster_at(view: AsOfView, team_ids, at: pd.Timestamp) -> pd.DataFrame:
    """Players whose visible roster spell covers `at` (valid_from <= at < valid_to)."""
    r = view.table("ROSTER")
    vt = r["valid_to"]
    m = r["team_id"].isin(list(team_ids)) & (r["valid_from"] <= at) & (vt.isna() | (vt > at))
    return r.loc[m, ["player_id", "team_id", "depth_chart_slot"] if "depth_chart_slot" in r else
                 ["player_id", "team_id"]]


def latest_injury_status(view: AsOfView, game_ids) -> pd.DataFrame:
    """Latest visible availability status per (game, player); missing = no report."""
    if not view.has("INJURY_REPORT"):
        return pd.DataFrame(columns=["game_id", "player_id", "status"])
    ir = view.table("INJURY_REPORT")
    ir = ir[ir["game_id"].isin(list(game_ids))]
    return ir.sort_values("knowledge_time", kind="stable").drop_duplicates(["game_id", "player_id"], keep="last")[
        ["game_id", "player_id", "status"]]


def ew_weights(n: int, half_life: float) -> np.ndarray:
    """Exponential weights for the last n observations (oldest first)."""
    if half_life is None or np.isinf(half_life):
        return np.ones(n)
    age = np.arange(n)[::-1]
    return 0.5 ** (age / half_life)
