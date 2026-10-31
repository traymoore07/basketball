"""Forecast targets and conditioning modes (docs/prereg/08_metric_spec.md).

Player targets are integer counts over the FULL game including overtime.

Conditioning modes (never silently mixed; every score row carries its mode):
  U  unconditional: every player on the as-of roster at forecast time. A player who
     does not appear has 0 for every stat. The forecast is p_appear * cond + (1 - p_appear) * delta_0.
  A  conditional on appearing (seconds > 0). Scored only on player-games that appeared.
  S  conditional on starting (scored on starters; requires STARTING_LINEUP or outcome).
  L  conditional on announced lineups for both teams (forecast issued at lineup lock).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER_TARGETS = {
    # name: (k_max, function of a PLAYER_GAME-like frame)
    "MIN": (70, lambda d: np.rint(d["seconds"] / 60.0)),
    "PTS": (90, lambda d: d["pts"]),
    "REB": (40, lambda d: d["oreb"] + d["dreb"]),
    "AST": (30, lambda d: d["ast"]),
    "FG3M": (16, lambda d: d["fg3m"]),
    "TOV": (15, lambda d: d["tov"]),
    "STL": (12, lambda d: d["stl"]),
    "BLK": (14, lambda d: d["blk"]),
    "PRA": (140, lambda d: d["pts"] + d["oreb"] + d["dreb"] + d["ast"]),
    "PR": (120, lambda d: d["pts"] + d["oreb"] + d["dreb"]),
    "PA": (110, lambda d: d["pts"] + d["ast"]),
}
CORE_TARGETS = ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]
MODES = ["U", "A", "S", "L"]

# Pre-registered bins for the ordinal (RPS) view of selected targets.
BINS = {
    "PTS": [10, 15, 20, 25, 30, 35],
    "REB": [4, 7, 10, 13],
    "AST": [3, 5, 7, 10],
    "PRA": [20, 25, 30, 35, 40, 45, 50],
    "FG3M": [1, 2, 3, 4, 5],
    "MIN": [10, 20, 25, 30, 35, 40],
    "TOV": [1, 2, 3, 4, 5],
}

TEAM_TARGETS = {
    "TEAM_PTS": 200,
    "TOTAL": 380,
    "POSSESSIONS": 160,
}
MARGIN_OFFSET = 80  # margin pmf support is [-80, 80] stored as index margin + 80
MARGIN_K = 160


def player_outcomes(pg: pd.DataFrame) -> pd.DataFrame:
    """Compute all player targets (unconditional: non-appearance -> 0) from PLAYER_GAME rows."""
    out = pg[["game_id", "player_id", "team_id"]].copy()
    out["appeared"] = pg["seconds"].to_numpy() > 0
    for name, (k, fn) in PLAYER_TARGETS.items():
        out[name] = np.minimum(np.asarray(fn(pg), dtype=float), k).astype(int)
    return out


def sim_to_targets(stats: np.ndarray, stat_index: dict) -> dict[str, np.ndarray]:
    """Map engine box samples (..., n_stats) to target samples."""
    g = lambda s: stats[..., stat_index[s]]  # noqa: E731
    return {
        "MIN": np.rint(g("SEC") / 60.0),
        "PTS": g("PTS"),
        "REB": g("OREB") + g("DREB"),
        "AST": g("AST"),
        "FG3M": g("FG3M"),
        "TOV": g("TOV"),
        "STL": g("STL"),
        "BLK": g("BLK"),
        "PRA": g("PTS") + g("OREB") + g("DREB") + g("AST"),
        "PR": g("PTS") + g("OREB") + g("DREB"),
        "PA": g("PTS") + g("AST"),
    }
