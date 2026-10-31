"""As-of strata for stratified evaluation (H6, H9).

Every stratum flag is computed from the AsOfView at the forecast time, never from
outcomes or hindsight (leakage checklist L-E3). Pre-registered definitions:

  teammate_star_out   a top-2 teammate (by as-of points per appearance, >= 5 appearances)
                      has latest visible status 'out' for this game
  roster_change_21d   the player's team has a roster spell that started in the last 21
                      days (a trade/signing visible at forecast time), incl. the player's own
  own_move_21d        the player's own current spell started within 21 days
  low_sample          the player has < 10 as-of appearances
  role_change_alarm   a CUSUM usage alarm fired within the player's last 10 appearances
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..asof import AsOfStore
from ..diagnostics.changepoint import cusum_flags
from ..features import latest_injury_status, player_history


def compute_strata(store: AsOfStore, requests: pd.DataFrame, window_days: int = 21) -> pd.DataFrame:
    out = []
    for t, req in requests.groupby("forecast_time"):
        view = store.view(t)
        h = player_history(view)
        app = h[h["seconds"] > 0]
        ppg = app.groupby("player_id")["pts"].agg(["mean", "count"])
        ppg = ppg[ppg["count"] >= 5]["mean"]
        status = latest_injury_status(view, req["game_id"].unique())
        out_set = set(zip(status.loc[status["status"] == "out", "game_id"],
                          status.loc[status["status"] == "out", "player_id"]))
        r = view.table("ROSTER")
        tip = req["tip"].iloc[0]
        recent = r[(r["valid_from"] > tip - pd.Timedelta(days=window_days)) & (r["valid_from"] <= tip)]
        changed_teams = set(recent["team_id"])
        moved = set(recent["player_id"])
        n_app = app.groupby("player_id").size()
        alarms = {}
        for pid in req["player_id"].unique():
            g = app[app["player_id"] == pid]
            if len(g) >= 12:
                f = cusum_flags(g)
                alarms[pid] = bool(f["alarm"].iloc[-10:].any())
        rows = req[["forecast_time", "game_id", "player_id", "team_id"]].copy()
        star_out = []
        for (gid, tm), grp in req.groupby(["game_id", "team_id"]):
            mates = grp["player_id"].tolist()
            top = ppg.reindex(mates).dropna().sort_values(ascending=False).index[:2].tolist()
            for pid in mates:
                star_out.append(((gid, pid), any((gid, s) in out_set for s in top if s != pid)))
        so = dict(star_out)
        rows["teammate_star_out"] = [so.get((g, p), False) for g, p in zip(rows["game_id"], rows["player_id"])]
        rows["roster_change_21d"] = rows["team_id"].isin(changed_teams).to_numpy()
        rows["own_move_21d"] = rows["player_id"].isin(moved).to_numpy()
        rows["low_sample"] = rows["player_id"].map(n_app).fillna(0).to_numpy() < 10
        rows["role_change_alarm"] = rows["player_id"].map(alarms).fillna(False).astype(bool).to_numpy()
        out.append(rows)
    return pd.concat(out, ignore_index=True)
