"""Simulation realism checks (posterior predictive checks), run BEFORE trusting
simulator-derived player distributions.

For each held-out game, the simulator produces a predictive distribution of game-level
summaries. The realised summary gets a randomised PIT under that distribution. Across
many games, the PITs of a realistic simulator are Uniform(0,1). Per summary, the
pre-registered tolerances (docs/prereg/01 E-REALISM) are practical tolerances widened by
two sampling standard errors for n games, so that a correct simulator is not failed by
noise in small samples. At n ~ 1230 (one NBA season) the widening is ~0.02-0.05:

    |coverage(80%) - 0.80| <= 0.05 + 2*sqrt(0.16/n)
    dispersion ratio in [0.80 - 2*se_d, 1.25 + 2*se_d],  se_d = 0.894/sqrt(n)   (Var(PIT)*12; >1 too narrow)
    |mean(PIT) - 0.5|      <= 0.05 + 2*sqrt(1/(12 n))

Summaries (full-game unless noted): possessions, total points, margin, lead changes,
overtime indicator (reliability of P(OT)), team FTA, team OREB, team AST, team TOV,
starters' minutes, max player minutes, top-scorer points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOL = {"coverage80": 0.05, "disp_lo": 0.80, "disp_hi": 1.25, "mean": 0.05}


def rpit(samples: np.ndarray, y: float, rng) -> float:
    s = np.asarray(samples, float)
    return float(np.mean(s < y) + rng.random() * np.mean(s == y))


def realism_report(pits: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for name, v in pits.items():
        v = np.asarray(v)
        if len(v) < 20:
            continue
        cov = float(np.mean((v > 0.1) & (v < 0.9)))
        disp = float(np.var(v) * 12)
        mu = float(np.mean(v))
        n = len(v)
        se_c, se_d, se_m = np.sqrt(0.16 / n), 0.894 / np.sqrt(n), np.sqrt(1 / (12 * n))
        ok = abs(cov - 0.8) <= TOL["coverage80"] + 2 * se_c and \
            TOL["disp_lo"] - 2 * se_d <= disp <= TOL["disp_hi"] + 2 * se_d and \
            abs(mu - 0.5) <= TOL["mean"] + 2 * se_m
        rows.append({"summary": name, "n": len(v), "coverage80": cov, "dispersion": disp, "mean_pit": mu,
                     "passed": ok})
    return pd.DataFrame(rows)


def game_summary_pits(forecasts, team_games: pd.DataFrame, games: pd.DataFrame, rng) -> dict[str, list[float]]:
    """PITs of game-level summaries from simulator forecasts (PlayerForecast.team)."""
    tg = team_games.sort_values("knowledge_time").drop_duplicates(["game_id", "team_id"], keep="last")
    home = tg[tg["is_home"]].set_index("game_id")
    away = tg[~tg["is_home"].astype(bool)].set_index("game_id")
    per = games.sort_values("knowledge_time").drop_duplicates("game_id", keep="last").set_index("game_id")
    out = {k: [] for k in ["total_points", "margin", "possessions", "home_pts", "overtime", "lead_changes"]}
    for f in forecasts:
        for gid, T in f.team.items():
            if gid not in home.index:
                continue
            h, a = home.loc[gid], away.loc[gid]
            out["total_points"].append(rpit(T["home_pts"] + T["away_pts"], h["pts"] + a["pts"], rng))
            out["margin"].append(rpit(T["home_pts"] - T["away_pts"], h["pts"] - a["pts"], rng))
            out["home_pts"].append(rpit(T["home_pts"], h["pts"], rng))
            if "possessions" in h and not pd.isna(h.get("possessions")):
                out["possessions"].append(rpit(T["possessions"], h["possessions"] + a["possessions"], rng))
            out["overtime"].append(rpit(T["periods"] > 4, float(per.loc[gid, "periods_played"] > 4), rng))
            if "lead_changes" in T and "lead_changes" in per.columns:
                out["lead_changes"].append(rpit(T["lead_changes"], per.loc[gid, "lead_changes"], rng))
    return out
