"""R1 SEASON-CARRYOVER: the frozen L2 (PlayerMean, half-life 5 appearances) with the historical weight
multiplied by a season-transition factor and a context-similarity factor (research_season/01_carryover_spec.md,
frozen 2026-11-02, C25). Grid point A reproduces the frozen L2 exactly (tests/test_carryover.py).

    w_i = recency(age_i) x transition_i
    recency(age) = 0.5 ** (age / 5)                                  (frozen, hoopslab.features.ew_weights)
    transition_i = m ** b_i, b_i = season boundaries between appearance i and the forecast game
      A  m = 1
      B  m = 0  (hard reset; with n_cur = 0 the frozen league-pmf fallback applies)
      C  m = c
      D  m = max(floor, rho ** n_cur)
      E  m = D's value x kappa_team^[team changed] x kappa_role^[role changed] x kappa_turnover^[high turnover]

The frozen baselines module is not modified; this class recomputes the per-player weighted means for every
grid point in one pass over the same appearance histories, fits the frozen negative-binomial dispersion per
grid point and target, and emits one PlayerForecast per grid point. Mode A (conditional on appearance) is
the only mode scored in R1; p_appear is the frozen history rate (reported, never used).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..diagnostics.changepoint import cusum_flags
from ..features import ew_weights, player_history, roster_at
from ..forecast import PlayerForecast
from ..targets import PLAYER_TARGETS, player_outcomes
from .baselines import fit_nb_dispersion, minutes_pmf, nb_pmf_matrix

HALF_LIFE = 5.0
TARGETS = ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]
# frozen thresholds (spec section 2)
PREV_STARTER_WINDOW = 10
N_CUR_FOR_ROLE = 3
LOW_MINUTES_PER_APP = 12.0
TURNOVER_RETAINED = 0.60


def grid() -> list[dict]:
    """The 44 frozen grid points, in the spec's order; ids are stable strings used as model names."""
    pts = [{"id": "A", "family": "A"}, {"id": "B", "family": "B"}]
    for c in (0.10, 0.25, 0.50, 0.75):
        pts.append({"id": f"C_c{c:g}", "family": "C", "c": c})
    for rho, floor in itertools.product((0.50, 0.70, 0.85), (0.0, 0.10)):
        pts.append({"id": f"D_rho{rho:g}_f{floor:g}", "family": "D", "rho": rho, "floor": floor})
    for rho, floor, kt, kr, ku in itertools.product((0.50, 0.85), (0.0, 0.10), (0.25, 0.50), (0.25, 0.50), (0.50, 1.00)):
        pts.append({"id": f"E_rho{rho:g}_f{floor:g}_kt{kt:g}_kr{kr:g}_ku{ku:g}", "family": "E", "rho": rho, "floor": floor,
                    "kappa_team": kt, "kappa_role": kr, "kappa_turnover": ku})
    assert len(pts) == 44
    return pts


GRID = grid()
GRID_IDS = [g["id"] for g in GRID]


def season_ordinal(season_id) -> int:
    """nba-2010 -> 2010; works for any '<league>-<year>' id."""
    return int(str(season_id).rsplit("-", 1)[-1])


def multipliers(n_cur: int, team_changed: bool, role_changed: bool, high_turnover: bool) -> np.ndarray:
    """m per grid point (shape 44) for ONE player at one forecast time; transition_i = m ** b_i."""
    m = np.ones(len(GRID))
    for j, g in enumerate(GRID):
        f = g["family"]
        if f == "A":
            m[j] = 1.0
        elif f == "B":
            m[j] = 0.0
        elif f == "C":
            m[j] = g["c"]
        else:
            v = max(g["floor"], g["rho"] ** n_cur)
            if f == "E":
                v *= (g["kappa_team"] if team_changed else 1.0) * (g["kappa_role"] if role_changed else 1.0) * \
                     (g["kappa_turnover"] if high_turnover else 1.0)
            m[j] = v
    return m


def weights_matrix(n_app: int, boundaries: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(44, n_app) weights: recency (oldest first, as ew_weights) x m ** boundaries."""
    rec = ew_weights(n_app, HALF_LIFE)
    b = np.asarray(boundaries, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(b[None, :] == 0, 1.0, np.power(m[:, None], b[None, :]))
    return rec[None, :] * tr


class SeasonAwareMean:
    """All grid points of the R1 chain in one fit. fit(view, current_season, requests) then predict()."""

    name = "R1_season_aware_grid"

    def __init__(self, grid_ids: list[str] | None = None):
        self.grid_ids = grid_ids or GRID_IDS
        self.idx = [GRID_IDS.index(g) for g in self.grid_ids]

    # ---- context (as-of, spec section 2) --------------------------------------------------------------
    def _context(self, view, app: pd.DataFrame, requests: pd.DataFrame, cur: int, tip) -> pd.DataFrame:
        prev = cur - 1
        rows = []
        ap = app[app["player_id"].isin(set(requests["player_id"]))]
        prev_app = app[app["season_ord"] == prev]
        team_prev_minutes = prev_app.groupby(["team_id", "player_id"])["seconds"].sum()
        team_prev_total = prev_app.groupby("team_id")["seconds"].sum()
        roster_cache = {}
        groups = {p: g for p, g in ap.groupby("player_id", sort=False)}
        for r in requests.itertuples(index=False):
            pid, team = r.player_id, r.team_id
            g = groups.get(pid)
            n_cur = int((g["season_ord"] == cur).sum()) if g is not None else 0
            gp = g[g["season_ord"] == prev] if g is not None else None
            has_prev = gp is not None and len(gp) > 0
            team_changed = (not has_prev) or (gp["team_id"].iloc[-1] != team)
            # team turnover: share of the team's previous-season minutes played by players on its current roster
            if team not in roster_cache:
                ros = set(roster_at(view, [team], tip)["player_id"])
                tot = float(team_prev_total.get(team, 0.0))
                kept = float(team_prev_minutes.loc[team].reindex(list(ros)).fillna(0).sum()) if team in team_prev_total.index else 0.0
                roster_cache[team] = (kept / tot) if tot > 0 else np.nan
            retained = roster_cache[team]
            high_turnover = bool(np.isnan(retained) or retained < TURNOVER_RETAINED)
            # role
            if has_prev:
                prev_share = float(gp["started"].astype(bool).to_numpy()[-PREV_STARTER_WINDOW:].mean())
                prev_mpa = float(gp["seconds"].mean() / 60.0)
            else:
                prev_share, prev_mpa = np.nan, np.nan
            starter_prev = bool(has_prev and prev_share >= 0.5)
            if n_cur >= N_CUR_FOR_ROLE:
                cur_share = float(g[g["season_ord"] == cur]["started"].astype(bool).mean())
                role_changed = bool(has_prev and ((prev_share >= 0.5) != (cur_share >= 0.5)))
            else:
                role_changed = bool(has_prev and prev_mpa < LOW_MINUTES_PER_APP and high_turnover)
            if not role_changed and g is not None and len(g) >= 12:
                f = cusum_flags(g)
                role_changed = bool(f["alarm"].iloc[-10:].any())
            rows.append((pid, team, n_cur, has_prev, team_changed, role_changed, high_turnover, starter_prev, retained))
        return pd.DataFrame(rows, columns=["player_id", "team_id", "n_cur", "has_prev_season", "team_changed", "role_changed",
                                           "high_turnover", "starter_prev", "team_minutes_retained"])

    # ---- fit ---------------------------------------------------------------------------------------------
    def fit(self, view, current_season: int, requests: pd.DataFrame, tip):
        h = player_history(view)
        out = player_outcomes(h)
        out["seconds"] = h["seconds"].to_numpy()
        for c in ("fga", "fta", "tov"):          # raw columns the frozen CUSUM usage alarm reads
            out[c] = h[c].to_numpy()
        out["season_ord"] = h["season_id"].map(season_ordinal).to_numpy()
        out["started"] = h["started"].to_numpy() if "started" in h else False
        app = out[out["appeared"]].reset_index(drop=True)
        self.app = app
        self.league_pmf = {}
        for t, (k, _) in PLAYER_TARGETS.items():
            v = np.bincount(np.minimum(app[t].to_numpy(), k), minlength=k + 1).astype(float) + 0.01
            self.league_pmf[t] = v / v.sum()
        self.league_appear = float(out["appeared"].mean()) if len(out) else 0.85
        self.appear_counts = out.groupby("player_id")["appeared"].agg(["sum", "count"])
        self.ctx = self._context(view, app, requests, current_season, tip).set_index(["player_id", "team_id"])
        # weighted means for EVERY player with history (the dispersion is fitted over all appearances, as L2 does),
        # at the context of the request where the player is requested, else at the neutral context (A-like m = 1
        # only matters for transition families; a non-requested player's m uses n_cur and no changes)
        G = len(GRID)
        means = {}   # pid -> (G, 7)
        Y = app[TARGETS].to_numpy(float)
        cur = current_season
        ctx_by_pid = {}
        for (pid, _team), c in self.ctx.iterrows():
            ctx_by_pid[pid] = c
        for pid, idx in app.groupby("player_id", sort=False).indices.items():
            y = Y[idx]
            b = cur - app["season_ord"].to_numpy()[idx]
            c = ctx_by_pid.get(pid)
            if c is not None:
                m = multipliers(int(c["n_cur"]), bool(c["team_changed"]), bool(c["role_changed"]), bool(c["high_turnover"]))
            else:
                m = multipliers(int((b == 0).sum()), False, False, False)
            W = weights_matrix(len(idx), b, m)
            ws = W.sum(1)
            mu = (W @ y) / np.where(ws > 0, ws, np.nan)[:, None]
            means[pid] = mu   # NaN row = no usable history under that grid point (hard reset before the first game)
        self.means = means
        # frozen dispersion per grid point and target, over all visible appearances with their own means
        pids = app["player_id"].to_numpy()
        M = np.stack([means[p] for p in pids])  # (n_app, G, 7)
        self.r = np.full((G, len(TARGETS)), 10.0)
        if len(app) >= 50:
            for j in range(G):
                for ti, t in enumerate(TARGETS):
                    mu = M[:, j, ti]
                    ok = ~np.isnan(mu)
                    self.r[j, ti] = fit_nb_dispersion(Y[ok, ti], mu[ok]) if ok.sum() >= 50 else 10.0
        return self

    def _p_appear(self, pids):
        c = self.appear_counts.reindex(pids)
        s = c["sum"].fillna(0).to_numpy()
        n = c["count"].fillna(0).to_numpy()
        return (s + 5 * self.league_appear) / (n + 5)

    # ---- predict: one PlayerForecast per grid point ------------------------------------------------------
    def predict(self, view, requests: pd.DataFrame, targets=None) -> dict[str, PlayerForecast]:
        targets = targets or TARGETS
        pids = requests["player_id"].to_numpy()
        n = len(pids)
        keys = requests[["game_id", "player_id", "team_id"]].reset_index(drop=True)
        pa = self._p_appear(pids)
        out = {}
        for j in self.idx:
            cond = {}
            for t in targets:
                ti = TARGETS.index(t)
                k = PLAYER_TARGETS[t][0]
                mu = np.array([self.means[p][j, ti] if p in self.means else np.nan for p in pids])
                known = ~np.isnan(mu)
                pm = np.repeat(self.league_pmf[t][None], n, 0)
                if known.any():
                    if t == "MIN":
                        pm[known] = minutes_pmf(mu[known], np.full(known.sum(), 6.0), k)
                    else:
                        pm[known] = nb_pmf_matrix(mu[known], self.r[j, ti], k)
                cond[t] = pm
            out[GRID_IDS[j]] = PlayerForecast(GRID_IDS[j], view.t, keys, pa, cond)
        return out
