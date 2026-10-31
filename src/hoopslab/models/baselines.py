"""Lower rungs of the baseline ladder (registry/baseline_ladder.yaml).

L0 Climatology            league pmf per target, league appearance rate
L1 PlayerMean             player's all-history mean, negative binomial
L2 RecentMean             exponentially weighted player mean, negative binomial
L3 MinutesRate            availability-aware minutes pmf x per-minute rates (compound NB)
L4 HierShrink             L3 with empirical-Bayes shrinkage of rates (gamma-Poisson) and
                          optional posterior-predictive parameter uncertainty

These are deliberately simple, transparent, and cheap. A complex model gets credit
only for improvement beyond the best of these at the same information level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from ..features import ew_weights, latest_injury_status, player_history
from ..forecast import PlayerForecast
from ..targets import PLAYER_TARGETS, player_outcomes
from .base import Forecaster

RATE_TARGETS = ["PTS", "REB", "AST", "FG3M", "TOV", "STL", "BLK", "PRA", "PR", "PA"]


def nb_pmf_fast(mean: np.ndarray, r: np.ndarray, k_max: int) -> np.ndarray:
    """Negative binomial pmf over 0..k_max for broadcastable (mean, r); shape (..., k_max+1).

    Log-space recurrence: log p(k) = r log q + sum_{j<=k} log((j-1+r)/j) + k log(1-q), q = r/(r+mean).
    """
    mean = np.maximum(np.asarray(mean, float), 1e-9)
    r = np.broadcast_to(np.asarray(r, float), mean.shape)
    q = r / (r + mean)
    ks = np.arange(1, k_max + 1, dtype=float)
    steps = np.log((ks - 1.0 + r[..., None]) / ks) + np.log1p(-q)[..., None]
    logp = np.concatenate([np.zeros(mean.shape + (1,)), np.cumsum(steps, axis=-1)], axis=-1)
    logp += (r * np.log(q))[..., None]
    pm = np.exp(logp)
    pm[..., -1] += np.clip(1.0 - pm.sum(-1), 0, None)  # tail mass into the top bin
    return pm / pm.sum(-1, keepdims=True)


def nb_pmf_matrix(mean: np.ndarray, r: float | np.ndarray, k_max: int) -> np.ndarray:
    mean = np.asarray(mean, float)
    return nb_pmf_fast(mean, np.broadcast_to(np.asarray(r, float), mean.shape), k_max)


def minutes_pmf(mu: np.ndarray, sd: np.ndarray, k_max: int) -> np.ndarray:
    ks = np.arange(k_max + 1)
    edges_lo = ks - 0.5
    edges_hi = ks + 0.5
    edges_lo[0] = -np.inf
    edges_hi[-1] = np.inf
    sd = np.maximum(sd, 1.0)
    pm = norm.cdf(edges_hi[None, :], mu[:, None], sd[:, None]) - norm.cdf(edges_lo[None, :], mu[:, None], sd[:, None])
    return pm / pm.sum(1, keepdims=True)


def fit_nb_dispersion(y: np.ndarray, mu: np.ndarray, lo: float = 1.5, hi: float = 500.0) -> float:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    excess = np.sum((y - mu) ** 2 - mu)
    if excess <= 0:
        return hi
    return float(np.clip(np.sum(mu ** 2) / excess, lo, hi))


class _HistoryModel(Forecaster):
    """Shared machinery: per-player appearance histories from the as-of view."""

    def fit(self, view):
        h = player_history(view)
        out = player_outcomes(h)
        out["scheduled_tip"] = h["scheduled_tip"].to_numpy()
        out["seconds"] = h["seconds"].to_numpy()
        self.hist = out
        app = out[out["appeared"]]
        self.app = app
        self.league_pmf = {}
        for t, (k, _) in PLAYER_TARGETS.items():
            v = np.bincount(np.minimum(app[t].to_numpy(), k), minlength=k + 1).astype(float) + 0.01
            self.league_pmf[t] = v / v.sum()
        self.league_appear = float(out["appeared"].mean()) if len(out) else 0.85
        self.groups = {pid: g for pid, g in app.groupby("player_id", sort=False)}
        self.appear_counts = out.groupby("player_id")["appeared"].agg(["sum", "count"])
        self._fit_extra(view)

    def _fit_extra(self, view):
        pass

    def _p_appear_history(self, pids):
        c = self.appear_counts.reindex(pids)
        s = c["sum"].fillna(0).to_numpy()
        n = c["count"].fillna(0).to_numpy()
        return (s + 5 * self.league_appear) / (n + 5)


class Climatology(_HistoryModel):
    name, rung = "L0_climatology", "L0"

    def predict(self, view, requests, targets):
        n = len(requests)
        cond = {t: np.repeat(self.league_pmf[t][None], n, 0) for t in targets}
        return PlayerForecast(self.name, view.t, requests[["game_id", "player_id", "team_id"]].reset_index(drop=True),
                              np.full(n, self.league_appear), cond)


class PlayerMean(_HistoryModel):
    """L1 (half_life=None) and L2 (finite half-life, in appearances)."""

    def __init__(self, half_life: float | None = None, name: str | None = None, rung: str | None = None):
        self.half_life = half_life
        self.name = name or ("L1_player_mean" if half_life is None else f"L2_recent_hl{half_life:g}")
        self.rung = rung or ("L1" if half_life is None else "L2")

    def _fit_extra(self, view):
        self.means = {}
        for pid, g in self.groups.items():
            w = ew_weights(len(g), self.half_life)
            self.means[pid] = {t: float(np.sum(w * g[t].to_numpy()) / w.sum()) for t in PLAYER_TARGETS}
        # league dispersion around player means (in-sample, as-of)
        self.r = {}
        for t in PLAYER_TARGETS:
            if len(self.app) < 50:
                self.r[t] = 10.0
                continue
            mu = self.app["player_id"].map({p: m[t] for p, m in self.means.items()}).to_numpy()
            self.r[t] = fit_nb_dispersion(self.app[t].to_numpy(), mu)

    def predict(self, view, requests, targets):
        pids = requests["player_id"].to_numpy()
        n = len(pids)
        cond = {}
        for t in targets:
            k = PLAYER_TARGETS[t][0]
            known = np.array([p in self.means for p in pids])
            mu = np.array([self.means[p][t] if p in self.means else 0.0 for p in pids])
            pm = np.repeat(self.league_pmf[t][None], n, 0)
            if known.any():
                if t == "MIN":
                    pm[known] = minutes_pmf(mu[known], np.full(known.sum(), 6.0), k)
                else:
                    pm[known] = nb_pmf_matrix(mu[known], self.r[t], k)
            cond[t] = pm
        return PlayerForecast(self.name, view.t, requests[["game_id", "player_id", "team_id"]].reset_index(drop=True),
                              self._p_appear_history(pids), cond)


class MinutesRate(_HistoryModel):
    """L3 (shrink=False) / L4 (shrink=True): availability-aware minutes x per-minute rates.

    param_uncertainty=True integrates the gamma posterior of each rate (posterior
    predictive); False plugs in the posterior mean (the H9 contrast).
    """

    def __init__(self, half_life: float = 15.0, shrink: bool = False, param_uncertainty: bool = True,
                 name: str | None = None, rung: str | None = None, prior_strength_min: float = 150.0,
                 lead: pd.Timedelta = pd.Timedelta(minutes=60)):
        self.lead = lead
        self.half_life = half_life
        self.shrink = shrink
        self.param_uncertainty = param_uncertainty
        self.prior_strength_min = prior_strength_min
        base = "L4_hier_shrink" if shrink else "L3_minutes_rate"
        self.name = name or (base + ("" if param_uncertainty else "_plugin"))
        self.rung = rung or ("L4" if shrink else "L3")

    def _fit_extra(self, view):
        self.state = {}
        app = self.app
        mins = np.maximum(app["seconds"].to_numpy() / 60.0, 0.5)
        league_rate = {t: app[t].sum() / max(mins.sum(), 1) for t in RATE_TARGETS}
        self.league_rate = league_rate
        self.league_min = (float(app["MIN"].mean()), float(app["MIN"].std() or 8.0)) if len(app) else (20.0, 8.0)
        for pid, g in self.groups.items():
            w = ew_weights(len(g), self.half_life)
            m = np.maximum(g["seconds"].to_numpy() / 60.0, 0.5)
            wm = np.sum(w * m)
            st = {"n": len(g), "wmin": wm, "min_mu": float(np.sum(w * m) / w.sum()),
                  "min_sd": float(np.sqrt(np.sum(w * (m - np.sum(w * m) / w.sum()) ** 2) / w.sum()) + 2.0)}
            for t in RATE_TARGETS:
                st[t] = float(np.sum(w * g[t].to_numpy()))
            self.state[pid] = st
        # empirical-Bayes gamma prior per (listed position, target): mean and between-player
        # variance by method of moments (sampling variance removed), as-of data only
        pos = view.table("PLAYER").drop_duplicates("player_id", keep="last").set_index("player_id")["listed_position"] \
            if view.has("PLAYER") else pd.Series(dtype=str)
        self.pos = pos
        self.prior = {}
        pids = [p for p, s_ in self.state.items() if s_["wmin"] >= 60]
        for grp in list(pos.unique()) + ["__all__"]:
            members = [p for p in pids if grp == "__all__" or pos.get(p) == grp]
            for t in RATE_TARGETS:
                if len(members) < 5:
                    self.prior[(grp, t)] = None
                    continue
                y = np.array([self.state[p][t] for p in members])
                m = np.array([self.state[p]["wmin"] for p in members])
                rates = y / m
                mean = float(np.sum(y) / np.sum(m))
                var_obs = float(np.sum(m * (rates - mean) ** 2) / np.sum(m))
                samp = float(np.mean(mean / m))
                tau2 = max(var_obs - samp, (0.05 * mean) ** 2)
                self.prior[(grp, t)] = (mean ** 2 / tau2, mean / tau2)  # gamma (a0, b0)
        for t in RATE_TARGETS:
            if self.prior.get(("__all__", t)) is None:
                self.prior[("__all__", t)] = (league_rate[t] * self.prior_strength_min, self.prior_strength_min)
        # game-level overdispersion given minutes and rate (as-of residuals)
        self.r = {}
        for t in RATE_TARGETS:
            rate_i = app["player_id"].map({p: s[t] / max(s["wmin"], 1e-6) for p, s in self.state.items()}).to_numpy()
            mu = rate_i * mins
            self.r[t] = fit_nb_dispersion(app[t].to_numpy(), mu) if len(app) > 50 else 20.0
        # appearance rate by latest injury status (as-of, learned from history)
        self.status_rates = self._status_rates(view)

    def _status_rates(self, view):
        """Appearance rate by the status that was visible `lead` before each past tip.

        Calibrating on the FINAL status (which includes late scratches posted after the
        forecast offset) would be a subtle leak: see leakage checklist L-B6.
        """
        if not view.has("INJURY_REPORT") or len(self.hist) == 0:
            return {}
        ir = view.table("INJURY_REPORT")
        tips = self.hist.drop_duplicates("game_id").set_index("game_id")["scheduled_tip"]
        ir = ir[ir["game_id"].isin(tips.index)]
        ir = ir[ir["knowledge_time"].to_numpy() <= (ir["game_id"].map(tips) - self.lead).to_numpy()]
        st = ir.sort_values("knowledge_time", kind="stable").drop_duplicates(["game_id", "player_id"], keep="last")
        m = self.hist.merge(st[["game_id", "player_id", "status"]], on=["game_id", "player_id"], how="left")
        m["status"] = m["status"].fillna("none")
        return m.groupby("status")["appeared"].mean().to_dict()

    def predict(self, view, requests, targets):
        req = requests.reset_index(drop=True)
        pids = req["player_id"].to_numpy()
        n = len(pids)
        st = latest_injury_status(view, req["game_id"].unique())
        status = req.merge(st, on=["game_id", "player_id"], how="left")["status"].fillna("none").to_numpy()
        p_hist = self._p_appear_history(pids)
        p_app = np.array([self.status_rates.get(s, np.nan) for s in status])
        # blend: status-specific base rate adjusted by player's own tendency when no report
        none_rate = self.status_rates.get("none", self.league_appear)
        p_app = np.where(status == "none", np.clip(p_hist / max(none_rate, 1e-3) * none_rate, 0, 1), p_app)
        p_app = np.where(np.isnan(p_app), p_hist, p_app)
        p_app = np.clip(p_app, 0.0, 1.0)

        known = np.array([p in self.state for p in pids])
        mu_m = np.array([self.state[p]["min_mu"] if p in self.state else self.league_min[0] * 0.6 for p in pids])
        sd_m = np.array([self.state[p]["min_sd"] if p in self.state else self.league_min[1] for p in pids])
        kmin = PLAYER_TARGETS["MIN"][0]
        mpmf = minutes_pmf(mu_m, sd_m, kmin)
        cond = {}
        if "MIN" in targets:
            cond["MIN"] = mpmf
        grid = np.arange(kmin + 1).astype(float)
        grid[0] = 0.25
        for t in targets:
            if t == "MIN":
                continue
            k = PLAYER_TARGETS[t][0]
            pri = [self.prior.get((self.pos.get(p), t)) or self.prior[("__all__", t)] for p in pids]
            a0 = np.array([x[0] for x in pri])
            b0 = np.array([x[1] for x in pri])
            has = np.array([p in self.state for p in pids])
            ys = np.array([self.state[p][t] if p in self.state else 0.0 for p in pids])
            ms = np.array([self.state[p]["wmin"] if p in self.state else 0.0 for p in pids])
            if self.shrink:
                a, b = a0 + ys, b0 + ms
            else:  # raw rate with a vanishing prior (no pooling)
                eps = 1e-3
                a = np.where(has, ys + eps * a0, a0)
                b = np.where(has, ms + eps * b0, b0)
            lam = a / b
            var_lam = a / b ** 2 if self.param_uncertainty else np.zeros(n)
            # compound over the minutes grid; moment-matched NB per (player, minutes)
            mu = lam[:, None] * grid[None, :]
            var = mu + mu ** 2 / self.r[t] + var_lam[:, None] * grid[None, :] ** 2
            r_eff = np.clip(mu ** 2 / np.maximum(var - mu, 1e-9), 0.5, 1e4)
            # evaluate NB for each (player, minute) with non-negligible minutes mass, then mix
            used = mpmf.max(0) > 1e-7
            pm = nb_pmf_fast(mu[:, used], r_eff[:, used], k)
            pm = np.einsum("nm,nmk->nk", mpmf[:, used], pm)
            pm[:, -1] += np.clip(1 - pm.sum(1), 0, None)
            cond[t] = pm / pm.sum(1, keepdims=True)
        del known
        return PlayerForecast(self.name, view.t, req[["game_id", "player_id", "team_id"]], p_app, cond)
