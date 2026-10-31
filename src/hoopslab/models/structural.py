"""Reference structural (possession-level) simulator: ladder rungs L10-L12.

Estimation (as-of, from POSSESSION + ROSTER + INJURY_REPORT + PLAYER_GAME):
  * usage weights: first-choice Plackett-Luce ("attention") among the 5 on-court
    players, fitted by minorisation-maximisation with a pseudo-count prior;
  * per-player terminal-event rates (TO, FT trip, 3PA share, make probabilities,
    FT%) shrunk toward league rates;
  * rebound, assist, steal and block actor weights: the same choice model;
  * league start-type (transition) effects, home advantage, team defence;
  * rotation: depth chart plus slot minute shares from recent box scores;
  * availability: sampled per simulation from as-of status-conditional rates;
  * optional game-level shooting latent (variance estimated as-of) = L11;
  * optional posterior parameter sampling per simulation = L12.

This is an instrument-grade reference implementation for the synthetic suite and
a template for the real-data version. It is not a tuned forecasting system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import latest_injury_status, player_history
from ..forecast import PlayerForecast
from ..metrics.scoring import smoothed_pmf_from_samples
from ..sim.engine import GameBatch, SI, simulate
from ..synthetic.world import resolve_rotation
from ..targets import PLAYER_TARGETS, sim_to_targets
from .base import Forecaster

START_IDX = {"after_make": 0, "after_dreb": 1, "after_live_turnover": 2, "after_oreb": 3,
             "after_dead_ball": 0, "period_start": 0, "after_ft": 0}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mm_choice_weights(lineups: np.ndarray, chosen: np.ndarray, n_items: int, weights: np.ndarray | None = None,
                      mask: np.ndarray | None = None, prior: float = 3.0, iters: int = 25) -> np.ndarray:
    """First-choice Plackett-Luce / multinomial-logit weights by MM (Hunter 2004).

    lineups: (n, k) item indices available at each choice; chosen: (n,) item index;
    mask: optional (n, k) bool of eligible items (e.g. exclude the shooter for assists).
    A symmetric prior adds `prior` pseudo-choices of an average item.
    """
    n, k = lineups.shape
    om = np.ones(n) if weights is None else weights
    elig = np.ones((n, k), bool) if mask is None else mask
    wins = np.bincount(chosen, weights=om, minlength=n_items) + prior
    w = np.ones(n_items)
    for _ in range(iters):
        W = (w[lineups] * elig).sum(1)
        contrib = (om / W)[:, None] * elig
        denom = np.bincount(lineups.ravel(), weights=contrib.ravel(), minlength=n_items) + prior
        w = wins / denom
        w /= np.exp(np.mean(np.log(w[np.bincount(lineups.ravel(), minlength=n_items) > 0])))
    return w


class PossessionSimulator(Forecaster):
    data_level = "C"

    def __init__(self, n_sims: int = 400, game_latent: str = "none", param_uncertainty: bool = False,
                 half_life_days: float | None = 120.0, lead: pd.Timedelta = pd.Timedelta(minutes=60),
                 blowout_margin: float | None = 20.0, pace_scale: float = 1.0, seed: int = 0,
                 refit_days: float = 7.0, name: str | None = None, rung: str | None = None):
        self.refit_days = refit_days  # full re-estimation cadence; availability/rotation update every call
        self._last_full_fit = None
        self.n_sims = n_sims
        self.game_latent = game_latent
        self.param_uncertainty = param_uncertainty
        self.half_life_days = half_life_days
        self.lead = lead
        self.blowout_margin = blowout_margin
        self.pace_scale = pace_scale
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        tag = {("none", False): "L10_sim", ("estimate", False): "L11_sim_latent",
               ("estimate", True): "L12_sim_latent_postpred", ("none", True): "L10b_sim_postpred"}
        self.name = name or tag[(game_latent, param_uncertainty)]
        self.rung = rung or self.name.split("_")[0]

    # ------------------------------------------------------------------ fit
    def _encode(self, P: pd.DataFrame) -> dict:
        """Integer-encode actor columns, incrementally over the visible prefix.

        Views are prefixes of one knowledge_time-sorted table, so rows already encoded
        at an earlier forecast time are reused. A consistency check on the last cached
        row guards against a different underlying store (then the cache is rebuilt).
        """
        c = getattr(self, "_enc", None)
        if c is not None and (c["n"] > len(P) or (c["n"] > 0 and P["game_id"].iat[c["n"] - 1] != c["last_gid"])):
            c = None
        if c is None:
            self.pidx = {}
            c = {"n": 0, "last_gid": None, "OL": np.zeros((0, 5), int), "DL": np.zeros((0, 5), int),
                 **{k: np.zeros(0, int) for k in ("user", "reb", "ast", "stl", "blk")}}
        new = P.iloc[c["n"]:]
        if len(new):
            def code(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return -1
                if x not in self.pidx:
                    self.pidx[x] = len(self.pidx)
                return self.pidx[x]
            OLn = np.array([[code(p) for p in lu] for lu in new["off_lineup"]], int).reshape(-1, 5)
            DLn = np.array([[code(p) for p in lu] for lu in new["def_lineup"]], int).reshape(-1, 5)
            c["OL"] = np.concatenate([c["OL"], OLn])
            c["DL"] = np.concatenate([c["DL"], DLn])
            for k, col in (("user", "user_player_id"), ("reb", "rebounder_id"), ("ast", "assister_id"),
                           ("stl", "stealer_id"), ("blk", "blocker_id")):
                c[k] = np.concatenate([c[k], np.array([code(x) for x in new[col]], int)])
            c["n"] = len(P)
            c["last_gid"] = P["game_id"].iat[-1]
        self._enc = c
        return c

    def fit(self, view):
        if self._last_full_fit is not None and getattr(self, "ready", False) and \
                (view.t - self._last_full_fit) < pd.Timedelta(days=self.refit_days):
            self._fit_rotation(view)
            return
        self._last_full_fit = view.t
        P = view.table("POSSESSION")
        if len(P) == 0:
            self.ready = False
            return
        self.ready = True
        enc = self._encode(P)
        N = len(self.pidx)
        age = (view.t - P["knowledge_time"]).dt.total_seconds().to_numpy() / 86400.0
        om = np.ones(len(P)) if self.half_life_days is None else 0.5 ** (age / self.half_life_days)
        OL, DL, user = enc["OL"], enc["DL"], enc["user"]
        ev = P["event_code"].to_numpy()
        made = P["made"].to_numpy()
        st = P["start_type"].map(START_IDX).to_numpy()
        home = P["off_is_home"].to_numpy().astype(bool)

        # usage
        self.usage = mm_choice_weights(OL, user, N, om)
        # terminal-event rates per user, shrunk to league
        def wsum(mask):
            return np.bincount(user, weights=om * mask, minlength=N)
        uses = wsum(np.ones(len(P)))
        to, ft = wsum(ev == 0), wsum(ev == 1)
        shots = wsum(ev >= 2)
        s3 = wsum(ev == 3)
        m2, a2 = wsum((ev == 2) & (made == 1)), wsum(ev == 2)
        m3, a3 = wsum((ev == 3) & (made == 1)), wsum(ev == 3)
        ftm = np.bincount(user, weights=om * P["ftm"].to_numpy() * (ev == 1), minlength=N)
        fta = 2 * ft
        L = lambda num, den: num.sum() / max(den.sum(), 1e-9)  # noqa: E731
        lg = {"to": L(to, uses), "ft": L(ft, uses), "s3": L(s3, shots), "p2": L(m2, a2), "p3": L(m3, a3),
              "pft": L(ftm, fta)}
        self.lg = lg
        k = {"to": 40, "ft": 40, "s3": 30, "p2": 60, "p3": 60, "pft": 40}
        self.to_rate = (to + k["to"] * lg["to"]) / (uses + k["to"])
        self.ft_rate = (ft + k["ft"] * lg["ft"]) / (uses + k["ft"])
        self.share3 = (s3 + k["s3"] * lg["s3"]) / (shots + k["s3"])
        p2 = (m2 + k["p2"] * lg["p2"]) / (a2 + k["p2"])
        p3 = (m3 + k["p3"] * lg["p3"]) / (a3 + k["p3"])
        self.pft = (ftm + k["pft"] * lg["pft"]) / (fta + k["pft"])
        self.n2, self.n3 = a2 + k["p2"], a3 + k["p3"]
        self.n_uses = uses

        # start-type and home effects on make probability (league level)
        shot = ev >= 2
        base = made[shot & (st == 0)].mean() if (shot & (st == 0)).any() else 0.5
        tb = []
        for s_ in range(4):
            m_ = shot & (st == s_)
            tb.append(float(_logit(made[m_].mean()) - _logit(base)) if m_.sum() > 50 else 0.0)
        self.transition_bonus = tuple(tb)
        # the engine applies home_adv to home offence only, so the home-away logit gap IS the effect
        self.home_adv = float(_logit(made[shot & home].mean()) - _logit(made[shot & ~home].mean())) \
            if shot.sum() > 100 else 0.0
        # raw rates already contain each player's average transition and home contributions;
        # remove them, because the engine adds them back per shot (else they are double-counted)
        avg_tb = np.bincount(user, weights=om * shot * np.asarray(tb)[st], minlength=N) / np.maximum(shots, 1e-9)
        home_share = np.bincount(user, weights=om * shot * home, minlength=N) / np.maximum(shots, 1e-9)
        avg_ctx = avg_tb + self.home_adv * np.where(shots > 0, home_share, 0.5)
        self.logit2 = _logit(p2) - avg_ctx
        self.logit3 = _logit(p3) - avg_ctx

        # team defence: one-step logit residual of opponent makes, shrunk
        exp_p = 1 / (1 + np.exp(-(np.where(ev == 3, self.logit3[user], self.logit2[user]) + np.asarray(tb)[st]
                                  + self.home_adv * home)))
        dteam = P["def_team_id"].to_numpy()
        self.def_eff, self.def_se = {}, {}
        for tm in pd.unique(dteam):
            m_ = shot & (dteam == tm)
            r_ = np.sum(om[m_] * (made[m_] - exp_p[m_]))
            info = np.sum(om[m_] * exp_p[m_] * (1 - exp_p[m_])) + 300.0
            self.def_eff[tm] = float(r_ / info)
            self.def_se[tm] = float(1.0 / np.sqrt(info))

        # actor weights: rebounds, assists, steals, blocks
        orb = P["orb"].to_numpy() == 1
        rid = enc["reb"]
        drb = (~orb) & (rid >= 0) & (ev >= 2) & (made == 0)
        rid = np.maximum(rid, 0)
        self.orb_w = mm_choice_weights(OL[orb], rid[orb], N, om[orb])
        self.drb_w = mm_choice_weights(DL[drb], rid[drb], N, om[drb])
        missed = (ev >= 2) & (made == 0)
        A = self.orb_w[OL[missed]].sum(1)
        D = self.drb_w[DL[missed]].sum(1)
        target = np.average(orb[missed], weights=om[missed])
        lo, hi = 1e-3, 1e3
        for _ in range(60):
            c = np.sqrt(lo * hi)
            if np.average(c * A / (c * A + D), weights=om[missed]) < target:
                lo = c
            else:
                hi = c
        self.orb_w = self.orb_w * np.sqrt(lo * hi)
        ast = P["assisted"].to_numpy() == 1
        aid = np.maximum(enc["ast"], 0)
        elig = OL != user[:, None]
        self.ast_w = mm_choice_weights(OL[ast], aid[ast], N, om[ast], mask=elig[ast])
        made_fg = (ev >= 2) & (made == 1)
        self.assist_p = (float(np.average(ast[made_fg & (ev == 2)], weights=om[made_fg & (ev == 2)])),
                         float(np.average(ast[made_fg & (ev == 3)], weights=om[made_fg & (ev == 3)])))
        stl = P["steal"].to_numpy() == 1
        sid = np.maximum(enc["stl"], 0)
        self.stl_w = mm_choice_weights(DL[stl], sid[stl], N, om[stl])
        self.steal_share = float(np.average(stl[ev == 0], weights=om[ev == 0]))
        blk = P["blocked"].to_numpy() == 1
        bid = np.maximum(enc["blk"], 0)
        self.blk_w = mm_choice_weights(DL[blk], bid[blk], N, om[blk])
        m2mask = (ev == 2) & (made == 0)
        self.block_p2 = float(np.average(blk[m2mask], weights=om[m2mask]))

        # pace: mean duration ratio vs the engine's default ranges, per team
        dflt = GameBatch.__dataclass_fields__["duration_ranges"].default_factory()
        exp_d = np.array([(dflt[s_][0] + dflt[s_][1]) / 2 for s_ in range(4)])[st]
        dur = P["duration_s"].to_numpy()
        # exclude segments truncated by the end of a period (they are not pace information)
        end = P["start_elapsed_s"].to_numpy() + dur
        reg_b = np.abs(end - np.round(end / 720.0) * 720.0) < 1e-6
        ot_b = (end > 2880) & (np.abs((end - 2880) - np.round((end - 2880) / 300.0) * 300.0) < 1e-6)
        full = (dur > 0) & ~(reg_b & (end <= 2880 + 1e-6)) & ~ot_b
        self.team_pace = {}
        for tm in pd.unique(P["off_team_id"]):
            m_ = full & ((P["off_team_id"].to_numpy() == tm) | (P["def_team_id"].to_numpy() == tm))
            self.team_pace[tm] = float(np.sum(om[m_] * dur[m_]) / np.sum(om[m_] * exp_d[m_])) if m_.any() else 1.0

        # game-level shooting latent variance (as-of): excess over binomial variance per team-game
        self.latent_sd = 0.0
        if self.game_latent == "estimate":
            gid = P["game_id"].to_numpy()
            oteam = P["off_team_id"].to_numpy()
            key = pd.Series(gid[shot]).astype(str) + "|" + pd.Series(oteam[shot]).astype(str)
            df = pd.DataFrame({"k": key.to_numpy(), "res": made[shot] - exp_p[shot],
                               "v": exp_p[shot] * (1 - exp_p[shot])})
            agg = df.groupby("k").agg(res=("res", "sum"), v=("v", "sum"))
            if len(agg) > 30:
                excess = np.mean(agg["res"] ** 2 - agg["v"])
                self.latent_sd = float(np.sqrt(max(excess, 0.0)) / agg["v"].mean())

        # rotation shares from recent box scores; appearance rates by as-of status
        self._fit_rotation(view)

    def _fit_rotation(self, view):
        h = player_history(view)
        r = view.table("ROSTER").sort_values("knowledge_time").drop_duplicates(
            ["player_id", "team_id", "valid_from"], keep="last")
        self.roster_tbl = r
        # starter share per (team, slot) from games where starter and bench both played
        spell = r.dropna(subset=["depth_chart_slot"])
        self.slot_frac = {}
        recent = h[h["scheduled_tip"] >= view.t - pd.Timedelta(days=90)]
        sec = recent.set_index(["game_id", "player_id"])["seconds"]
        for tm, g in spell[spell["valid_to"].isna()].groupby("team_id"):
            slots = g.set_index("depth_chart_slot")["player_id"].to_dict()
            games = recent.loc[recent["team_id"] == tm, "game_id"].unique()
            fr = []
            for k_ in range(5):
                s_p, b_p = slots.get(k_), slots.get(k_ + 5)
                vals = []
                if s_p is not None and b_p is not None:
                    for gm in games:
                        a, b = sec.get((gm, s_p), 0), sec.get((gm, b_p), 0)
                        if a > 0 and b > 0 and a + b > 2400:
                            vals.append(a / (a + b))
                fr.append(float(np.median(vals)) if len(vals) >= 3 else 0.7)
            self.slot_frac[tm] = np.array(fr)
        # appearance probability given status visible `lead` before tip (as in L3)
        self.status_p = {"out": 0.0, "none": 1.0}
        if view.has("INJURY_REPORT") and len(h):
            ir = view.table("INJURY_REPORT")
            tips = h.drop_duplicates("game_id").set_index("game_id")["scheduled_tip"]
            ir = ir[ir["game_id"].isin(tips.index)]
            ir = ir[ir["knowledge_time"].to_numpy() <= (ir["game_id"].map(tips) - self.lead).to_numpy()]
            st = ir.sort_values("knowledge_time").drop_duplicates(["game_id", "player_id"], keep="last")
            st = st.rename(columns={"status": "inj_status"})
            m = h[["game_id", "player_id", "status"]].merge(st[["game_id", "player_id", "inj_status"]],
                                                            on=["game_id", "player_id"], how="left")
            m["inj_status"] = m["inj_status"].fillna("none")
            m["avail"] = m["status"] != "inactive"  # DNP-coach players were available
            self.status_p.update(m.groupby("inj_status")["avail"].mean().to_dict())

    # ------------------------------------------------------------------ predict
    def _team_arrays(self, view, team_id, game_id, tip, S, status_map):
        r = self.roster_tbl
        vt = r["valid_to"]
        cur = r[(r["team_id"] == team_id) & (r["valid_from"] <= tip) & (vt.isna() | (vt > tip))]
        cur = cur.sort_values("depth_chart_slot")
        pids = cur["player_id"].tolist()[:12]
        R = 12
        pids += [None] * (R - len(pids))
        slots = cur["depth_chart_slot"].tolist()[:12]
        avail_p = np.array([0.0 if p is None else self.status_p.get(status_map.get(p, "none"), 1.0) for p in pids])
        avail = self.rng.random((S, R)) < avail_p[None, :]
        del slots
        idx = np.array([self.pidx.get(p, -1) if p is not None else -1 for p in pids])
        known = idx >= 0

        def g(arr, default):
            v = np.where(known, arr[np.maximum(idx, 0)], default)
            return np.repeat(v[None, :], S, 0)

        lg = self.lg
        out = {
            "usage": g(self.usage, 0.8), "to_rate": g(self.to_rate, lg["to"]), "ft_rate": g(self.ft_rate, lg["ft"]),
            "share3": g(self.share3, lg["s3"]), "logit2": g(self.logit2, _logit(lg["p2"])),
            "logit3": g(self.logit3, _logit(lg["p3"])), "pft": g(self.pft, lg["pft"]),
            "orb_w": g(self.orb_w, np.median(self.orb_w)), "drb_w": g(self.drb_w, np.median(self.drb_w)),
            "ast_w": g(self.ast_w, 1.0), "stl_w": g(self.stl_w, 1.0), "blk_w": g(self.blk_w, 1.0),
        }
        if self.param_uncertainty:
            n2 = np.where(known, self.n2[np.maximum(idx, 0)], 60.0)
            n3 = np.where(known, self.n3[np.maximum(idx, 0)], 60.0)
            p2 = 1 / (1 + np.exp(-out["logit2"][0]))
            p3 = 1 / (1 + np.exp(-out["logit3"][0]))
            out["logit2"] = out["logit2"] + self.rng.standard_normal((S, R)) / np.sqrt(n2 * p2 * (1 - p2))
            out["logit3"] = out["logit3"] + self.rng.standard_normal((S, R)) / np.sqrt(n3 * p3 * (1 - p3))
            nu = np.where(known, self.n_uses[np.maximum(idx, 0)], 0.0) + 5.0
            out["usage"] = out["usage"] * np.exp(self.rng.standard_normal((S, R)) / np.sqrt(nu))
        out["usage"] = out["usage"] * avail
        st = np.zeros((S, 5), int)
        bn = np.zeros((S, 5), int)
        for s_ in range(S):
            st[s_], bn[s_] = resolve_rotation(avail[s_])
        frac = np.repeat(self.slot_frac.get(team_id, np.full(5, 0.7))[None, :], S, 0)
        return pids, out, st, bn, frac

    def predict(self, view, requests, targets):
        req = requests.reset_index(drop=True)
        S = self.n_sims
        n = len(req)
        p_app = np.zeros(n)
        cond = {t: np.zeros((n, PLAYER_TARGETS[t][0] + 1)) for t in targets}
        joint, team_out = {}, {}
        if not getattr(self, "ready", False):
            raise RuntimeError("PossessionSimulator.fit found no POSSESSION data")
        status = latest_injury_status(view, req["game_id"].unique())
        smap = {(g_, p_): s_ for g_, p_, s_ in zip(status["game_id"], status["player_id"], status["status"])}
        games = req.drop_duplicates("game_id")
        g_view = view.table("GAME").drop_duplicates("game_id", keep="last").set_index("game_id")
        batches, meta = [], []
        for _, row in games.iterrows():
            gm = g_view.loc[row["game_id"]]
            sides = []
            for tm in (gm["home_team_id"], gm["away_team_id"]):
                sm = {p_: s_ for (g_, p_), s_ in smap.items() if g_ == row["game_id"]}
                sides.append(self._team_arrays(view, tm, row["game_id"], gm["scheduled_tip"], S, sm))
            meta.append((row["game_id"], gm["home_team_id"], gm["away_team_id"], sides))
            arr = {k: np.stack([sides[0][1][k], sides[1][1][k]], 1) for k in sides[0][1]}
            b = dict(arr)
            b["starter_idx"] = np.stack([sides[0][2], sides[1][2]], 1)
            b["bench_idx"] = np.stack([sides[0][3], sides[1][3]], 1)
            b["starter_frac"] = np.stack([sides[0][4], sides[1][4]], 1)
            b["def_eff"] = np.tile([self.def_eff.get(gm["home_team_id"], 0.0),
                                    self.def_eff.get(gm["away_team_id"], 0.0)], (S, 1))
            if self.param_uncertainty:
                se = np.array([self.def_se.get(gm["home_team_id"], 0.05), self.def_se.get(gm["away_team_id"], 0.05)])
                b["def_eff"] = b["def_eff"] + self.rng.standard_normal((S, 2)) * se
            pace = np.sqrt(self.team_pace.get(gm["home_team_id"], 1.0) * self.team_pace.get(gm["away_team_id"], 1.0))
            b["pace_mult"] = np.full(S, pace * self.pace_scale) * np.exp(self.rng.normal(0, 0.03, S))
            b["shoot_latent"] = self.rng.normal(0, self.latent_sd, (S, 2)) if self.latent_sd > 0 else np.zeros((S, 2))
            b["usage_mult"] = np.ones((S, 2, 12))
            batches.append(b)
        big = {k: np.concatenate([b[k] for b in batches], 0) for k in batches[0]}
        gb = GameBatch(**big, home_adv=self.home_adv, transition_bonus=self.transition_bonus,
                       assist_p=self.assist_p, steal_share=self.steal_share, block_p2=self.block_p2,
                       blowout_margin=self.blowout_margin)
        res = simulate(gb, self.rng)
        tsamp = sim_to_targets(res.box, SI)  # each (S*G, 2, 12)
        for gi, (gid, h_id, a_id, sides) in enumerate(meta):
            sl = slice(gi * S, (gi + 1) * S)
            pid_to_pos = {}
            for side in range(2):
                for r_, p_ in enumerate(sides[side][0]):
                    if p_ is not None:
                        pid_to_pos[p_] = (side, r_)
            rows = np.nonzero(req["game_id"].to_numpy() == gid)[0]
            plist, samp = [], []
            for i in rows:
                pos = pid_to_pos.get(req.at[i, "player_id"])
                if pos is None:
                    p_app[i] = 0.0
                    for t in targets:
                        cond[t][i, 0] = 1.0
                    continue
                sec = res.box[sl, pos[0], pos[1], SI["SEC"]]
                app = sec > 0
                p_app[i] = app.mean()
                vals = np.stack([tsamp[t][sl, pos[0], pos[1]] for t in targets], 1)
                for j, t in enumerate(targets):
                    k_ = PLAYER_TARGETS[t][0]
                    cond[t][i] = smoothed_pmf_from_samples(vals[app, j], k_) if app.any() else np.eye(k_ + 1)[0]
                plist.append(req.at[i, "player_id"])
                samp.append(vals)
            joint[gid] = {"player_ids": plist, "targets": list(targets),
                          "team_ids": [req.at[i, "team_id"] for i in rows if req.at[i, "player_id"] in plist],
                          "samples": np.stack(samp, 1) if samp else None}
            team_out[gid] = {"home_pts": res.score[sl, 0], "away_pts": res.score[sl, 1],
                             "possessions": res.possessions[sl].sum(1), "periods": res.periods[sl],
                             "lead_changes": res.lead_changes[sl],
                             "home_team_id": h_id, "away_team_id": a_id}
        # smooth conditional pmfs slightly (simulation noise) toward themselves: none here; the
        # scorer's pre-registered epsilon mixture handles zero-probability outcomes.
        return PlayerForecast(self.name, view.t, req[["game_id", "player_id", "team_id"]], p_app, cond,
                              joint=joint, team=team_out)
