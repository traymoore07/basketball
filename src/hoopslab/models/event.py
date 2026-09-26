"""Event-level (possession-segment) predictive models for H2/H3/H5/H7.

Outcome: 8 classes per segment:
    0 TO, 1 FT trip 0/2, 2 FT 1/2, 3 FT 2/2, 4 2PA miss, 5 2PA make, 6 3PA miss, 7 3PA make.

Feature blocks (pre-registered; docs/prereg/01_experimental_blueprint.md, E-HIST):
    S     compact state: start type, home, score-margin bucket, period, clutch flag,
          offense and defense team, on-court offense and defense players (bag of players)
    L1    offense team's previous possession points (this game)
    L3    mean of its last 3 possessions' points
    L10   mean of its last 10
    EW    exponentially weighted history summaries (half-lives 3 and 10 possessions).
          A tractable stand-in for "entire game history"; a neural sequence model plugs
          into the same interface for real data.
    GL    game-level latent estimate: offense team's running shooting residual this game
    NOISE a high-cardinality context with no information (H7 placebo)

The model is L2-regularised multinomial logistic regression (fixed, pre-registered
penalty), so feature sets are compared with a single model family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

N_CLASSES = 8
START = ["after_make", "after_dreb", "after_live_turnover", "after_oreb"]


def event_classes(P: pd.DataFrame) -> np.ndarray:
    ev = P["event_code"].to_numpy()
    made = P["made"].to_numpy()
    ftm = P["ftm"].to_numpy()
    return np.select([ev == 0, ev == 1, ev == 2, ev == 3],
                     [0, 1 + ftm, 4 + made, 6 + made]).astype(int)


def _hist_features(P: pd.DataFrame) -> pd.DataFrame:
    """Within-game history features, computed only from EARLIER segments of the same game."""
    df = P[["game_id", "possession_seq", "off_team_id", "points", "event_code", "made"]].copy()
    df = df.sort_values(["game_id", "possession_seq"], kind="stable")
    ends = df["event_code"].to_numpy() != -1  # every segment
    df["is_fga"] = (df["event_code"] >= 2).astype(float)
    df["fgm"] = ((df["event_code"] >= 2) & (df["made"] == 1)).astype(float)
    g = df.groupby(["game_id", "off_team_id"], sort=False)
    prev_pts = g["points"].shift(1)
    out = pd.DataFrame(index=df.index)
    out["L1"] = prev_pts.fillna(0.0)
    out["L1_missing"] = prev_pts.isna().astype(float)
    shifted = g["points"].shift(1)
    key = [df["game_id"], df["off_team_id"]]
    out["L3"] = shifted.groupby(key, sort=False).rolling(3, min_periods=1).mean() \
        .reset_index(level=[0, 1], drop=True).reindex(df.index).fillna(0.0)
    out["L10"] = shifted.groupby(key, sort=False).rolling(10, min_periods=1).mean() \
        .reset_index(level=[0, 1], drop=True).reindex(df.index).fillna(0.0)
    for hl in (3, 10):
        out[f"EW{hl}"] = shifted.groupby(key, sort=False).transform(
            lambda s: s.ewm(halflife=hl, adjust=False, ignore_na=True).mean()).fillna(0.0)
    fga_cum = g["is_fga"].cumsum() - df["is_fga"]
    fgm_cum = g["fgm"].cumsum() - df["fgm"]
    out["GL"] = (fgm_cum - 0.5 * fga_cum) / (fga_cum + 10.0)
    out["n_prev"] = np.log1p(g.cumcount())
    del ends
    return out.reindex(P.index)


class EventDesign:
    """Builds design matrices for named feature blocks with a fixed vocabulary."""

    def __init__(self, P_train: pd.DataFrame, noise_levels: int = 500, seed: int = 0):
        self.teams = sorted(set(P_train["off_team_id"]) | set(P_train["def_team_id"]))
        players = set()
        for col in ("off_lineup", "def_lineup"):
            for lu in P_train[col]:
                players.update(lu)
        self.players = {p: i for i, p in enumerate(sorted(players))}
        self.noise_levels = noise_levels
        self.seed = seed

    def all_blocks(self, P: pd.DataFrame) -> dict[str, np.ndarray]:
        """Compute every feature block once for a dataset; `select` assembles subsets."""
        out = {}
        H = _hist_features(P)
        st = P["start_type"].map({s: i for i, s in enumerate(START)}).fillna(0).astype(int).to_numpy()
        mb = np.clip(np.round(P["off_margin_start"].to_numpy() / 6.0), -3, 3).astype(int) + 3
        per = np.minimum((P["start_elapsed_s"].to_numpy() // 720).astype(int), 4)
        clutch = (P["start_elapsed_s"].to_numpy() >= 2760) & (np.abs(P["off_margin_start"].to_numpy()) <= 5)
        ti = {t: i for i, t in enumerate(self.teams)}
        n = len(P)
        lineup_cols = []
        rows5 = np.repeat(np.arange(n), 5)
        for col in ("off_lineup", "def_lineup"):
            L = np.stack(P[col].to_numpy())
            codes = np.vectorize(lambda p: self.players.get(p, -1))(L).ravel()
            ok = codes >= 0
            lineup_cols.append(sp.csr_matrix((np.ones(ok.sum()), (rows5[ok], codes[ok])),
                                             shape=(n, len(self.players))))
        dense_state = np.concatenate([
            np.eye(4)[st][:, 1:], P["off_is_home"].to_numpy().astype(float)[:, None], np.eye(7)[mb],
            np.eye(5)[per], clutch.astype(float)[:, None],
            np.eye(len(self.teams))[P["off_team_id"].map(ti).fillna(0).astype(int)],
            np.eye(len(self.teams))[P["def_team_id"].map(ti).fillna(0).astype(int)]], 1)
        out["S"] = sp.hstack([sp.csr_matrix(dense_state)] + lineup_cols, format="csr")
        out["L1"] = sp.csr_matrix(H[["L1", "L1_missing"]].to_numpy(float))
        out["L3"] = sp.csr_matrix(H[["L3"]].to_numpy(float))
        out["L10"] = sp.csr_matrix(H[["L10"]].to_numpy(float))
        out["EW"] = sp.csr_matrix(H[["EW3", "EW10", "n_prev"]].to_numpy(float))
        out["GL"] = sp.csr_matrix(10.0 * H[["GL"]].to_numpy(float))
        h = pd.util.hash_pandas_object(P[["game_id", "possession_seq"]], index=False).to_numpy()
        cell = (h % np.uint64(self.noise_levels)).astype(int)
        out["NOISE"] = sp.csr_matrix((np.ones(n), (np.arange(n), cell)), shape=(n, self.noise_levels))
        return out

    @staticmethod
    def select(blocks: dict, which: list[str]):
        n = next(iter(blocks.values())).shape[0]
        return sp.hstack([sp.csr_matrix(np.ones((n, 1)))] + [blocks[b] for b in which], format="csr")

    def blocks(self, P: pd.DataFrame, which: list[str]):
        return self.select(self.all_blocks(P), which)


class MultinomialLogit:
    """L2-regularised softmax regression fitted by L-BFGS (pre-registered penalty).

    Works on sparse design matrices; features are not standardised (one-hots and small
    bounded summaries), so the penalty is on the raw scale. Column 0 is the intercept
    (unpenalised).
    """

    def __init__(self, l2: float = 1e-4, maxiter: int = 300):
        self.l2 = l2
        self.maxiter = maxiter

    def fit(self, X, y: np.ndarray):
        X = sp.csr_matrix(X, dtype=np.float64)
        n, d = X.shape
        K = N_CLASSES
        Y = np.eye(K)[y]
        XT = X.T.tocsr()

        def f(wflat):
            W = wflat.reshape(d, K)
            Z = X @ W
            Z -= Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z).sum(1))
            nll = np.mean(lse - (Z * Y).sum(1))
            Pm = np.exp(Z - lse[:, None])
            G = np.asarray(XT @ (Pm - Y)) / n
            reg = 0.5 * self.l2 * np.sum(W[1:] ** 2)
            G[1:] += self.l2 * W[1:]
            return nll + reg, G.ravel()

        res = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter})
        self.W = res.x.reshape(d, K)
        self.converged = bool(res.success)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Z = np.asarray(sp.csr_matrix(X, dtype=np.float64) @ self.W)
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)
