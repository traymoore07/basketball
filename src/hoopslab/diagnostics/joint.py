"""Joint-forecast (dependence) diagnostics for H8.

A simulator may tie a direct model on every marginal score and still be far better
(or worse) at the JOINT game. These instruments score dependence directly:

  * multivariate proper scores on within-team vectors (variogram score, primary;
    energy score, secondary);
  * derived-quantity calibration (PRA, team totals): PIT of sums computed from each
    model's own joint samples;
  * pair-dependence calibration: predicted vs realised Kendall tau of PIT pairs
    (teammate minutes, teammate points, points vs assists, ...);
  * named structural checks: blowout-driven starter minutes, rebound/made-shot and
    assist/teammate-FGM coupling, overtime, correlated team environments.

`independent_copula` builds the canonical comparison model: identical marginals,
dependence destroyed by permuting each column's samples independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..metrics import scoring as sc


def independent_copula(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(S, n_players, n_targets) -> same marginals, independent across all columns."""
    S = samples.shape[0]
    flat = samples.reshape(S, -1).copy()
    for j in range(flat.shape[1]):
        flat[:, j] = flat[rng.permutation(S), j]
    return flat.reshape(samples.shape)


def within_player_independent(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep each target's cross-player dependence? No: permute each (player, target) column."""
    return independent_copula(samples, rng)


def _rpit(samples: np.ndarray, y: float, rng) -> float:
    return float(np.mean(samples < y) + rng.random() * np.mean(samples == y))


def _rpit_samples(x: np.ndarray, rng) -> np.ndarray:
    """Randomised PIT of each sample under the samples' own empirical marginal."""
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    p = cnt / len(x)
    below = np.concatenate([[0.0], np.cumsum(p)[:-1]])
    return below[inv] + rng.random(len(x)) * p[inv]


def team_vector_scores(joint: dict, outcomes: pd.DataFrame, target: str, scale: float,
                       rng: np.random.Generator, alt_samples: dict | None = None) -> pd.DataFrame:
    """Variogram and energy scores of the within-team vector of `target` per (game, team).

    joint: {game_id: {"player_ids", "team_ids", "targets", "samples" (S, n_players, n_targets)}}
    outcomes: player_outcomes table (unconditional targets).
    """
    rows = []
    oc = outcomes.set_index(["game_id", "player_id"])
    for gid, J in joint.items():
        if J.get("samples") is None:
            continue
        samples = J["samples"] if alt_samples is None else alt_samples[gid]
        ti = J["targets"].index(target)
        tids = np.asarray(J["team_ids"])
        for tm in np.unique(tids):
            cols = np.nonzero(tids == tm)[0]
            y = np.array([oc[target].get((gid, J["player_ids"][c]), 0) for c in cols], float)
            X = samples[:, cols, ti].astype(float)
            rows.append({"game_id": gid, "team_id": tm, "target": target,
                         "variogram": sc.variogram_score(X, y, p=0.5, scale=scale),
                         "energy": sc.energy_score(X, y, scale=np.full(len(cols), scale), rng=rng)})
    return pd.DataFrame(rows)


def derived_pit(joint: dict, outcomes: pd.DataFrame, components: list[str], rng: np.random.Generator,
                alt_samples: dict | None = None) -> np.ndarray:
    """Randomised PIT of a per-player sum of targets (e.g. PTS+REB+AST) from joint samples."""
    oc = outcomes.set_index(["game_id", "player_id"])
    pits = []
    for gid, J in joint.items():
        if J.get("samples") is None:
            continue
        samples = J["samples"] if alt_samples is None else alt_samples[gid]
        idx = [J["targets"].index(c) for c in components]
        tot = samples[:, :, idx].sum(-1)                       # (S, n_players)
        for j, pid in enumerate(J["player_ids"]):
            y = sum(oc[c].get((gid, pid), 0) for c in components)
            x = tot[:, j]
            below = np.mean(x < y)
            at = np.mean(x == y)
            pits.append(below + rng.random() * at)
    return np.array(pits)


def pair_tau_calibration(joint: dict, outcomes: pd.DataFrame, pairs: list[tuple], rng: np.random.Generator,
                         alt_samples: dict | None = None, min_minutes_target: str | None = None) -> pd.DataFrame:
    """Predicted vs realised Kendall tau for (player a target x, player b target y) pairs.

    pairs: list of (kind, target_x, target_y), kind in {"same_player", "teammates"}.
    Realised tau: across games, tau between the two PITs (each from the model's own
    marginal samples). Predicted tau: mean over games of tau within the joint samples.
    A model with correct dependence has predicted ~= realised.
    """
    oc = outcomes.set_index(["game_id", "player_id"])
    out = []
    for kind, tx, ty in pairs:
        u_list, v_list, pred = [], [], []
        for gid, J in joint.items():
            if J.get("samples") is None:
                continue
            samples = J["samples"] if alt_samples is None else alt_samples[gid]
            ix, iy = J["targets"].index(tx), J["targets"].index(ty)
            pids = J["player_ids"]
            tids = np.asarray(J["team_ids"])
            if kind == "same_player":
                combos = [(j, j) for j in range(len(pids))]
            else:
                combos = []
                for tm in np.unique(tids):
                    cols = np.nonzero(tids == tm)[0]
                    # the two highest expected-minutes players of the team: stable pair definition
                    mi = J["targets"].index("MIN") if "MIN" in J["targets"] else ix
                    order = cols[np.argsort(-samples[:, cols, mi].mean(0))]
                    if len(order) >= 2:
                        combos.append((order[0], order[1]))
            for a, b in combos:
                xa, yb = samples[:, a, ix].astype(float), samples[:, b, iy].astype(float)
                if xa.std() == 0 or yb.std() == 0:
                    continue
                oa = oc[tx].get((gid, pids[a]), 0)
                ob = oc[ty].get((gid, pids[b]), 0)
                u_list.append(_rpit(xa, oa, rng))
                v_list.append(_rpit(yb, ob, rng))
                # predicted dependence on the SAME jittered-PIT scale as the realised pairs
                # (raw-sample tau is not comparable when marginals are concentrated on few values)
                pred.append(kendalltau(_rpit_samples(xa, rng), _rpit_samples(yb, rng)).statistic)
        if len(u_list) > 10:
            out.append({"pair": f"{kind}:{tx}~{ty}", "n": len(u_list),
                        "tau_pred": float(np.nanmean(pred)),
                        "tau_real": float(kendalltau(u_list, v_list).statistic)})
    return pd.DataFrame(out)


def blowout_minutes_check(joint: dict, team: dict, outcomes: pd.DataFrame, team_results: pd.DataFrame) -> dict:
    """Predicted vs realised correlation between |final margin| and team starters' total minutes.

    team_results: TEAM_GAME-like frame with game_id, team_id, is_home, pts.
    Starters are the five highest expected-minutes players per team in the forecast.
    """
    oc = outcomes.set_index(["game_id", "player_id"])
    pred, real_m, real_s = [], [], []
    tr = team_results.set_index(["game_id", "is_home"])["pts"]
    for gid, J in joint.items():
        if J.get("samples") is None or gid not in team:
            continue
        mi = J["targets"].index("MIN")
        tids = np.asarray(J["team_ids"])
        margin = np.abs(team[gid]["home_pts"] - team[gid]["away_pts"])
        real_margin = abs(tr.get((gid, True), 0) - tr.get((gid, False), 0))
        for tm in np.unique(tids):
            cols = np.nonzero(tids == tm)[0]
            top = cols[np.argsort(-J["samples"][:, cols, mi].mean(0))[:5]]
            smin = J["samples"][:, top, mi].sum(1)
            if smin.std() > 0 and margin.std() > 0:
                pred.append(np.corrcoef(margin, smin)[0, 1])
            real_m.append(real_margin)
            real_s.append(sum(oc["MIN"].get((gid, J["player_ids"][c]), 0) for c in top))
    return {"corr_pred_mean": float(np.mean(pred)) if pred else float("nan"),
            "corr_real": float(np.corrcoef(real_m, real_s)[0, 1]) if len(real_m) > 3 else float("nan"),
            "n": len(real_m)}
