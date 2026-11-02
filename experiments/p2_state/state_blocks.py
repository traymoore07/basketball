"""E-STATE candidate state blocks (docs/p2/02_estate_spec.md section 3). Pure functions from a segment frame
(with the E-HIST slow variables and the E-STATE state variables merged in) to sparse design blocks. Every
input column is as-of the segment's first event.

Blocks: FB (team fouls / bonus, 16), GAP (2), TM (late clock x margin, 48; the E-HIST exploratory block),
SC (finer score, 73), PF (player fouls, 26), ROT (rotation, 4), PC (previous class, 9)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

CLOCK_BINS = [10, 30, 60, 120, 180]  # seconds left in the period -> 6 bins
MARGIN_BINS = [-8.5, -3.5, -0.5, 0.5, 3.5, 8.5]  # offence margin -> 7 levels
BLOCK_ORDER = ["FB", "GAP", "TM", "SC", "PF", "ROT", "PC"]
STATE_COLS = ["def_fouls_period", "off_fouls_period", "def_in_bonus", "def_bonus_l2m", "secs_since_prev", "stint_len_s", "lineup_change",
              "def_stint_len_s", "def_lineup_change", "def_pf_max", "def_pf_trouble", "def_pf_5", "off_pf_max", "off_pf_trouble", "off_pf_5"]


def _onehot(v, k):
    v = np.clip(np.asarray(v, int), 0, k - 1)
    return np.eye(k)[v]


def seconds_left(Q):
    per = Q["period"].to_numpy()
    el = Q["start_elapsed_s"].to_numpy(float)
    per_len = np.where(per <= 4, 720.0, 300.0)
    per_start = np.where(per <= 4, (per - 1) * 720.0, 2880.0 + (per - 5) * 300.0)
    return np.clip(per_start + per_len - el, 0, None)


def fb_block(Q):
    d = _onehot(Q["def_fouls_period"].to_numpy(), 7)
    o = _onehot(Q["off_fouls_period"].to_numpy(), 7)
    return sp.csr_matrix(np.concatenate([d, o, Q[["def_in_bonus"]].to_numpy(float), Q[["def_bonus_l2m"]].to_numpy(float)], 1))


def gap_block(Q):
    s = Q["secs_since_prev"].to_numpy(float)
    first = (s < 0).astype(float)
    return sp.csr_matrix(np.stack([np.log1p(np.clip(s, 0, None)), first], 1))


def tm_block(Q):
    per = Q["period"].to_numpy()
    cb = np.searchsorted(CLOCK_BINS, seconds_left(Q), side="right")  # 0..5
    mb = np.searchsorted(MARGIN_BINS, Q["off_margin_start"].to_numpy(float), side="right")  # 0..6
    late = per >= 4
    cell = np.where(late, cb * 7 + mb, 42 + cb)
    return sp.csr_matrix(np.eye(48)[cell])


def sc_block(Q):
    m = np.clip(Q["off_margin_start"].to_numpy(int), -15, 15) + 15
    fine = np.eye(31)[m]
    per = Q["period"].to_numpy()
    cb = np.searchsorted(CLOCK_BINS, seconds_left(Q), side="right")
    mb = np.searchsorted(MARGIN_BINS, Q["off_margin_start"].to_numpy(float), side="right")
    early = np.zeros((len(Q), 42))
    e = per <= 3
    early[np.where(e)[0], (cb * 7 + mb)[e]] = 1.0
    return sp.csr_matrix(np.concatenate([fine, early], 1))


def pf_block(Q):
    parts = []
    for side in ("def", "off"):
        parts += [_onehot(Q[f"{side}_pf_max"].to_numpy(), 7), _onehot(Q[f"{side}_pf_trouble"].to_numpy(), 3), _onehot(Q[f"{side}_pf_5"].to_numpy(), 3)]
    return sp.csr_matrix(np.concatenate(parts, 1))


def rot_block(Q):
    X = np.stack([np.log1p(np.clip(Q["stint_len_s"].to_numpy(float), 0, None)), Q["lineup_change"].to_numpy(float),
                  np.log1p(np.clip(Q["def_stint_len_s"].to_numpy(float), 0, None)), Q["def_lineup_change"].to_numpy(float)], 1)
    return sp.csr_matrix(X)


def pc_block(Q, y):
    df = pd.DataFrame({"g": Q["game_id"].to_numpy(), "t": Q["off_team_id"].to_numpy(), "s": Q["possession_seq"].to_numpy(), "y": y}, index=Q.index)
    pv = df.sort_values(["g", "t", "s"], kind="stable").groupby(["g", "t"], sort=False)["y"].shift(1).reindex(Q.index).to_numpy()
    M = np.zeros((len(Q), 9))
    has = ~np.isnan(pv)
    M[has, pv[has].astype(int)] = 1
    M[~has, 8] = 1
    return sp.csr_matrix(M)


def state_blocks(Q, y):
    return {"FB": fb_block(Q), "GAP": gap_block(Q), "TM": tm_block(Q), "SC": sc_block(Q), "PF": pf_block(Q), "ROT": rot_block(Q), "PC": pc_block(Q, y)}


def player_foul_features(counts, period):
    """counts: (n, 5) personal fouls of the on-court players before the segment; period: (n,).
    Returns max fouls (capped 6), players in foul trouble (fouls >= period + 1 in periods 1-4, >= 5 in OT; capped 2),
    players with >= 5 fouls (capped 2)."""
    counts = np.asarray(counts, int)
    thr = np.where(period <= 4, period + 1, 5)[:, None]
    mx = np.minimum(counts.max(1), 6)
    trouble = np.minimum((counts >= thr).sum(1), 2)
    five = np.minimum((counts >= 5).sum(1), 2)
    return mx, trouble, five
