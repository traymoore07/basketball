"""E-STATE block builders (experiments/p2_state/state_blocks.py, state_vars.py helpers) on hand-made frames."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "p2_state"))
from state_blocks import fb_block, gap_block, pc_block, pf_block, player_foul_features, sc_block, seconds_left, tm_block  # noqa: E402
from state_vars import counts_before, is_personal  # noqa: E402


def _frame():
    return pd.DataFrame({
        "game_id": ["g"] * 4, "off_team_id": ["a", "a", "b", "a"], "possession_seq": [1, 2, 3, 4],
        "period": [1, 3, 4, 5], "start_elapsed_s": [0.0, 1440.0 + 700.0, 2160.0 + 700.0, 2880.0 + 290.0],
        "off_margin_start": [0, -20, 2, 9], "def_fouls_period": [0, 7, 5, 2], "off_fouls_period": [1, 0, 0, 0],
        "def_in_bonus": [0, 1, 1, 0], "def_bonus_l2m": [0, 0, 1, 0], "secs_since_prev": [-1, 30, 0, 900],
        "stint_len_s": [0, 10, 100, 0], "lineup_change": [0, 1, 0, 1], "def_stint_len_s": [0, 5, 0, 0], "def_lineup_change": [0, 0, 1, 0],
        "def_pf_max": [0, 3, 6, 8], "def_pf_trouble": [0, 1, 2, 5], "def_pf_5": [0, 0, 1, 3],
        "off_pf_max": [1, 2, 3, 4], "off_pf_trouble": [0, 0, 0, 1], "off_pf_5": [0, 0, 0, 0]})


def test_seconds_left_and_tm_cells():
    Q = _frame()
    left = seconds_left(Q)
    assert list(left) == [720.0, 20.0, 20.0, 10.0]
    M = tm_block(Q).toarray()
    assert M.shape == (4, 48) and (M.sum(1) == 1).all()
    assert M[0, 42 + 5] == 1  # period 1, 720 s left -> early clock bin 5
    assert M[1, 42 + 1] == 1  # period 3, 20 s left -> early bin 1
    assert M[2, 1 * 7 + 4] == 1  # period 4, 20 s left (bin 1), margin +2 -> level 4
    assert M[3, 1 * 7 + 6] == 1  # OT, exactly 10 s left (bin 1: bins are right-closed below), margin +9 -> level 6


def test_sc_block_fine_margin_and_early_cells():
    Q = _frame()
    M = sc_block(Q).toarray()
    assert M.shape == (4, 73)
    assert M[1, 0] == 1  # margin -20 clipped to -15 -> column 0
    assert M[0, 15] == 1  # margin 0
    assert M[:, :31].sum(1).tolist() == [1, 1, 1, 1]
    assert M[:, 31:].sum(1).tolist() == [1, 1, 0, 0]  # early cells only in periods 1-3


def test_fb_gap_pf_rot_shapes_and_caps():
    Q = _frame()
    F = fb_block(Q).toarray()
    assert F.shape == (4, 16) and F[1, 6] == 1 and F[0, 7 + 1] == 1 and F[2, 15] == 1
    G = gap_block(Q).toarray()
    assert G.shape == (4, 2) and G[0, 1] == 1 and G[0, 0] == 0 and abs(G[1, 0] - np.log1p(30)) < 1e-9
    P = pf_block(Q).toarray()
    assert P.shape == (4, 26) and (P.sum(1) == 6).all()
    assert P[3, 6] == 1 and P[3, 7 + 2] == 1 and P[3, 10 + 2] == 1  # caps 6 / 2 / 2


def test_pc_block_previous_class_same_offence():
    Q = _frame()
    y = np.array([5, 0, 7, 3])
    M = pc_block(Q, y).toarray()
    assert M[0, 8] == 1 and M[1, 5] == 1 and M[2, 8] == 1 and M[3, 0] == 1


def test_player_foul_features_rules():
    counts = np.array([[0, 1, 2, 2, 1], [3, 4, 5, 6, 0], [5, 5, 5, 0, 0]])
    period = np.array([1, 4, 5])
    mx, trouble, five = player_foul_features(counts, period)
    assert mx.tolist() == [2, 6, 5]
    assert trouble.tolist() == [2, 2, 2]  # period 1: fouls >= 2 -> two players; capped at 2
    assert five.tolist() == [0, 2, 2]


def test_counts_before_strict_and_personal_rule():
    groups = {("g", "p"): np.array([10.0, 20.0, 30.0])}
    keys = [("g", "p"), ("g", "p"), ("g", "q")]
    assert counts_before(groups, keys, np.array([20.0, 31.0, 5.0])).tolist() == [1, 3, 0]
    assert is_personal("Shooting") and is_personal("Offensive") and is_personal("Loose Ball")
    assert not is_personal("Technical") and not is_personal("Double Technical") and not is_personal("Defense 3 Second")
