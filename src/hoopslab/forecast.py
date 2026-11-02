"""Forecast containers shared by all models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .targets import PLAYER_TARGETS


@dataclass
class PlayerForecast:
    """Player-game forecasts from one model at one forecast time.

    keys:      DataFrame [game_id, player_id, team_id] (n rows)
    p_appear:  (n,) probability the player appears (seconds > 0)
    cond:      target -> (n, K+1) pmf conditional on appearing
    joint:     optional game_id -> {"player_ids": [...], "targets": [...],
               "samples": (S, n_players, n_targets)}. Samples are UNCONDITIONAL
               (non-appearance -> zeros) and come from the same draws as the marginals.
    """
    model: str
    forecast_time: pd.Timestamp
    keys: pd.DataFrame
    p_appear: np.ndarray
    cond: dict[str, np.ndarray]
    joint: dict = field(default_factory=dict)
    team: dict = field(default_factory=dict)   # game_id -> {"home_pts": samples, "away_pts": samples, ...}

    def unconditional(self, target: str) -> np.ndarray:
        pmf = self.cond[target] * self.p_appear[:, None]
        pmf[:, 0] += 1.0 - self.p_appear
        return pmf

    def validate(self):
        n = len(self.keys)
        assert self.p_appear.shape == (n,)
        assert np.all((self.p_appear >= 0) & (self.p_appear <= 1))
        for t, p in self.cond.items():
            k = PLAYER_TARGETS[t][0]
            assert p.shape == (n, k + 1), (t, p.shape)
            assert np.allclose(p.sum(1), 1.0, atol=1e-6), t
        return True
