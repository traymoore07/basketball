"""As-of change-point detection for role changes (used to DEFINE strata as-of, H6).

A two-sided CUSUM on standardised residuals of a player's usage-like rate
(FGA+FTA/2+TOV per minute) relative to its trailing exponentially weighted mean.
Flags are computed only from games before each forecast time, so a stratum defined
by "a change was flagged in the last N games" is itself leak-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def usage_rate(h: pd.DataFrame) -> pd.Series:
    mins = np.maximum(h["seconds"] / 60.0, 1.0)
    return (h["fga"] + 0.44 * h["fta"] + h["tov"]) / mins


def cusum_flags(h: pd.DataFrame, k: float = 0.75, thresh: float = 6.0, warmup: int = 10,
                half_life: float = 20.0) -> pd.DataFrame:
    """Per appearance row: whether a CUSUM alarm fired at that game (player-level, sequential).

    h must be one player's appearances sorted by time, with seconds, fga, fta, tov.
    """
    r = usage_rate(h).to_numpy()
    n = len(r)
    alarm = np.zeros(n, bool)
    mu, var, w = 0.0, 0.0, 0.0
    sp = sm = 0.0
    a = 0.5 ** (1 / half_life)
    for i in range(n):
        if w > 0 and i >= warmup:
            sd = np.sqrt(max(var / w, 1e-4))
            z = (r[i] - mu / w) / sd
            sp = max(0.0, sp + z - k)
            sm = max(0.0, sm - z - k)
            if sp > thresh or sm > thresh:
                alarm[i] = True
                sp = sm = 0.0
                mu, var, w = r[i], 0.0, 1.0  # restart the baseline after an alarm
                continue
        # update EW mean/var (sums)
        m_old = mu / w if w > 0 else r[i]
        mu = a * mu + r[i]
        w = a * w + 1.0
        var = a * var + (r[i] - m_old) * (r[i] - mu / w)
    return pd.DataFrame({"alarm": alarm}, index=h.index)
