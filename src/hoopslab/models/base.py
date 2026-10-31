"""Forecaster interface.

Rules every model obeys (enforced by backtest/leakage.py):
  * the only data access is through the AsOfView passed to fit/predict;
  * predict() must not depend on anything not visible at view.t;
  * outputs are full distributions (PlayerForecast), never bare point estimates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..asof import AsOfView
from ..forecast import PlayerForecast


class Forecaster(ABC):
    name: str = "model"
    rung: str = "?"               # baseline-ladder rung id (registry/baseline_ladder.yaml)
    data_level: str = "A"         # minimum data level required
    info_set: str = "pre"         # pre (T-60, no lineups) | lock (announced starters) | live

    def fit(self, view: AsOfView) -> None:
        """Update internal state using data visible in `view` (called at each forecast time)."""

    @abstractmethod
    def predict(self, view: AsOfView, requests: pd.DataFrame, targets: list[str]) -> PlayerForecast:
        """Forecast each (game_id, player_id, team_id, tip) row in `requests`."""
