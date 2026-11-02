"""P1 ladder models: the FROZEN reference implementations (hoopslab.models.baselines) with one
engineering change only: the per-player appearance history that every _HistoryModel.fit
rebuilds from the as-of view is computed once per forecast time and shared through
view.cache. Outputs are bit-identical to the unmodified classes (checked by
experiments/p1_ladder/check_shared_equals_vanilla.py). No model logic is changed.
"""
from __future__ import annotations

import numpy as np

from hoopslab.features import player_history
from hoopslab.models import baselines as B
from hoopslab.targets import PLAYER_TARGETS, player_outcomes

_KEY = "_p1_shared_history_state"


def _shared_state(view):
    """Exactly the body of _HistoryModel.fit up to _fit_extra, computed once per view."""
    if _KEY in view.cache:
        return view.cache[_KEY]
    h = player_history(view)
    out = player_outcomes(h)
    out["scheduled_tip"] = h["scheduled_tip"].to_numpy()
    out["seconds"] = h["seconds"].to_numpy()
    app = out[out["appeared"]]
    league_pmf = {}
    for t, (k, _) in PLAYER_TARGETS.items():
        v = np.bincount(np.minimum(app[t].to_numpy(), k), minlength=k + 1).astype(float) + 0.01
        league_pmf[t] = v / v.sum()
    st = {
        "hist": out, "app": app, "league_pmf": league_pmf,
        "league_appear": float(out["appeared"].mean()) if len(out) else 0.85,
        "groups": {pid: g for pid, g in app.groupby("player_id", sort=False)},
        "appear_counts": out.groupby("player_id")["appeared"].agg(["sum", "count"]),
    }
    view.cache[_KEY] = st
    return st


class _Shared:
    def fit(self, view):
        st = _shared_state(view)
        self.hist, self.app, self.league_pmf = st["hist"], st["app"], st["league_pmf"]
        self.league_appear, self.groups, self.appear_counts = st["league_appear"], st["groups"], st["appear_counts"]
        self._fit_extra(view)


class Climatology(_Shared, B.Climatology):
    pass


class PlayerMean(_Shared, B.PlayerMean):
    pass


class MinutesRate(_Shared, B.MinutesRate):
    pass


def ladder_models(half_lives=(5.0, 10.0, 20.0, 40.0)):
    """The P1 model set. L2 is run on a pre-registered-now grid of half-lives (appearances);
    the per-season choice rule is in experiments/p1_ladder/config.json."""
    ms = [Climatology(), PlayerMean(None)]
    ms += [PlayerMean(h, name=f"L2_recent_hl{h:g}") for h in half_lives]
    ms += [MinutesRate(shrink=False), MinutesRate(shrink=True, param_uncertainty=True),
           MinutesRate(shrink=True, param_uncertainty=False)]
    return ms
