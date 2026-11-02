"""Replay harness: the whole prospective pipeline on development-period dates only.

ResearchCorpusProvider implements SnapshotProvider over the frozen corpus adapter tables and refuses
any cutoff outside the development window (2000-2020 seasons): it is physically unable to serve the
lockbox (2021-2022) or the holdout (>= 2023, absent from the package anyway). It is the ONLY provider
the research lane may use, and it exists to test the machinery, not to forecast real future games.

It truncates its tables at the cutoff before serving them (store.truncated), so even a bug in the
snapshot builder could not see later rows; the builder's own cutoff filter is then exercised with a
deliberately leaky wrapper (LeakyProvider) that appends post-cutoff rows, which must be dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adapters.base import build_store
from ..adapters.corpus_v102 import CorpusV102Adapter
from ..asof import to_utc
from ..features import roster_at
from .interface import GAME_COLUMNS, PLAYER_GAME_COLUMNS, AvailabilityState

DEV_FIRST, DEV_LAST = pd.Timestamp("1999-10-01", tz="UTC"), pd.Timestamp("2020-10-31", tz="UTC")


class ResearchCorpusProvider:
    provider_id = "research-corpus-v1.0.2-replay"

    def __init__(self, store=None):
        if store is None:
            store, _ = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A"), strict=True)
        self.store = store
        self._g = store.full_table("GAME")
        self._pg = store.full_table("PLAYER_GAME")

    def _guard(self, cutoff):
        cutoff = to_utc(cutoff)
        if not (DEV_FIRST <= cutoff <= DEV_LAST):
            raise PermissionError(f"replay provider serves development-window cutoffs only (1999-10-01..2020-10-31); got {cutoff}")
        return cutoff

    def schedule(self, cutoff) -> pd.DataFrame:
        cutoff = self._guard(cutoff)
        g = self._g[self._g["knowledge_time"] <= cutoff]
        g = g.sort_values("knowledge_time", kind="stable").drop_duplicates("game_id", keep="last")
        return g[GAME_COLUMNS].reset_index(drop=True)

    def appearance_history(self, cutoff) -> pd.DataFrame:
        cutoff = self._guard(cutoff)
        pg = self._pg[self._pg["knowledge_time"] <= cutoff]
        pg = pg.sort_values("knowledge_time", kind="stable").drop_duplicates(["game_id", "player_id"], keep="last")
        return pg[PLAYER_GAME_COLUMNS].reset_index(drop=True)

    def eligible_players(self, cutoff, game_ids) -> pd.DataFrame:
        """Historical replay has no availability feed: the as-of roster spells are listed as QUESTIONABLE_UNKNOWN."""
        cutoff = self._guard(cutoff)
        view = self.store.view(cutoff)
        g = view.table("GAME").set_index("game_id")
        rows = []
        for gid in game_ids:
            gm = g.loc[gid]
            r = roster_at(view, [gm["home_team_id"], gm["away_team_id"]], gm["scheduled_tip"])
            for _, x in r.iterrows():
                rows.append({"game_id": gid, "player_id": x["player_id"], "team_id": x["team_id"],
                             "availability_state": AvailabilityState.QUESTIONABLE_UNKNOWN.value,
                             "status_source": "as-of roster spell (no availability feed in the corpus)", "status_time": str(cutoff)})
        return pd.DataFrame(rows)

    # outcome side (scoring only; never used by the forecaster)
    def final_box_score(self, game_id) -> pd.DataFrame:
        pg = self._pg[self._pg["game_id"] == game_id].sort_values("knowledge_time", kind="stable").drop_duplicates(["game_id", "player_id"], keep="last")
        return pg[PLAYER_GAME_COLUMNS].reset_index(drop=True)

    def result_time(self, game_id):
        return self._pg.loc[self._pg["game_id"] == game_id, "knowledge_time"].max()

    def games_at(self, tip) -> list[str]:
        tip = to_utc(tip)
        g = self._g[(self._g["record_version"] == 1) & (self._g["scheduled_tip"] == tip)]
        return sorted(g["game_id"])


class LeakyProvider:
    """Wraps a provider and appends rows known AFTER the cutoff (future games / future box rows) to every table,
    plus a perturbed copy of future outcomes. The snapshot builder must drop all of them; the forecast must be
    identical to the clean provider's."""
    provider_id = "leaky-wrapper-for-tests"

    def __init__(self, base: ResearchCorpusProvider, horizon_days: int = 30, seed: int = 7):
        self.base, self.h, self.rng = base, pd.Timedelta(days=horizon_days), np.random.default_rng(seed)

    def schedule(self, cutoff):
        c = to_utc(cutoff)
        g = self.base._g
        fut = g[(g["knowledge_time"] > c) & (g["knowledge_time"] <= c + self.h)].drop_duplicates("game_id")[GAME_COLUMNS]
        return pd.concat([self.base.schedule(c), fut], ignore_index=True).drop_duplicates("game_id", keep="first")

    def appearance_history(self, cutoff):
        c = to_utc(cutoff)
        pg = self.base._pg
        fut = pg[(pg["knowledge_time"] > c) & (pg["knowledge_time"] <= c + self.h)].drop_duplicates(["game_id", "player_id"])[PLAYER_GAME_COLUMNS].copy()
        for col in ("pts", "seconds", "ast"):
            fut[col] = self.rng.permutation(fut[col].to_numpy())
        return pd.concat([self.base.appearance_history(c), fut], ignore_index=True)

    def eligible_players(self, cutoff, game_ids):
        return self.base.eligible_players(cutoff, game_ids)
