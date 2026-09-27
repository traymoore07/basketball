"""As-of engine and leakage detectors: clean reference models pass, deliberately leaky ones are caught."""
import numpy as np
import pandas as pd
import pytest

from hoopslab import features
from hoopslab.adapters.base import build_store
from hoopslab.adapters.synthetic import SyntheticAdapter
from hoopslab.backtest import leakage as lk
from hoopslab.backtest.runner import (BacktestSpec, LockboxError, make_requests, outcomes_table,
                                      run_backtest)
from hoopslab.contract.schema import available_level, validate_table
from hoopslab.models import baselines, structural
from hoopslab.models.baselines import MinutesRate, PlayerMean
from hoopslab.synthetic.world import EPOCH, ScenarioConfig, generate_world


@pytest.fixture(scope="module")
def world():
    return generate_world(ScenarioConfig(seed=3, n_seasons=2, rounds_per_season=2,
                                         trades=[{"season": 1, "day": 4, "team_a": 0, "pos_a": 0,
                                                  "team_b": 1, "pos_b": 1}]))


@pytest.fixture(scope="module")
def store(world):
    s, rep = build_store(SyntheticAdapter(world))
    assert rep["issues"] == {}
    return s


def eval_times(store, n=3):
    g = store.full_table("GAME")
    tips = sorted(set(g.loc[g["scheduled_tip"] >= EPOCH + pd.Timedelta(days=370), "scheduled_tip"]))
    return [t - pd.Timedelta(minutes=60) for t in tips[:n]]


# --------------------------------------------------------------------- contract / as-of
def test_contract_level_and_validation(world):
    assert available_level(world.tables) == "C"
    for name, df in world.tables.items():
        assert validate_table(name, df, "B") == [], name


def test_asof_hides_future_and_late_reports(store, world):
    ir = world.tables["INJURY_REPORT"]
    late = ir[ir["status"] == "out"].merge(ir[ir["status"] == "questionable"], on=["game_id", "player_id"])
    assert len(late) > 0, "fixture should contain late scratches"
    gid, pid = late.iloc[0]["game_id"], late.iloc[0]["player_id"]
    tip = world.tables["GAME"].set_index("game_id").loc[gid, "scheduled_tip"].iloc[0]
    st60 = features.latest_injury_status(store.view(tip - pd.Timedelta(minutes=60)), [gid])
    st5 = features.latest_injury_status(store.view(tip - pd.Timedelta(minutes=5)), [gid])
    assert st60.set_index("player_id").loc[pid, "status"] == "questionable"
    assert st5.set_index("player_id").loc[pid, "status"] == "out"
    # the box score of that game is invisible before it ends
    v = store.view(tip - pd.Timedelta(minutes=60))
    assert gid not in set(v.table("PLAYER_GAME")["game_id"])


def test_versioned_roster_spells(store, world):
    ch = world.truth["changes"]
    tr = ch[ch["kind"] == "trade"].iloc[0]
    tip = EPOCH + pd.Timedelta(days=365 + 4)
    before_announce = store.view(tip - pd.Timedelta(hours=21))
    after = store.view(tip - pd.Timedelta(hours=1))
    r_before = features.roster_at(before_announce, [tr["from_team"]], tip)
    r_after = features.roster_at(after, [tr["from_team"], tr["team_id"]], tip)
    assert tr["player_id"] in set(r_before["player_id"])          # not yet known to have moved
    assert r_after.set_index("player_id").loc[tr["player_id"], "team_id"] == tr["team_id"]


def test_view_has_no_raw_access(store):
    v = store.view(EPOCH)
    with pytest.raises(AttributeError):
        v.store  # noqa: B018
    assert not hasattr(v, "full_table")


# --------------------------------------------------------------------- leaky reference models
class LeakyFullSeasonMean(PlayerMean):
    """Bug: features computed from the full dataset (includes the target game and the future)."""

    def __init__(self, raw_store):
        super().__init__(None, name="leaky_full_season")
        self.raw = raw_store

    def fit(self, view):
        super().fit(view)
        pg = self.raw.full_table("PLAYER_GAME")
        m = pg[pg["seconds"] > 0].groupby("player_id")["pts"].mean()
        for pid, mu in m.items():
            self.means.setdefault(pid, {t: 0.0 for t in baselines.PLAYER_TARGETS})["PTS"] = float(mu)


class LeakyFinalInjuryStatus(MinutesRate):
    """Bug: availability from the FINAL injury status (includes post-forecast late scratches)."""

    def __init__(self, raw_store):
        super().__init__(name="leaky_final_status")
        self.raw = raw_store

    def predict(self, view, requests, targets):
        f = super().predict(view, requests, targets)
        ir = self.raw.full_table("INJURY_REPORT").sort_values("knowledge_time").drop_duplicates(
            ["game_id", "player_id"], keep="last")
        out = set(zip(ir.loc[ir["status"] == "out", "game_id"], ir.loc[ir["status"] == "out", "player_id"]))
        mask = np.array([(g, p) in out for g, p in zip(f.keys["game_id"], f.keys["player_id"])])
        f.p_appear = np.where(mask, 0.0, f.p_appear)
        return f


class LeakyIdContent(PlayerMean):
    """Bug: uses the content of an opaque identifier as a feature."""

    def predict(self, view, requests, targets):
        f = super().predict(view, requests, targets)
        f.p_appear = np.clip(f.p_appear * (0.9 + 0.1 * (requests["player_id"].str[-1].astype(int) % 2).to_numpy()),
                             0, 1)
        return f


def test_T1_T2_clean_models_pass(store):
    times = eval_times(store)
    for fac in (lambda s: PlayerMean(None), lambda s: MinutesRate(shrink=True)):
        assert lk.truncation_invariance(store, fac, times)["passed"]
        assert lk.future_perturbation_invariance(store, fac, times)["passed"]


def test_T1_T2_catch_future_leaks(store):
    times = eval_times(store)
    r1 = lk.truncation_invariance(store, lambda s: LeakyFullSeasonMean(s), times)
    r2 = lk.future_perturbation_invariance(store, lambda s: LeakyFullSeasonMean(s), times)
    assert not r1["passed"] and not r2["passed"]


def test_side_channel_evades_dynamic_tests_by_design(store):
    """Documented limitation: a model holding a reference to the FULL data outside the
    factory is invisible to T1/T2. Hence rule L-X1 (no side channels) plus T5 (static scan)."""
    times = eval_times(store)
    r = lk.truncation_invariance(store, lambda s: LeakyFullSeasonMean(store), times)
    assert r["passed"]  # evades: this is why T5 exists
    import sys
    assert not lk.static_access_scan([sys.modules[__name__]])["passed"]


def test_T1_catches_final_injury_status_leak(store):
    # scan many times so that at least one late scratch is covered
    g = store.full_table("GAME")
    tips = sorted(set(g.loc[g["scheduled_tip"] >= EPOCH + pd.Timedelta(days=366), "scheduled_tip"]))
    times = [t - pd.Timedelta(minutes=60) for t in tips]
    r = lk.truncation_invariance(store, lambda s: LeakyFinalInjuryStatus(s), times, targets=("MIN",))
    assert not r["passed"]


def test_T3_id_relabel(world):
    tables = {k: v for k, v in world.tables.items()}
    s = build_store(SyntheticAdapter(world))[0]
    times = eval_times(s, 2)
    assert lk.id_relabel_invariance(tables, lambda s: MinutesRate(shrink=True), times)["passed"]
    assert not lk.id_relabel_invariance(tables, lambda s: LeakyIdContent(None), times)["passed"]


def test_T4_knowledge_time_audit(world):
    assert lk.knowledge_time_audit(world.tables)["passed"]
    bad = {k: v.copy() for k, v in world.tables.items()}
    tips = bad["GAME"].drop_duplicates("game_id").set_index("game_id")["scheduled_tip"]
    bad["PLAYER_GAME"]["knowledge_time"] = bad["PLAYER_GAME"]["game_id"].map(tips)  # "known at tip"
    assert not lk.knowledge_time_audit(bad)["passed"]
    bad2 = {k: v.copy() for k, v in world.tables.items()}
    bad2["TEAM_GAME"]["ingested_time"] = bad2["TEAM_GAME"]["knowledge_time"]
    assert not lk.knowledge_time_audit(bad2)["passed"]


def test_T5_static_scan():
    import sys
    assert lk.static_access_scan([baselines, structural, features])["passed"]
    assert not lk.static_access_scan([sys.modules[__name__]])["passed"]  # this file contains the leaky models


def test_T6_population_includes_non_appearing(store):
    t = eval_times(store, 1)[0]
    v = store.view(t)
    g = v.table("GAME")
    games = g[(g["scheduled_tip"] == t + pd.Timedelta(minutes=60)) & (g["status"] == "scheduled")]
    req = make_requests(v, games)
    assert lk.population_check(req, outcomes_table(store))["passed"]


def test_T7_feature_recompute(store):
    times = eval_times(store, 2)
    assert lk.feature_recompute_test(store, features.player_history, times)["passed"]


def test_lockbox_guard(store):
    t0 = EPOCH + pd.Timedelta(days=370)
    spec = BacktestSpec(start=t0, end=t0 + pd.Timedelta(days=1), lockbox=[(t0 - pd.Timedelta(days=1),
                                                                           t0 + pd.Timedelta(days=30))])
    with pytest.raises(LockboxError):
        run_backtest(store, spec, [PlayerMean(None)])
