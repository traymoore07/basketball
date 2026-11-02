"""Read-only corpus adapter (BASKETBALL_CORPUS_v1.0.2): loading, schema conversion, split
enforcement, provenance, determinism, temporal filtering. NO model is fitted and no
hypothesis quantity is computed here (exposure class S1/S2).

Skipped when the researcher release is not on this machine.
"""
import hashlib
import os

import pandas as pd
import pytest

from hoopslab.adapters.base import build_store
from hoopslab.adapters.corpus_v102 import DEFAULT_DB, HOLDOUT_FROM_SEASON, CorpusV102Adapter
from hoopslab.backtest import leakage
from hoopslab.backtest.runner import BacktestSpec, forecast_points, make_requests
from hoopslab.contract.schema import available_level
from hoopslab.targets import player_outcomes

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_DB), reason="researcher release not present")


@pytest.fixture(scope="module")
def nba_a():
    ad = CorpusV102Adapter(league="nba", family="espn", seasons=[2016], level="A", include_holdout_schedule=True)
    store, report = build_store(ad, strict=True)      # strict: zero contract violations at level A
    return ad, store, report


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def test_level_a_strict_and_provenance(nba_a):
    ad, store, report = nba_a
    assert report["level"] == "A"
    assert report["issues"] == {}
    prov = report["provenance"]
    assert prov["corpus_package"] == "BASKETBALL_CORPUS_v1.0.2/researcher_isolated"
    assert "knowledge_time" in prov and "ROSTER" in prov["knowledge_time"]


def test_knowledge_time_audit_t4(nba_a):
    ad, store, report = nba_a
    tables = ad.load_tables()
    out = leakage.knowledge_time_audit(tables)
    assert out["passed"], out
    # no table claims an exact knowledge time: the corpus records effective time only
    for name, df in tables.items():
        assert not (df["knowledge_time_quality"] == "exact").any(), name
        assert "ingested_time" not in df.columns, name


def test_outcomes_not_visible_before_the_game(nba_a):
    ad, store, report = nba_a
    g = store.full_table("GAME")
    tip = g.loc[g["record_version"] == 1].iloc[100]
    t = tip["scheduled_tip"] - pd.Timedelta(minutes=60)
    view = store.view(t)
    pg = view.table("PLAYER_GAME")
    assert not (pg["game_id"] == tip["game_id"]).any()
    tg = view.table("TEAM_GAME")
    assert not (tg["game_id"] == tip["game_id"]).any()
    gv = view.table("GAME")
    row = gv[gv["game_id"] == tip["game_id"]]
    assert len(row) == 1 and row["status"].iloc[0] == "scheduled"
    later = store.view(tip["scheduled_tip"] + pd.Timedelta(hours=5)).table("GAME")
    assert later.loc[later["game_id"] == tip["game_id"], "status"].iloc[0] == "final"


def test_request_population_contains_non_appearing_players_t6(nba_a):
    ad, store, report = nba_a
    g = store.full_table("GAME")
    mid = g[g["record_version"] == 1]["scheduled_tip"].sort_values().iloc[len(g) // 3]
    spec = BacktestSpec(start=mid, end=mid + pd.Timedelta(days=3))
    pts = list(forecast_points(store, spec))
    assert pts
    t, games = pts[0]
    req = make_requests(store.view(t), games)
    assert len(req) > 0
    outc = player_outcomes(store.full_table("PLAYER_GAME").drop_duplicates(["game_id", "player_id"], keep="last"))
    assert leakage.population_check(req, outc)["passed"]


def test_split_enforcement_no_holdout_outcomes():
    ad = CorpusV102Adapter(league="nba", family="espn", seasons=None, level="A", include_holdout_schedule=True)
    tables = ad.load_tables()
    g = tables["GAME"]
    hold = g["season_id"].str.split("-").str[1].astype(int) >= HOLDOUT_FROM_SEASON
    assert hold.any(), "holdout schedule rows expected"
    assert (g.loc[hold, "record_version"] == 1).all() and (g.loc[hold, "status"] == "scheduled").all()
    for name in ("PLAYER_GAME", "TEAM_GAME"):
        season = tables[name]["game_id"].map(g.drop_duplicates("game_id").set_index("game_id")["season_id"])
        assert (season.str.split("-").str[1].astype(int) < HOLDOUT_FROM_SEASON).all(), name


def test_deterministic_and_namespaced(nba_a):
    ad, store, report = nba_a
    t1 = ad.load_tables()
    t2 = CorpusV102Adapter(league="nba", family="espn", seasons=[2016], level="A").load_tables()
    for name in ("GAME", "PLAYER_GAME", "TEAM_GAME", "ROSTER"):
        assert _hash(t1[name]) == _hash(t2[name]), name
    assert t1["PLAYER_GAME"]["player_id"].str.startswith("nba:").all()
    assert t1["GAME"]["home_team_id"].str.startswith("nba:espn:").all()
    assert not t1["PLAYER_GAME"].duplicated(["game_id", "player_id", "record_version"]).any()


def test_level_c_window_validates():
    ad = CorpusV102Adapter(league="nba", family="espn", seasons=[2016], level="C")
    tables = ad.load_tables()
    assert available_level(tables) == "C"
    assert len(tables["EVENT"]) > 100_000 and len(tables["POSSESSION"]) > 10_000 and len(tables["SHOT"]) > 10_000
    store, report = build_store(ad, strict=False)
    assert report["issues"] == {}, report["issues"]


def test_static_access_scan_t5_on_adapter_consumers():
    from hoopslab import features
    from hoopslab.models import baselines
    assert leakage.static_access_scan([features, baselines])["passed"]
