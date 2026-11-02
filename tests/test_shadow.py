"""Shadow lane unit tests on a synthetic world (no corpus, no network, no 2023+ data).

Covers: identity of L2-A-SHADOW-v0.1 with the frozen PlayerMean(5) on the same as-of store; the snapshot
builder dropping post-cutoff rows (and the forecast hash being unchanged by them); canonical hashing;
the immutable lock store; the scoring step and the append-only ledger.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from hoopslab.asof import AsOfStore
from hoopslab.models.baselines import PlayerMean
from hoopslab.shadow import model as M
from hoopslab.shadow.interface import PLAYER_GAME_COLUMNS, AvailabilityState
from hoopslab.shadow.package import LockStore, build_snapshot, lock_hash, make_forecast_package, verify_locked
from hoopslab.shadow.scoring import Ledger, score_locked_forecast

T0 = pd.Timestamp("2010-01-01 00:00", tz="UTC")


def synthetic_world(seed=3, n_players=12, n_games=30):
    rng = np.random.default_rng(seed)
    teams = ["T:A", "T:B", "T:C"]
    players = [f"P{i}" for i in range(n_players)]
    team_of = {p: teams[i % 3] for i, p in enumerate(players)}
    games, rows = [], []
    for k in range(n_games):
        tip = T0 + pd.Timedelta(days=k, hours=19)
        home, away = teams[k % 3], teams[(k + 1) % 3]
        gid = f"G{k:03d}"
        games.append({"game_id": gid, "league": "nba", "season_id": "nba-2010", "phase": "regular", "scheduled_tip": tip,
                      "home_team_id": home, "away_team_id": away, "status": "final", "knowledge_time": tip - pd.Timedelta(days=7)})
        for p in players:
            if team_of[p] not in (home, away):
                continue
            secs = int(rng.integers(0, 2400)) if rng.random() > 0.15 else 0
            mins = secs / 60
            rows.append({"game_id": gid, "player_id": p, "team_id": team_of[p], "seconds": secs,
                         "pts": int(rng.poisson(0.5 * mins)), "fgm": 3, "fga": 8, "fg3m": int(rng.poisson(0.05 * mins)), "fg3a": 3,
                         "ftm": 2, "fta": 3, "oreb": int(rng.poisson(0.05 * mins)), "dreb": int(rng.poisson(0.15 * mins)),
                         "ast": int(rng.poisson(0.1 * mins)), "stl": 1, "blk": 0, "tov": int(rng.poisson(0.07 * mins)), "pf": 2,
                         "knowledge_time": tip + pd.Timedelta(hours=4)})
    # the game to forecast: known 7 days before, not played
    tip = T0 + pd.Timedelta(days=n_games, hours=19)
    games.append({"game_id": "GFUT", "league": "nba", "season_id": "nba-2010", "phase": "regular", "scheduled_tip": tip,
                  "home_team_id": "T:A", "away_team_id": "T:B", "status": "scheduled", "knowledge_time": tip - pd.Timedelta(days=7)})
    # a future game AFTER the forecast game (must never enter)
    tip2 = tip + pd.Timedelta(days=1)
    games.append({"game_id": "GFUT2", "league": "nba", "season_id": "nba-2010", "phase": "regular", "scheduled_tip": tip2,
                  "home_team_id": "T:B", "away_team_id": "T:C", "status": "scheduled", "knowledge_time": tip2 - pd.Timedelta(days=7)})
    fut_rows = [{"game_id": "GFUT2", "player_id": p, "team_id": team_of[p], "seconds": 1500, "pts": 40, "fgm": 3, "fga": 8, "fg3m": 9,
                 "fg3a": 3, "ftm": 2, "fta": 3, "oreb": 5, "dreb": 15, "ast": 12, "stl": 1, "blk": 0, "tov": 7, "pf": 2,
                 "knowledge_time": tip2 + pd.Timedelta(hours=4)} for p in players if team_of[p] in ("T:B", "T:C")]
    el = pd.DataFrame([{"game_id": "GFUT", "player_id": p, "team_id": team_of[p],
                        "availability_state": [AvailabilityState.CONFIRMED_ACTIVE, AvailabilityState.EXPECTED_ACTIVE,
                                               AvailabilityState.QUESTIONABLE_UNKNOWN, AvailabilityState.OUT][i % 4].value,
                        "status_source": "test", "status_time": str(tip - pd.Timedelta(minutes=90))}
                       for i, p in enumerate(players) if team_of[p] in ("T:A", "T:B")])
    return pd.DataFrame(games), pd.DataFrame(rows), pd.DataFrame(fut_rows), el, tip


class Provider:
    provider_id = "synthetic"

    def __init__(self, game, pg, el, leak=None):
        self.game, self.pg, self.el, self.leak = game, pg, el, leak

    def schedule(self, cutoff):
        return self.game

    def appearance_history(self, cutoff):
        return pd.concat([self.pg, self.leak], ignore_index=True) if self.leak is not None else self.pg

    def eligible_players(self, cutoff, game_ids):
        return self.el[self.el["game_id"].isin(game_ids)]


def test_identity_with_frozen_playermean():
    game, pg, _, el, tip = synthetic_world()
    cutoff = tip - pd.Timedelta(minutes=60)
    snap = build_snapshot(Provider(game, pg, el), ["GFUT"], cutoff)
    fc = M.forecast(snap)
    # the frozen class on the frozen store, directly
    g = game.copy(); g["record_version"] = 1
    p = pg.copy(); p["record_version"] = 1
    store = AsOfStore({"GAME": g, "PLAYER_GAME": p})
    view = store.view(cutoff)
    ref = PlayerMean(half_life=5.0)
    ref.fit(view)
    rows = [r for r in fc["players"] if r["pmf"] is not None]
    req = pd.DataFrame([{"game_id": r["game_id"], "player_id": r["player_id"], "team_id": r["team_id"]} for r in rows])
    f = ref.predict(view, req, M.TARGETS)
    for i, r in enumerate(rows):
        for t in M.TARGETS:
            assert np.array_equal(np.asarray(r["pmf"][t]), f.cond[t][i]), (r["player_id"], t)
    assert all(r["pmf"] is None for r in fc["players"] if r["availability_state"] == "OUT")
    assert "p_appear" not in json.dumps(fc)


def test_builder_drops_post_cutoff_rows_and_hash_is_invariant(tmp_path):
    game, pg, fut, el, tip = synthetic_world()
    cutoff = tip - pd.Timedelta(minutes=60)
    clean = build_snapshot(Provider(game, pg, el), ["GFUT"], cutoff)
    leaky = build_snapshot(Provider(game, pg, el, leak=fut), ["GFUT"], cutoff)
    assert leaky.manifest["rows_dropped_after_cutoff"]["PLAYER_GAME"] == len(fut)
    assert leaky.manifest["rows_dropped_after_cutoff"]["GAME"] == 0  # GFUT2's schedule row is known 7 days ahead (allowed)
    assert not (leaky.player_game["game_id"] == "GFUT2").any()  # but none of its box rows survive the cutoff
    assert clean.manifest["data_snapshot_id"] == leaky.manifest["data_snapshot_id"]
    created = cutoff + pd.Timedelta(minutes=5)
    a = make_forecast_package(clean, "GFUT", created_at=created)
    b = make_forecast_package(leaky, "GFUT", created_at=created)
    assert json.dumps(a["players"], sort_keys=True) == json.dumps(b["players"], sort_keys=True)
    assert a["data_snapshot_id"] == b["data_snapshot_id"]
    # the only difference is the manifest's record of what was dropped; with that equalised the lock hashes coincide
    b2 = json.loads(json.dumps(b)); b2["input_data_manifest"]["rows_dropped_after_cutoff"] = a["input_data_manifest"]["rows_dropped_after_cutoff"]
    assert lock_hash(a) == lock_hash(b2)
    assert a["information_cutoff"] == str(cutoff) and a["forecast_mode"].startswith("A")
    with pytest.raises(ValueError):
        make_forecast_package(clean, "GFUT", created_at=tip + pd.Timedelta(minutes=1))  # after tip: refused
    with pytest.raises(ValueError):
        make_forecast_package(clean, "GFUT", created_at=cutoff - pd.Timedelta(hours=1))  # before cutoff without the flag


def test_lock_store_is_immutable(tmp_path):
    game, pg, _, el, tip = synthetic_world()
    cutoff = tip - pd.Timedelta(minutes=60)
    snap = build_snapshot(Provider(game, pg, el), ["GFUT"], cutoff)
    pkg = make_forecast_package(snap, "GFUT", created_at=cutoff + pd.Timedelta(minutes=1))
    store = LockStore(str(tmp_path))
    l1 = store.lock(pkg)
    assert l1["forecast_version"] == 1 and verify_locked(l1)
    path1 = os.path.join(str(tmp_path), "forecasts", "GFUT", l1["forecast_id"] + ".json")
    bytes1 = open(path1, "rb").read()
    l2 = store.lock(make_forecast_package(snap, "GFUT", created_at=cutoff + pd.Timedelta(minutes=2)))  # a correction = new version
    assert l2["forecast_version"] == 2 and l2["lock_hash"] != l1["lock_hash"]
    assert open(path1, "rb").read() == bytes1  # v1 untouched
    assert store.load("GFUT", l1["forecast_id"])["lock_hash"] == l1["lock_hash"]
    # tampering is detected
    tampered = json.load(open(path1, encoding="utf-8"))
    tampered["players"][0]["distributions"]["PTS"]["mean"] += 1.0
    assert not verify_locked(tampered)


def test_scoring_and_ledger(tmp_path):
    game, pg, _, el, tip = synthetic_world()
    cutoff = tip - pd.Timedelta(minutes=60)
    snap = build_snapshot(Provider(game, pg, el), ["GFUT"], cutoff)
    pkg = LockStore(str(tmp_path)).lock(make_forecast_package(snap, "GFUT", created_at=cutoff + pd.Timedelta(minutes=1)))
    rng = np.random.default_rng(1)
    box = []
    first_issued = next(i for i, p in enumerate(pkg["players"]) if p["distributions"] is not None)
    for i, p in enumerate(pkg["players"]):
        secs = 0 if i == first_issued else 1800  # one late scratch among the issued forecasts
        box.append({"game_id": "GFUT", "player_id": p["player_id"], "team_id": p["team_id"], "seconds": secs, "pts": int(rng.integers(0, 30)),
                    "fgm": 3, "fga": 8, "fg3m": 2, "fg3a": 3, "ftm": 2, "fta": 3, "oreb": 1, "dreb": 4, "ast": 3, "stl": 1, "blk": 0,
                    "tov": 2, "pf": 2, "knowledge_time": tip + pd.Timedelta(hours=4)})
    box = pd.DataFrame(box)[PLAYER_GAME_COLUMNS]
    scored = score_locked_forecast(pkg, box, result_time=tip + pd.Timedelta(hours=4))
    issued = [p for p in pkg["players"] if p["distributions"] is not None]
    assert scored["counts"]["NOT_APPEARED"] == 1
    assert len(scored["rows"]) == (len(issued) - 1) * 7
    assert {r["population"] for r in scored["rows"]} <= {"STATUS_CONFIRMED", "STATUS_EXPECTED", "UNKNOWN"}
    assert all(np.isfinite(r["crps"]) and r["crps"] >= 0 for r in scored["rows"])
    led = Ledger(str(tmp_path))
    assert led.append(scored) == len(scored["rows"])
    with pytest.raises(ValueError):
        led.append(scored)  # same forecast, same revision: refused
    cum = led.cumulative()
    assert set(cum["population"]) <= {"STATUS_CONFIRMED", "STATUS_EXPECTED", "UNKNOWN"} and (cum["n"] > 0).all()
    # tampered package is refused by the scorer
    bad = dict(pkg); bad["lock_hash"] = "0" * 64
    with pytest.raises(ValueError):
        score_locked_forecast(bad, box, result_time=tip)


def test_cli_end_to_end(tmp_path):
    """The executor path: files in, locked package out, outcome file in, ledger rows out; no code edited."""
    from hoopslab.shadow.cli import main as cli
    game, pg, _, el, tip = synthetic_world()
    snapdir = tmp_path / "snap"; snapdir.mkdir()
    game.to_csv(snapdir / "game.csv", index=False); pg.to_csv(snapdir / "player_game.csv", index=False); el.to_csv(snapdir / "eligible.csv", index=False)
    lane = str(tmp_path / "lane")
    created = str(tip - pd.Timedelta(minutes=30))
    cli(["forecast", "--snapshot", str(snapdir), "--game-id", "GFUT", "--created-at", created, "--lane", lane])
    fdir = os.path.join(lane, "forecasts", "GFUT")
    fid = [f[:-5] for f in os.listdir(fdir) if f.endswith(".json")][0]
    pkg = json.load(open(os.path.join(fdir, fid + ".json"), encoding="utf-8"))
    assert verify_locked(pkg) and pkg["forecast_version"] == 1
    box = pd.DataFrame([{"game_id": "GFUT", "player_id": p["player_id"], "team_id": p["team_id"], "seconds": 1200, "pts": 10, "fgm": 3, "fga": 8,
                         "fg3m": 1, "fg3a": 3, "ftm": 2, "fta": 3, "oreb": 1, "dreb": 4, "ast": 3, "stl": 1, "blk": 0, "tov": 2, "pf": 2,
                         "knowledge_time": str(tip + pd.Timedelta(hours=4))} for p in pkg["players"]])
    box.to_csv(tmp_path / "box.csv", index=False)
    cli(["score", "--lane", lane, "--game-id", "GFUT", "--forecast-id", fid, "--box", str(tmp_path / "box.csv"),
         "--result-time", str(tip + pd.Timedelta(hours=4))])
    cli(["verify", "--lane", lane]); cli(["report", "--lane", lane])
    cli(["void", "--lane", lane, "--forecast-id", fid, "--reason", "postponed"])
    assert os.path.exists(os.path.join(lane, "ledger", "cumulative.csv"))
    with pytest.raises(ValueError):  # scoring the same revision twice is refused
        cli(["score", "--lane", lane, "--game-id", "GFUT", "--forecast-id", fid, "--box", str(tmp_path / "box.csv"),
             "--result-time", str(tip + pd.Timedelta(hours=4))])
