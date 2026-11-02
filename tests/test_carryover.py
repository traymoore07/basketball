"""R1 carryover model on a synthetic world (no corpus, no network, no 2023+ data).

Grid point A must reproduce the frozen PlayerMean(5) exactly; B drops every previous-season appearance; C/D/E
multipliers follow the frozen formulas; the as-of context flags (team changed, starter status, turnover,
n_cur) come out as constructed; one PlayerForecast per grid point with valid pmfs.
"""
import numpy as np
import pandas as pd

from hoopslab.asof import AsOfStore
from hoopslab.models import carryover as CO
from hoopslab.models.baselines import PlayerMean

T0 = pd.Timestamp("2009-11-01 00:00", tz="UTC")


def world(seed=5):
    """Two seasons: nba-2009 (20 games) then nba-2010 (6 games so far); P0 changes team at the boundary."""
    rng = np.random.default_rng(seed)
    teams = ["T:A", "T:B", "T:C"]
    players = [f"P{i}" for i in range(9)]
    team_of = {p: teams[i % 3] for i, p in enumerate(players)}
    games, rows, roster = [], [], []
    k = 0
    for season, n in (("nba-2009", 20), ("nba-2010", 6)):
        if season == "nba-2010":
            team_of["P0"] = "T:B"   # traded over the summer
        for _ in range(n):
            tip = T0 + pd.Timedelta(days=k if season == "nba-2009" else 200 + k, hours=19)
            k += 1
            home, away = teams[k % 3], teams[(k + 1) % 3]
            gid = f"G{k:03d}"
            games.append({"game_id": gid, "league": "nba", "season_id": season, "phase": "regular", "scheduled_tip": tip,
                          "home_team_id": home, "away_team_id": away, "status": "final", "knowledge_time": tip - pd.Timedelta(days=7)})
            for p in players:
                if team_of[p] not in (home, away):
                    continue
                starter = int(p[1:]) % 3 == 0
                secs = int(rng.integers(600, 2400)) if rng.random() > 0.1 else 0
                m = secs / 60
                rows.append({"game_id": gid, "player_id": p, "team_id": team_of[p], "seconds": secs, "started": starter and secs > 0,
                             "status": "played" if secs > 0 else "dnp_coach",
                             "pts": int(rng.poisson(0.5 * m)), "fgm": 3, "fga": int(rng.poisson(0.4 * m)), "fg3m": int(rng.poisson(0.05 * m)), "fg3a": 3,
                             "ftm": 2, "fta": int(rng.poisson(0.1 * m)), "oreb": int(rng.poisson(0.05 * m)), "dreb": int(rng.poisson(0.15 * m)),
                             "ast": int(rng.poisson(0.1 * m)), "stl": 1, "blk": 0, "tov": int(rng.poisson(0.07 * m)), "pf": 2,
                             "knowledge_time": tip + pd.Timedelta(hours=4)})
    for p in players:
        t_first = teams[int(p[1:]) % 3]
        if p == "P0":
            roster.append({"player_id": p, "team_id": "T:A", "valid_from": T0 - pd.Timedelta(days=30), "valid_to": T0 + pd.Timedelta(days=150),
                           "knowledge_time": T0 - pd.Timedelta(days=30)})
            roster.append({"player_id": p, "team_id": "T:B", "valid_from": T0 + pd.Timedelta(days=150), "valid_to": pd.NaT,
                           "knowledge_time": T0 + pd.Timedelta(days=150)})
        else:
            roster.append({"player_id": p, "team_id": t_first, "valid_from": T0 - pd.Timedelta(days=30), "valid_to": pd.NaT,
                           "knowledge_time": T0 - pd.Timedelta(days=30)})
    fut_tip = T0 + pd.Timedelta(days=200 + k, hours=19)
    games.append({"game_id": "GFUT", "league": "nba", "season_id": "nba-2010", "phase": "regular", "scheduled_tip": fut_tip,
                  "home_team_id": "T:A", "away_team_id": "T:B", "status": "scheduled", "knowledge_time": fut_tip - pd.Timedelta(days=7)})
    tables = {"GAME": pd.DataFrame(games), "PLAYER_GAME": pd.DataFrame(rows), "ROSTER": pd.DataFrame(roster)}
    return AsOfStore(tables), fut_tip, team_of


def requests_for(view, tip, team_of):
    pl = [p for p in team_of if team_of[p] in ("T:A", "T:B")]
    return pd.DataFrame({"game_id": "GFUT", "player_id": pl, "team_id": [team_of[p] for p in pl], "tip": tip})


def test_grid_point_A_is_the_frozen_L2():
    store, tip, team_of = world()
    t = tip - pd.Timedelta(minutes=60)
    view = store.view(t)
    req = requests_for(view, tip, team_of)
    ref = PlayerMean(half_life=5.0)
    ref.fit(view)
    fr = ref.predict(view, req, CO.TARGETS)
    m = CO.SeasonAwareMean(["A"]).fit(view, 2010, req, tip)
    fa = m.predict(view, req, CO.TARGETS)["A"]
    for tg in CO.TARGETS:
        assert np.allclose(fa.cond[tg], fr.cond[tg], atol=1e-12), tg
    assert np.allclose(fa.p_appear, fr.p_appear)


def test_hard_reset_uses_current_season_only_and_context_flags():
    store, tip, team_of = world()
    t = tip - pd.Timedelta(minutes=60)
    view = store.view(t)
    req = requests_for(view, tip, team_of)
    m = CO.SeasonAwareMean(["A", "B"]).fit(view, 2010, req, tip)
    ctx = m.ctx.reset_index().set_index("player_id")
    assert bool(ctx.loc["P0", "team_changed"]) is True
    assert bool(ctx.loc["P3", "team_changed"]) is False
    assert bool(ctx.loc["P3", "starter_prev"]) is True and bool(ctx.loc["P1", "starter_prev"]) is False
    assert (ctx["n_cur"] >= 0).all() and ctx["has_prev_season"].all()
    # B's mean equals the recency-weighted mean over current-season appearances only
    app = m.app
    g = app[app["player_id"] == "P3"]
    cur = g[g["season_ord"] == 2010]
    w = CO.ew_weights(len(cur), 5.0)
    expect = float(np.sum(w * cur["PTS"].to_numpy()) / w.sum())
    jB = CO.GRID_IDS.index("B")
    assert abs(m.means["P3"][jB, CO.TARGETS.index("PTS")] - expect) < 1e-9
    fb = m.predict(view, req, ["PTS"])["B"]
    fb.validate()


def test_multipliers_follow_the_frozen_formulas():
    m0 = CO.multipliers(n_cur=0, team_changed=False, role_changed=False, high_turnover=False)
    m3 = CO.multipliers(n_cur=3, team_changed=True, role_changed=True, high_turnover=True)
    ids = CO.GRID_IDS
    assert m0[ids.index("A")] == 1 and m0[ids.index("B")] == 0 and m0[ids.index("C_c0.25")] == 0.25
    assert m0[ids.index("D_rho0.5_f0")] == 1.0 and abs(m3[ids.index("D_rho0.5_f0")] - 0.125) < 1e-12
    assert abs(m3[ids.index("D_rho0.5_f0.1")] - 0.125) < 1e-12 and abs(CO.multipliers(6, False, False, False)[ids.index("D_rho0.5_f0.1")] - 0.1) < 1e-12
    e = ids.index("E_rho0.5_f0_kt0.25_kr0.5_ku0.5")
    assert abs(m3[e] - 0.125 * 0.25 * 0.5 * 0.5) < 1e-12 and abs(m0[e] - 1.0) < 1e-12
    W = CO.weights_matrix(4, np.array([2, 1, 0, 0]), np.array([0.5]))
    assert np.allclose(W[0], CO.ew_weights(4, 5.0) * np.array([0.25, 0.5, 1, 1]))
    assert len(ids) == 44 and len(set(ids)) == 44


def test_every_grid_point_yields_a_valid_forecast():
    store, tip, team_of = world()
    t = tip - pd.Timedelta(minutes=60)
    view = store.view(t)
    req = requests_for(view, tip, team_of)
    m = CO.SeasonAwareMean().fit(view, 2010, req, tip)
    fc = m.predict(view, req, CO.TARGETS)
    assert set(fc) == set(CO.GRID_IDS)
    for f in fc.values():
        f.validate()
    assert m.r.shape == (44, 7)
