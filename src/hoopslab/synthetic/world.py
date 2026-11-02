"""Synthetic basketball worlds with known ground truth.

A world emits canonical data-contract tables (with event and knowledge times), so the
whole pipeline (adapter -> as-of store -> models -> backtest -> metrics) runs
unchanged on synthetic and real data. It also returns hidden TRUTH tables that the
instruments are checked against. Models never see those tables.

This is NOT evidence about real basketball. It verifies that the instruments
diagnose known mechanisms correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..sim.engine import GameBatch, SI, START_TYPES, simulate

ROSTER = 12
POS = np.array([0, 0, 1, 2, 2, 0, 0, 1, 2, 2, 1, 1])  # 0 guard, 1 wing, 2 big (by depth slot)
EPOCH = pd.Timestamp("2031-10-20 23:30", tz="UTC")
PARAMS = ["usage", "to_rate", "ft_rate", "share3", "logit2", "logit3", "pft",
          "orb_w", "drb_w", "ast_w", "stl_w", "blk_w"]
SRC = {"knowledge_time_quality": "exact", "source": "synthetic"}


@dataclass
class ScenarioConfig:
    name: str = "base"
    seed: int = 0
    n_teams: int = 8
    n_seasons: int = 3
    rounds_per_season: int = 6            # full round robins per season
    momentum: float = 0.0                 # TRUE history dependence (logit)
    game_latent_sd: float = 0.0           # team shooting logit sd per game
    usage_form_sd: float = 0.0            # player usage log-sd per game
    pace_sd: float = 0.03
    injury_hazard: float = 0.012
    injury_mean_len: float = 4.0
    late_scratch_frac: float = 0.15
    questionable_noise: float = 0.03
    skill_drift_sd: float = 0.08
    rookie_frac: float = 0.0              # share of roster replaced by new players each new season
    role_changes: list = field(default_factory=list)   # dicts: season, day, team, pos, usage_mult
    star_absences: list = field(default_factory=list)  # dicts: season, day, team, n_games
    trades: list = field(default_factory=list)         # dicts: season, day, team_a, pos_a, team_b, pos_b
    blowout_margin: float | None = 20.0


@dataclass
class World:
    config: ScenarioConfig
    tables: dict            # canonical contract tables (what an adapter would provide)
    truth: dict             # hidden ground truth (never given to models)
    batch: GameBatch        # true per-game parameters (rows = games, in truth['games'] order)


def _draw_players(rng, n, pos):
    big = pos == 2
    guard = pos == 0
    return {
        "usage": np.exp(rng.normal(0, 0.30, n)),
        "to_rate": np.clip(rng.normal(0.13, 0.02, n), 0.05, 0.25),
        "ft_rate": np.clip(rng.normal(0.09, 0.03, n), 0.02, 0.25),
        "share3": np.clip(np.where(big, 0.12, 0.42) + rng.normal(0, 0.06, n), 0.0, 0.8),
        "logit2": rng.normal(0.08, 0.22, n),
        "logit3": rng.normal(-0.58, 0.22, n),
        "pft": np.clip(rng.normal(0.77, 0.07, n), 0.45, 0.95),
        "orb_w": np.where(big, 1.3, 0.4) * np.exp(rng.normal(0, 0.2, n)),
        "drb_w": np.where(big, 3.2, 1.4) * np.exp(rng.normal(0, 0.2, n)),
        "ast_w": np.where(guard, 2.5, 0.8) * np.exp(rng.normal(0, 0.3, n)),
        "stl_w": np.exp(rng.normal(0, 0.3, n)),
        "blk_w": np.where(big, 2.0, 0.3) * np.exp(rng.normal(0, 0.3, n)),
    }


def resolve_rotation(avail: np.ndarray):
    """Depth-chart rotation policy: availability (12,) -> starter_idx (5,), bench_idx (5,)."""
    starters = np.arange(5)
    bench = np.arange(5, 10)
    reserves = [r for r in (10, 11) if avail[r]]
    for k in range(5):
        s, b = starters[k], bench[k]
        if not avail[s]:
            if avail[b]:
                starters[k] = b
                bench[k] = reserves.pop(0) if reserves else b
            elif reserves:
                starters[k] = reserves.pop(0)
                bench[k] = reserves.pop(0) if reserves else starters[k]
        elif not avail[b]:
            bench[k] = reserves.pop(0) if reserves else s
    return starters, bench


def _schedule(n_teams, rounds, rng):
    teams = list(range(n_teams))
    base = []
    for _ in range(n_teams - 1):
        base.append([(teams[i], teams[n_teams - 1 - i]) for i in range(n_teams // 2)])
        teams = [teams[0], teams[-1]] + teams[1:-1]
    days = []
    for rd in range(rounds):
        for pairs in base:
            days.append([(a, b) if rd % 2 == 0 else (b, a) for a, b in pairs])
    return [days[i] for i in rng.permutation(len(days))]


class _Registry:
    """Tracks players, roster spells, and the tables that describe them."""

    def __init__(self, rng):
        self.rng = rng
        self.params: dict[str, dict] = {}
        self.player_rows = []
        self.roster_rows = []
        self.open: dict[str, dict] = {}
        self.used = set()

    def pid(self, prefix="P"):
        while True:
            x = f"{prefix}{self.rng.integers(10 ** 6):06d}"
            if x not in self.used:
                self.used.add(x)
                return x

    def new_player(self, pos, kt, usage_mult=1.0):
        pid = self.pid()
        d = {k: float(v[0]) for k, v in _draw_players(self.rng, 1, np.array([pos])).items()}
        d["usage"] *= usage_mult
        self.params[pid] = d
        self.player_rows.append({"player_id": pid, "listed_position": "GWB"[pos], "knowledge_time": kt,
                                 "record_version": 1, **SRC})
        return pid

    def open_spell(self, pid, team_id, slot, valid_from, kt):
        row = {"player_id": pid, "team_id": team_id, "valid_from": valid_from, "valid_to": pd.NaT,
               "depth_chart_slot": slot, "transaction_type": "other", "knowledge_time": kt,
               "record_version": 1, **SRC}
        self.roster_rows.append(row)
        self.open[pid] = row

    def close_spell(self, pid, valid_to, kt):
        row = dict(self.open.pop(pid))
        row.update(valid_to=valid_to, knowledge_time=kt, record_version=2)
        self.roster_rows.append(row)


def generate_world(cfg: ScenarioConfig) -> World:
    rng = np.random.default_rng(cfg.seed)
    T = cfg.n_teams
    reg = _Registry(rng)
    team_ids = [reg.pid("T") for _ in range(T)]
    k0 = EPOCH - pd.Timedelta(days=60)
    roster = np.empty((T, ROSTER), dtype=object)
    for ti in range(T):
        for s in range(ROSTER):
            mult = 2.0 if s == 0 else (0.75 if s >= 5 else 1.0)
            roster[ti, s] = reg.new_player(POS[s], k0, mult)
            reg.open_spell(roster[ti, s], team_ids[ti], s, EPOCH - pd.Timedelta(days=30), k0)
    team_def = rng.normal(0, 0.10, T)
    team_pace = np.exp(rng.normal(0, 0.03, T))
    team_frac = rng.uniform(0.62, 0.76, (T, 5))

    games, inj_rows, changes = [], [], []
    injured: dict[str, int] = {}
    Pg = {k: [] for k in PARAMS}
    for season in range(cfg.n_seasons):
        s_start = EPOCH + pd.Timedelta(days=365 * season)
        if season > 0:
            for d in reg.params.values():
                d["logit2"] += rng.normal(0, cfg.skill_drift_sd)
                d["logit3"] += rng.normal(0, cfg.skill_drift_sd)
                d["usage"] *= np.exp(rng.normal(0, cfg.skill_drift_sd))
            n_new = int(round(cfg.rookie_frac * ROSTER))
            for ti in range(T):
                for s in (rng.choice(np.arange(1, ROSTER), size=n_new, replace=False) if n_new else []):
                    kt = s_start - pd.Timedelta(days=45)
                    reg.close_spell(roster[ti, s], s_start - pd.Timedelta(days=40), kt)
                    pid = reg.new_player(POS[s], kt, 0.75 if s >= 5 else 1.0)
                    roster[ti, s] = pid
                    reg.open_spell(pid, team_ids[ti], s, s_start - pd.Timedelta(days=40), kt)
                    changes.append({"kind": "rookie", "season": season, "day": 0, "player_id": pid,
                                    "team_id": team_ids[ti]})
            injured = {}
        for day, pairs in enumerate(_schedule(T, cfg.rounds_per_season, rng)):
            tip = s_start + pd.Timedelta(days=day)
            for tr in cfg.trades:
                if tr["season"] == season and tr["day"] == day:
                    a, pa, b, pb = tr["team_a"], tr["pos_a"], tr["team_b"], tr["pos_b"]
                    pid_a, pid_b = roster[a, pa], roster[b, pb]
                    kt = tip - pd.Timedelta(hours=20)
                    vt = tip - pd.Timedelta(hours=12)
                    for pid in (pid_a, pid_b):
                        reg.close_spell(pid, vt, kt)
                    roster[a, pa], roster[b, pb] = pid_b, pid_a
                    reg.open_spell(pid_b, team_ids[a], pa, vt, kt)
                    reg.open_spell(pid_a, team_ids[b], pb, vt, kt)
                    changes += [{"kind": "trade", "season": season, "day": day, "player_id": pid_a,
                                 "team_id": team_ids[b], "from_team": team_ids[a]},
                                {"kind": "trade", "season": season, "day": day, "player_id": pid_b,
                                 "team_id": team_ids[a], "from_team": team_ids[b]}]
            for rc in cfg.role_changes:
                if rc["season"] == season and rc["day"] == day:
                    pid = roster[rc["team"], rc["pos"]]
                    reg.params[pid]["usage"] *= rc["usage_mult"]
                    changes.append({"kind": "role_change", "season": season, "day": day, "player_id": pid,
                                    "team_id": team_ids[rc["team"]]})
            for sa in cfg.star_absences:
                if sa["season"] == season and sa["day"] == day:
                    pid = roster[sa["team"], 0]
                    injured[pid] = sa["n_games"]
                    changes.append({"kind": "star_absence", "season": season, "day": day, "player_id": pid,
                                    "team_id": team_ids[sa["team"]], "n_games": sa["n_games"]})
            for h, a in pairs:
                gid = f"G{reg.pid('')}"
                avail = np.ones((2, ROSTER), bool)
                for side, ti in enumerate((h, a)):
                    for s in range(ROSTER):
                        pid = roster[ti, s]
                        rem = injured.get(pid, 0)
                        new_inj = False
                        if rem <= 0 and rng.random() < cfg.injury_hazard:
                            rem = max(1, int(rng.geometric(1 / cfg.injury_mean_len)))
                            new_inj = True
                        if rem > 0:
                            avail[side, s] = False
                            injured[pid] = rem - 1
                            if new_inj and rng.random() < cfg.late_scratch_frac:
                                inj_rows.append((gid, pid, team_ids[ti], "questionable", tip - pd.Timedelta(hours=5)))
                                inj_rows.append((gid, pid, team_ids[ti], "out", tip - pd.Timedelta(minutes=10)))
                            else:
                                inj_rows.append((gid, pid, team_ids[ti], "out", tip - pd.Timedelta(hours=5)))
                        elif rng.random() < cfg.questionable_noise:
                            inj_rows.append((gid, pid, team_ids[ti], "questionable", tip - pd.Timedelta(hours=5)))
                            inj_rows.append((gid, pid, team_ids[ti], "available", tip - pd.Timedelta(minutes=10)))
                ids = np.array([roster[h].copy(), roster[a].copy()])
                for k in PARAMS:
                    Pg[k].append([[reg.params[p][k] for p in ids[0]], [reg.params[p][k] for p in ids[1]]])
                games.append({"game_id": gid, "season": season, "day": day, "tip": tip, "home": h, "away": a,
                              "avail": avail, "ids": ids})

    # ---- simulate all games in one batch ----------------------------------------
    G = len(games)
    st = np.zeros((G, 2, 5), int)
    bn = np.zeros((G, 2, 5), int)
    frac = np.zeros((G, 2, 5))
    for g, gm in enumerate(games):
        for side, ti in enumerate((gm["home"], gm["away"])):
            st[g, side], bn[g, side] = resolve_rotation(gm["avail"][side])
            frac[g, side] = team_frac[ti]
    avail_all = np.array([gm["avail"] for gm in games])
    usage = np.array(Pg["usage"]) * avail_all  # unavailable players can never be chosen
    hh = np.array([gm["home"] for gm in games])
    aa = np.array([gm["away"] for gm in games])
    batch = GameBatch(
        usage=usage, **{k: np.array(Pg[k]) for k in PARAMS if k != "usage"},
        starter_idx=st, bench_idx=bn, starter_frac=frac,
        def_eff=np.stack([team_def[hh], team_def[aa]], 1),
        shoot_latent=rng.normal(0, cfg.game_latent_sd, (G, 2)) if cfg.game_latent_sd > 0 else np.zeros((G, 2)),
        usage_mult=np.exp(rng.normal(0, cfg.usage_form_sd, (G, 2, ROSTER))) if cfg.usage_form_sd > 0
        else np.ones((G, 2, ROSTER)),
        pace_mult=np.sqrt(team_pace[hh] * team_pace[aa]) * np.exp(rng.normal(0, cfg.pace_sd, G)),
        momentum=cfg.momentum, blowout_margin=cfg.blowout_margin,
    )
    res = simulate(batch, rng, log_events=True)
    tables, truth = _emit_tables(cfg, rng, games, res, batch, team_ids, reg, inj_rows, changes)
    return World(config=cfg, tables=tables, truth=truth, batch=batch)


def _emit_tables(cfg, rng, games, res, batch, team_ids, reg, inj_rows, changes):
    G = len(games)
    final_kt = [gm["tip"] + pd.Timedelta(minutes=150 + int(rng.integers(0, 30))) for gm in games]
    ruleset = pd.DataFrame([{"ruleset_id": "SYN-1", "league": "OTHER", "effective_from": EPOCH.date(),
                             "effective_to": None, "periods": 4, "period_length_s": 720,
                             "overtime_length_s": 300, "shot_clock_s": 24, "three_point_distance_m": 7.24,
                             "personal_foul_limit": 99, "team_foul_bonus_rule": "none",
                             "knowledge_time": EPOCH - pd.Timedelta(days=400), **SRC}])
    team = pd.DataFrame([{"team_id": t, "franchise_id": t, "league": "OTHER", "valid_from": EPOCH.date(),
                          "knowledge_time": EPOCH - pd.Timedelta(days=400), **SRC} for t in team_ids])
    game_rows, tg_rows, pg_rows, sl_rows = [], [], [], []
    for g, gm in enumerate(games):
        sched_kt = EPOCH + pd.Timedelta(days=365 * gm["season"]) - pd.Timedelta(days=30)
        base = {"game_id": gm["game_id"], "league": "OTHER", "season_id": f"SYN-{gm['season']}",
                "phase": "regular", "ruleset_id": "SYN-1", "scheduled_tip": gm["tip"],
                "home_team_id": team_ids[gm["home"]], "away_team_id": team_ids[gm["away"]],
                "neutral_site": False, **SRC}
        game_rows.append({**base, "status": "scheduled", "periods_played": np.nan, "knowledge_time": sched_kt,
                          "record_version": 1})
        game_rows.append({**base, "status": "final", "periods_played": int(res.periods[g]),
                          "knowledge_time": final_kt[g], "record_version": 2})
        for side in range(2):
            tid = team_ids[gm["home"] if side == 0 else gm["away"]]
            bx = res.box[g, side]
            tg_rows.append({"game_id": gm["game_id"], "team_id": tid, "is_home": side == 0,
                            "pts": int(res.score[g, side]),
                            **{k.lower(): int(bx[:, SI[k]].sum()) for k in
                               ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK",
                                "TOV", "PF"]},
                            "possessions": int(res.possessions[g, side]), "knowledge_time": final_kt[g],
                            "record_version": 1, **SRC})
            starters = set(batch.starter_idx[g, side].tolist())
            for r in range(ROSTER):
                pid = gm["ids"][side, r]
                sec = int(round(bx[r, SI["SEC"]]))
                status = "played" if sec > 0 else ("inactive" if not gm["avail"][side, r] else "dnp_coach")
                pg_rows.append({"game_id": gm["game_id"], "player_id": pid, "team_id": tid, "status": status,
                                "started": r in starters and gm["avail"][side, r], "seconds": sec,
                                **{k.lower(): int(bx[r, SI[k]]) for k in
                                   ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST",
                                    "STL", "BLK", "TOV", "PF"]},
                                "knowledge_time": final_kt[g], "record_version": 1, **SRC})
                if r in starters and gm["avail"][side, r]:
                    sl_rows.append({"game_id": gm["game_id"], "team_id": tid, "player_id": pid,
                                    "knowledge_time": gm["tip"] - pd.Timedelta(minutes=30), **SRC})
    inj = pd.DataFrame(inj_rows, columns=["game_id", "player_id", "team_id", "status", "knowledge_time"])
    inj["report_id"] = inj["knowledge_time"].astype(str) + "_" + inj["team_id"]
    inj["reason_category"] = "injury"
    for k, v in SRC.items():
        inj[k] = v

    # ---- possessions (segments) from the event log --------------------------------
    ev = res.events
    gids = np.array([gm["game_id"] for gm in games])
    ids = np.array([gm["ids"] for gm in games])       # (G, 2, R)
    row = ev["row"].to_numpy()
    off = ev["off"].to_numpy()
    dfp = pd.DataFrame({
        "game_id": gids[row],
        "off_team_id": [team_ids[games[r]["home"] if o == 0 else games[r]["away"]] for r, o in zip(row, off)],
        "def_team_id": [team_ids[games[r]["away"] if o == 0 else games[r]["home"]] for r, o in zip(row, off)],
        "off_is_home": off == 0,
        "start_elapsed_s": ev["t"].to_numpy(), "duration_s": ev["dur"].to_numpy(),
        "start_type": np.array(START_TYPES)[ev["start"].to_numpy()],
        "event_code": ev["event"].to_numpy(), "made": ev["made"].to_numpy(), "ftm": ev["ftm"].to_numpy(),
        "points": ev["pts"].to_numpy(), "orb": ev["orb"].to_numpy(), "steal": ev["steal"].to_numpy(),
        "assisted": ev["assisted"].to_numpy(),
        "user_player_id": ids[row, off, ev["user"].to_numpy()],
        "assister_id": np.where(ev["assister"].to_numpy() >= 0,
                                ids[row, off, np.maximum(ev["assister"].to_numpy(), 0)], None),
        "rebounder_id": np.where(ev["orb"].to_numpy() == 1, ids[row, off, np.maximum(ev["rebounder"], 0)],
                                 np.where(ev["drb"].to_numpy() == 1,
                                          ids[row, 1 - off, np.maximum(ev["rebounder"], 0)], None)),
        "stealer_id": np.where(ev["stealer"].to_numpy() >= 0,
                               ids[row, 1 - off, np.maximum(ev["stealer"].to_numpy(), 0)], None),
        "blocked": ev["blocked"].to_numpy(),
        "blocker_id": np.where(ev["blocker"].to_numpy() >= 0,
                               ids[row, 1 - off, np.maximum(ev["blocker"].to_numpy(), 0)], None),
        "fouler_id": np.where(ev["fouler"].to_numpy() >= 0,
                              ids[row, 1 - off, np.maximum(ev["fouler"].to_numpy(), 0)], None),
        "off_margin_start": ev["margin"].to_numpy().astype(int),
        "prev_scored": ev["prev_scored"].to_numpy().astype(int),
    })
    ol = np.stack([ids[row, off, ev[f"ol{k}"].to_numpy()] for k in range(5)], 1)
    dl = np.stack([ids[row, 1 - off, ev[f"dl{k}"].to_numpy()] for k in range(5)], 1)
    dfp["off_lineup"] = [tuple(x) for x in ol]
    dfp["def_lineup"] = [tuple(x) for x in dl]
    dfp["end_type"] = np.select(
        [dfp.event_code == 0, dfp.event_code == 1, dfp.made == 1, dfp.orb == 1],
        ["turnover_live", "ft_trip", "made_fg", "missed_fg_oreb"], "missed_fg_dreb")
    dfp.loc[(dfp.event_code == 0) & (dfp.steal == 0), "end_type"] = "turnover_dead"
    dfp["segment"] = (dfp["start_type"] == "after_oreb").astype(int)
    dfp["possession_seq"] = dfp.groupby("game_id").cumcount()
    kt_map = dict(zip(gids, final_kt))
    dfp["knowledge_time"] = dfp["game_id"].map(kt_map)
    for k, v in SRC.items():
        dfp[k] = v

    tables = {
        "RULESET": ruleset, "TEAM": team, "PLAYER": pd.DataFrame(reg.player_rows),
        "ROSTER": pd.DataFrame(reg.roster_rows), "GAME": pd.DataFrame(game_rows),
        "TEAM_GAME": pd.DataFrame(tg_rows), "PLAYER_GAME": pd.DataFrame(pg_rows),
        "INJURY_REPORT": inj, "STARTING_LINEUP": pd.DataFrame(sl_rows), "POSSESSION": dfp,
    }
    truth = {
        "games": pd.DataFrame([{"game_id": gm["game_id"], "season": gm["season"], "day": gm["day"],
                                "tip": gm["tip"], "home_team_id": team_ids[gm["home"]],
                                "away_team_id": team_ids[gm["away"]]} for gm in games]),
        "changes": pd.DataFrame(changes),
        "game_ids_order": gids,
        "roster_ids": ids,
        "sim": res,
        "config": asdict(cfg),
        "team_ids": team_ids,
    }
    return tables, truth
