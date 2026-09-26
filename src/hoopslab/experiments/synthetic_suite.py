"""Synthetic verification suite: do the instruments diagnose known mechanisms correctly?

Run:  python -m hoopslab.experiments.synthetic_suite [--quick] [--seed N] [--out results/]

Each check returns {"check", "scenario", "expected", "diagnosis", "passed", ...details}.
A failure means an INSTRUMENT is broken or underpowered. It says nothing about real
basketball.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..adapters.base import build_store
from ..adapters.synthetic import SyntheticAdapter
from ..backtest.runner import BacktestSpec, outcomes_table, paired, run_backtest, score_forecasts
from ..backtest.strata import compute_strata
from ..diagnostics import joint as J
from ..diagnostics.changepoint import cusum_flags
from ..diagnostics.realism import realism_report, rpit
from ..metrics import compare as cp
from ..metrics import scoring as sc
from ..metrics.information import (bits_per_event, decode, encode, plugin_conditional_entropy,
                                   plugin_entropy, usable_information)
from ..models.baselines import MinutesRate, PlayerMean
from ..models.event import EventDesign, MultinomialLogit, event_classes
from ..models.structural import PossessionSimulator
from ..synthetic.scenarios import EVAL_SEASON, EXPECTED, scenario
from ..synthetic.world import EPOCH, generate_world
from .history import run_history_ablation

SEASON_START = EPOCH + pd.Timedelta(days=365 * EVAL_SEASON)


def _world_store(name, seed):
    w = generate_world(scenario(name, seed))
    store, report = build_store(SyntheticAdapter(w))
    return w, store


def _season_split(w):
    P = w.tables["POSSESSION"]
    season = P["game_id"].map(w.truth["games"].set_index("game_id")["season"])
    return P[season < EVAL_SEASON], P[season == EVAL_SEASON]


def _window(d0, d1):
    return BacktestSpec(start=SEASON_START + pd.Timedelta(days=d0), end=SEASON_START + pd.Timedelta(days=d1))


# ---------------------------------------------------------------------------- H2/H3 history
def check_history(name, seed=0, B=300):
    w, _ = _world_store(name, seed)
    tr, te = _season_split(w)
    r = run_history_ablation(tr, te, B=B, seed=seed)
    exp = EXPECTED[name]["history"]
    return {"check": "history_ablation", "scenario": name, "expected": exp, "diagnosis": r["diagnosis"],
            "passed": r["diagnosis"] == exp,
            "millibits": {k: round(1000 * r[k]["bits"], 3) for k in
                          ("history_signal", "latent_signal", "history_residual", "gain_B")},
            "ci_millibits": {k: [round(1000 * r[k]["ci_low"], 3), round(1000 * r[k]["ci_high"], 3)] for k in
                             ("history_signal", "latent_signal", "history_residual")},
            "n_test_events": r["n_test"]}


# ---------------------------------------------------------------------------- H7 usable info vs plug-in
def check_usable_information(seed=0, B=300):
    """A context with no information must show a fake plug-in gain but no held-out gain;
    a real context (start type) must show a held-out gain; a permuted real context must not."""
    w, _ = _world_store("A_no_history", seed)
    tr, te = _season_split(w)
    ytr, yte = event_classes(tr), event_classes(te)
    d = EventDesign(tr, noise_levels=2000)
    btr, bte = d.all_blocks(tr), d.all_blocks(te)
    cl = te["game_id"].to_numpy()
    rng = np.random.default_rng(seed)

    def fit_bits(Xtr, Xte, y_tr=ytr):
        m = MultinomialLogit(l2=1e-3).fit(Xtr, y_tr)
        return bits_per_event(m.predict_proba(Xte), yte)

    one_tr, one_te = d.select(btr, []), d.select(bte, [])
    base = fit_bits(one_tr, one_te)
    noise = fit_bits(d.select(btr, ["NOISE"]), d.select(bte, ["NOISE"]))
    import scipy.sparse as sps
    st_cols = slice(0, 3)  # start-type one-hot = first 3 columns of the S block
    start_tr = sps.hstack([one_tr, btr["S"][:, st_cols]], format="csr")
    start_te = sps.hstack([one_te, bte["S"][:, st_cols]], format="csr")
    start = fit_bits(start_tr, start_te)
    perm = rng.permutation(tr.shape[0])
    placebo = fit_bits(sps.hstack([one_tr, btr["S"][perm][:, st_cols]], format="csr"), start_te)
    noise_cell_te = np.asarray(bte["NOISE"].argmax(1)).ravel()
    plug_gain = plugin_entropy(yte, 8) - plugin_conditional_entropy(yte, noise_cell_te)
    ui_noise = usable_information(base, noise, cl, B=B, rng=rng)
    ui_start = usable_information(base, start, cl, B=B, rng=rng)
    ui_placebo = usable_information(base, placebo, cl, B=B, rng=rng)
    ok = (plug_gain > 0.05) and (ui_noise["ci_high"] < 0.0005) and (ui_start["ci_low"] > 0) and \
        (ui_placebo["ci_high"] < 0.0005)
    return {"check": "usable_information_vs_plugin", "scenario": "A_no_history",
            "expected": "PLUGIN_FAKE_GAIN; HELDOUT_NONE_FOR_NOISE; HELDOUT_POSITIVE_FOR_START_TYPE; PLACEBO_NONE",
            "diagnosis": ("PLUGIN_FAKE_GAIN; HELDOUT_NONE_FOR_NOISE; HELDOUT_POSITIVE_FOR_START_TYPE; PLACEBO_NONE"
                          if ok else "MISMATCH"),
            "passed": bool(ok), "plugin_fake_gain_bits": round(plug_gain, 4),
            "heldout_noise_millibits": [round(1000 * ui_noise[k], 3) for k in ("bits", "ci_low", "ci_high")],
            "heldout_start_millibits": [round(1000 * ui_start[k], 3) for k in ("bits", "ci_low", "ci_high")],
            "placebo_millibits": [round(1000 * ui_placebo[k], 3) for k in ("bits", "ci_low", "ci_high")]}


# ---------------------------------------------------------------------------- compression = log loss
def check_compression(seed=0, n_events=6000):
    w, _ = _world_store("A_no_history", seed)
    tr, te = _season_split(w)
    ytr, yte = event_classes(tr), event_classes(te)
    d = EventDesign(tr)
    btr, bte = d.all_blocks(tr), d.all_blocks(te)
    te_idx = np.arange(min(n_events, len(te)))
    res = {}
    for name, blocks in (("intercept", []), ("state", ["S"])):
        m = MultinomialLogit().fit(d.select(btr, blocks), ytr)
        Pm = m.predict_proba(d.select(bte, blocks))[te_idx]
        bits, ideal = encode(yte[te_idx], Pm)
        ok_rt = bool(np.array_equal(decode(bits, Pm), yte[te_idx]))
        res[name] = {"code_bits": len(bits), "ideal_bits": round(ideal, 1),
                     "logloss_bits": round(float(bits_per_event(Pm, yte[te_idx]).sum()), 1), "lossless": ok_rt}
    ok = all(r["lossless"] and abs(r["code_bits"] - r["ideal_bits"]) <= 3 + 1e-3 * r["ideal_bits"]
             for r in res.values()) and (res["state"]["code_bits"] < res["intercept"]["code_bits"]) == \
        (res["state"]["logloss_bits"] < res["intercept"]["logloss_bits"])
    return {"check": "compression_equals_logloss", "scenario": "A_no_history",
            "expected": "LOSSLESS; CODE_LENGTH~=LOGLOSS; ORDER_PRESERVED",
            "diagnosis": "LOSSLESS; CODE_LENGTH~=LOGLOSS; ORDER_PRESERVED" if ok else "MISMATCH",
            "passed": bool(ok), **res}


# ---------------------------------------------------------------------------- stratified (D, E, F)
def _paired_with_strata(scores, strata, ref, cand, stratum, target, mode, metric):
    m = paired(scores, ref, cand, target, mode, metric)
    s = strata.drop_duplicates(["game_id", "player_id"])
    m = m.merge(s[["game_id", "player_id", stratum]], on=["game_id", "player_id"], how="left")
    m["flag"] = m[stratum].fillna(False).astype(bool)
    return m


def _interaction_from_pairs(m, metric="crps", B=1000):
    flag = m["flag"].to_numpy()
    r = cp.stratum_interaction(m[f"{metric}_ref"], m[f"{metric}_cand"], m["cluster"].astype(str), flag, B=B)
    skill_in = 1 - m.loc[flag, f"{metric}_cand"].sum() / m.loc[flag, f"{metric}_ref"].sum()
    skill_out = 1 - m.loc[~flag, f"{metric}_cand"].sum() / m.loc[~flag, f"{metric}_ref"].sum()
    return {**r, "skill_in": float(skill_in), "skill_out": float(skill_out)}


def _interaction(scores, strata, ref, cand, stratum, target="PTS", mode="A", metric="crps", B=1000):
    return _interaction_from_pairs(_paired_with_strata(scores, strata, ref, cand, stratum, target, mode, metric),
                                   metric, B)


def check_role_change(seed=0, d1=40):
    w, store = _world_store("D_role_change", seed)
    spec = _window(0, d1)
    models = [PlayerMean(None), PlayerMean(5.0, name="L2_recent_hl5")]
    fc, req = run_backtest(store, spec, models)
    scores = score_forecasts(store, fc, ["PTS"], modes=("A",))
    strata = compute_strata(store, req)
    inter = _interaction(scores, strata, "L1_player_mean", "L2_recent_hl5", "role_change_alarm")
    # change-point detection accuracy against the hidden truth (instrument check)
    ch = w.truth["changes"]
    changed = set(ch.loc[ch["kind"] == "role_change", "player_id"])
    pg = store.full_table("PLAYER_GAME").merge(
        store.full_table("GAME").drop_duplicates("game_id", keep="last")[["game_id", "scheduled_tip"]], on="game_id")
    pg = pg[(pg["seconds"] > 0) & (pg["scheduled_tip"] >= SEASON_START - pd.Timedelta(days=400))]
    detected, false_alarm = 0, 0
    change_time = SEASON_START + pd.Timedelta(days=8)
    others = 0
    for pid, g in pg.sort_values("scheduled_tip").groupby("player_id"):
        f = cusum_flags(g)
        after = g["scheduled_tip"] >= change_time
        win = after & (g["scheduled_tip"] <= change_time + pd.Timedelta(days=12))
        if pid in changed:
            detected += bool(f.loc[win, "alarm"].any())
        else:
            others += 1
            false_alarm += bool(f.loc[g["scheduled_tip"] >= SEASON_START, "alarm"].any())
    det_rate = detected / max(len(changed), 1)
    fa_rate = false_alarm / max(others, 1)
    localised = inter["ci_low"] > 0
    ok = localised and det_rate >= 0.66 and fa_rate <= 0.25
    return {"check": "stratified_role_change", "scenario": "D_role_change",
            "expected": "ADVANTAGE_LOCALISED_IN_STRATUM; DETECTED",
            "diagnosis": ("ADVANTAGE_LOCALISED_IN_STRATUM" if localised else "NOT_LOCALISED") +
            ("; DETECTED" if det_rate >= 0.66 and fa_rate <= 0.25 else "; DETECTION_POOR"),
            "passed": bool(ok), "interaction": inter, "detection_rate": det_rate, "false_alarm_rate": fa_rate}


def _sim_vs_direct(name, stratum, seed=0, d0=0, d1=28, n_sims=300, n_worlds=1):
    """Structural (L10) vs direct (L4) interaction test. With n_worlds > 1, independent
    synthetic worlds are pooled (clusters = world x day), the synthetic analogue of more seasons."""
    pairs = {"PTS": [], "MIN": []}
    n_in = 0
    for k in range(n_worlds):
        sd = seed + 1000 * k
        w, store = _world_store(name, sd)
        spec = _window(d0, d1)
        models = [MinutesRate(shrink=True), PossessionSimulator(n_sims=n_sims, seed=sd)]
        fc, req = run_backtest(store, spec, models)
        scores = score_forecasts(store, fc, ["PTS", "MIN"], modes=("U",))
        strata = compute_strata(store, req)
        n_in += int(strata[stratum].sum())
        for t in pairs:
            m = _paired_with_strata(scores, strata, "L4_hier_shrink", "L10_sim", stratum, t, "U", "crps")
            m["cluster"] = f"w{k}_" + m["cluster"].astype(str)
            pairs[t].append(m)
    res = {t: _interaction_from_pairs(pd.concat(v, ignore_index=True)) for t, v in pairs.items()}
    localised = res["PTS"]["ci_low"] > 0
    return {"check": f"stratified_{stratum}", "scenario": name, "expected": EXPECTED[name]["stratified"],
            "diagnosis": "ADVANTAGE_LOCALISED_IN_STRATUM" if localised else "NOT_LOCALISED",
            "passed": bool(localised), "interaction": res, "n_in_stratum": n_in, "n_worlds": n_worlds}


def check_star_absence(seed=0):
    return _sim_vs_direct("E_star_absence", "teammate_star_out", seed)


def check_trade(seed=0):
    # primary H6c stratum is player-level (own_move_21d); a team-level stratum dilutes the
    # effect, and one world is underpowered (see docs/prereg/09 "instrument calibration notes")
    return _sim_vs_direct("F_trade", "own_move_21d", seed, d0=0, d1=26, n_worlds=2)


# ---------------------------------------------------------------------------- G parameter uncertainty
def check_param_uncertainty(seed=0, d1=20, B=1000):
    w, store = _world_store("G_param_uncertainty", seed)
    spec = _window(0, d1)
    models = [MinutesRate(shrink=True, param_uncertainty=True),
              MinutesRate(shrink=True, param_uncertainty=False)]
    fc, req = run_backtest(store, spec, models)
    scores = score_forecasts(store, fc, ["PTS", "REB", "AST", "PRA"], modes=("A",))
    strata = compute_strata(store, req)
    s = scores.merge(strata.drop_duplicates(["game_id", "player_id"])[["game_id", "player_id", "low_sample"]],
                     on=["game_id", "player_id"])
    out = {}
    for mdl in ("L4_hier_shrink", "L4_hier_shrink_plugin"):
        g = s[(s.model == mdl) & s.low_sample]
        out[mdl] = {"coverage80_low_sample": round(sc.pit_coverage(g["pit"].to_numpy(), 0.8), 4),
                    "dispersion_low_sample": round(sc.pit_dispersion_ratio(g["pit"].to_numpy()), 3),
                    "coverage80_rest": round(sc.pit_coverage(s[(s.model == mdl) & ~s.low_sample]["pit"].to_numpy(),
                                                             0.8), 4),
                    "n_low": int(len(g))}
    # bootstrap CI for the coverage gap of the plug-in in the low-sample stratum
    g = s[(s.model == "L4_hier_shrink_plugin") & s.low_sample]
    inside = ((g["pit"] > 0.1) & (g["pit"] < 0.9)).astype(float).to_numpy()
    est, draws = cp.cluster_bootstrap({"in": inside, "n": np.ones_like(inside)}, g["cluster"].to_numpy(),
                                      lambda z: z["in"].sum(-1) / z["n"].sum(-1), B=B, block=1)
    hi = float(np.quantile(draws, 0.975))
    ok = hi < 0.8 and out["L4_hier_shrink"]["coverage80_low_sample"] > out["L4_hier_shrink_plugin"][
        "coverage80_low_sample"]
    return {"check": "parameter_uncertainty", "scenario": "G_param_uncertainty",
            "expected": EXPECTED["G_param_uncertainty"]["param_uncertainty"],
            "diagnosis": "PLUGIN_OVERCONFIDENT_IN_LOW_SAMPLE_STRATUM" if ok else "NOT_DETECTED",
            "passed": bool(ok), "models": out, "plugin_cov80_ci_high": hi}


# ---------------------------------------------------------------------------- H tails vs latent
def _team_pits(fc_list, store, rng):
    tg = store.full_table("TEAM_GAME")
    home = tg[tg["is_home"].astype(bool)].set_index("game_id")["pts"]
    away = tg[~tg["is_home"].astype(bool)].set_index("game_id")["pts"]
    out = {"home_pts": [], "total_points": [], "margin": []}
    for f in fc_list:
        for gid, T in f.team.items():
            out["home_pts"].append(rpit(T["home_pts"], home[gid], rng))
            out["total_points"].append(rpit(T["home_pts"] + T["away_pts"], home[gid] + away[gid], rng))
            out["margin"].append(rpit(T["home_pts"] - T["away_pts"], home[gid] - away[gid], rng))
    return out


def check_tails(seed=0, d1=35, n_sims=300, B=300):
    w, store = _world_store("H_tail_latent", seed)
    # event level: how many bits does the latent carry per event?
    tr, te = _season_split(w)
    r = run_history_ablation(tr, te, sets={"A": ["S"], "E": ["S", "EW"], "F": ["S", "GL"],
                                           "G": ["S", "GL", "EW"], "B": ["S", "L1"], "C": ["S", "L1"],
                                           "D": ["S", "L1"]}, B=B)
    spec = _window(0, d1)
    models = [PossessionSimulator(n_sims=n_sims, game_latent="none", seed=seed),
              PossessionSimulator(n_sims=n_sims, game_latent="estimate", seed=seed)]
    fc, _ = run_backtest(store, spec, models)
    rng = np.random.default_rng(seed)
    rep = {m: realism_report(_team_pits(fc[m], store, rng)).set_index("summary").to_dict("index")
           for m in fc}
    scores = score_forecasts(store, fc, ["PTS"], modes=("A",))
    pc = {m: round(sc.pit_coverage(scores[scores.model == m]["pit"].to_numpy(), 0.9), 4) for m in fc}
    no_lat = rep["L10_sim"]["home_pts"]
    lat = rep["L11_sim_latent"]["home_pts"]
    event_bits = r["latent_signal"]["bits"]
    ok = (no_lat["dispersion"] > 1.25 and not no_lat["passed"]) and lat["passed"] and event_bits < 0.02
    return {"check": "tails_game_latent", "scenario": "H_tail_latent",
            "expected": EXPECTED["H_tail_latent"]["tails"],
            "diagnosis": "EVENT_OK_AGGREGATE_UNDERDISPERSED_WITHOUT_LATENT" if ok else "NOT_DIAGNOSED",
            "passed": bool(ok), "event_latent_millibits": round(1000 * event_bits, 3),
            "team_pts_realism": {"L10_sim": no_lat, "L11_sim_latent": lat},
            "player_pts_coverage90": pc}


# ---------------------------------------------------------------------------- H8 joint + realism
def check_joint_and_realism(seed=0, d1=35, n_sims=300, B=1000):
    w, store = _world_store("A_no_history", seed)
    spec = _window(0, d1)
    # the "correct" simulator carries parameter uncertainty (a plug-in L10 has too-narrow
    # margins, which the realism gates flag; see docs/prereg/09 calibration notes)
    good = PossessionSimulator(n_sims=n_sims, seed=seed, param_uncertainty=True, name="L10_sim", rung="L10b")
    bad = PossessionSimulator(n_sims=n_sims, seed=seed, pace_scale=0.90, blowout_margin=None,
                              name="L10_sim_misspecified", rung="L10")
    fc, req = run_backtest(store, spec, [good, bad])
    outc = outcomes_table(store)
    rng = np.random.default_rng(seed)
    joint = {}
    for f in fc["L10_sim"]:
        joint.update(f.joint)
    team = {}
    for f in fc["L10_sim"]:
        team.update(f.team)
    alt = {g: J.independent_copula(v["samples"], rng) for g, v in joint.items() if v["samples"] is not None}
    # 1) marginal CRPS identical (same marginals by construction)
    ti = joint[next(iter(joint))]["targets"].index("PTS")
    crps_diff = 0.0
    for g, v in joint.items():
        for j, pid in enumerate(v["player_ids"]):
            y = outc.set_index(["game_id", "player_id"])["PTS"].get((g, pid), 0)
            crps_diff = max(crps_diff, abs(sc.crps_samples(v["samples"][:, j, ti], y) -
                                           sc.crps_samples(alt[g][:, j, ti], y)))
    # 2) within-team variogram score (MIN: teammates' minutes are strongly dependent)
    res = {}
    for tgt, scale in (("MIN", 10.0), ("PTS", 8.0)):
        a = J.team_vector_scores(joint, outc, tgt, scale, rng)
        b = J.team_vector_scores(joint, outc, tgt, scale, rng, alt_samples=alt)
        m = a.merge(b, on=["game_id", "team_id"], suffixes=("_joint", "_ind"))
        cl = m["game_id"].map(lambda g: g)  # cluster = game
        c = cp.compare_losses(m["variogram_ind"], m["variogram_joint"], cl, B=B, block=1)
        e = cp.compare_losses(m["energy_ind"], m["energy_joint"], cl, B=B, block=1)
        res[tgt] = {"variogram_skill": round(c.effect, 4), "ci": [round(c.ci_low, 4), round(c.ci_high, 4)],
                    "energy_skill": round(e.effect, 4), "energy_ci": [round(e.ci_low, 4), round(e.ci_high, 4)]}
    # 3) derived-quantity calibration (PRA) and pair dependence
    pra_joint = J.derived_pit(joint, outc, ["PTS", "REB", "AST"], rng)
    pra_ind = J.derived_pit(joint, outc, ["PTS", "REB", "AST"], rng, alt_samples=alt)
    taus = J.pair_tau_calibration(joint, outc, [("teammates", "MIN", "MIN"), ("same_player", "MIN", "PTS"),
                                                ("same_player", "PTS", "AST"), ("teammates", "PTS", "PTS")], rng)
    taus_ind = J.pair_tau_calibration(joint, outc, [("teammates", "MIN", "MIN"), ("same_player", "MIN", "PTS")],
                                      rng, alt_samples=alt)
    tg = store.full_table("TEAM_GAME")
    blow = J.blowout_minutes_check(joint, team, outc, tg)
    # 4) realism gates: correct simulator passes, misspecified fails
    real = {}
    for m in fc:
        real[m] = realism_report(_team_pits(fc[m], store, rng)).set_index("summary")
    tj = taus.set_index("pair").loc["teammates:MIN~MIN"]
    ti_ = taus_ind.set_index("pair").loc["teammates:MIN~MIN"]
    tau_ok = abs(tj["tau_pred"] - tj["tau_real"]) <= 0.10 and \
        abs(ti_["tau_pred"] - ti_["tau_real"]) > abs(tj["tau_pred"] - tj["tau_real"]) + 0.10
    blow_ok = abs(blow["corr_pred_mean"] - blow["corr_real"]) <= 0.10
    # primary joint score (pre-registered after this calibration): energy score on MIN vectors
    ok_joint = res["MIN"]["energy_ci"][0] > 0 and crps_diff < 1e-9 and tau_ok and blow_ok
    ok_real = bool(real["L10_sim"]["passed"].all()) and not bool(real["L10_sim_misspecified"]["passed"].all())
    return {"check": "joint_and_realism", "scenario": "A_no_history",
            "expected": "JOINT_BEATS_INDEPENDENT_WITH_IDENTICAL_MARGINALS; REALISM_GATES_DISCRIMINATE",
            "diagnosis": ("JOINT_BEATS_INDEPENDENT_WITH_IDENTICAL_MARGINALS" if ok_joint else "JOINT_NOT_DETECTED") +
            ("; REALISM_GATES_DISCRIMINATE" if ok_real else "; REALISM_GATES_FAIL"),
            "passed": bool(ok_joint and ok_real), "max_marginal_crps_diff": crps_diff, "team_vector": res,
            "pra_dispersion": {"joint": round(sc.pit_dispersion_ratio(pra_joint), 3),
                               "independent": round(sc.pit_dispersion_ratio(pra_ind), 3)},
            "pair_tau": taus.round(3).to_dict("records"), "pair_tau_independent": taus_ind.round(3).to_dict("records"),
            "blowout_minutes": blow,
            "realism": {m: r.round(3).reset_index().to_dict("records") for m, r in real.items()}}


CHECKS = {
    "history_A": lambda s: check_history("A_no_history", s),
    "history_B": lambda s: check_history("B_true_history", s),
    "history_C": lambda s: check_history("C_latent_momentum", s),
    "usable_information": check_usable_information,
    "compression": check_compression,
    "role_change_D": check_role_change,
    "star_absence_E": check_star_absence,
    "trade_F": check_trade,
    "param_uncertainty_G": check_param_uncertainty,
    "tails_H": check_tails,
    "joint_and_realism": check_joint_and_realism,
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results/synthetic_suite")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, fn in CHECKS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        r = fn(a.seed)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(_jsonable(r))
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {name}: {r['diagnosis']} ({r['seconds']}s)", flush=True)
    (out / f"seed{a.seed}.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
