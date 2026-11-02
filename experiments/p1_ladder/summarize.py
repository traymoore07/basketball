"""P1 summary: credit chain, decisions, calibration, strata, gates, expectations vs observed.

Reads results/p1_ladder/scores_{S}_part*.parquet, requests_{S}.parquet, strata_{S}.parquet.
Writes results/p1_ladder/summary.json and summary_tables.md. Development seasons only are
reported; burn-in seasons 1998-1999 enter the L2 half-life selection and nothing else.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from hoopslab.metrics import compare as cp  # noqa: E402
from hoopslab.metrics import scoring as sc  # noqa: E402

OUT = os.path.join(ROOT, "results", "p1_ladder")
CFG = json.load(open(os.path.join(ROOT, "experiments", "p1_ladder", "config.json")))
DEV = list(range(CFG["windows"]["development"][0], CFG["windows"]["development"][1] + 1))
SEL = [1998, 1999]
TARGETS = CFG["scoring"]["targets"]
HL = CFG["models"]["L2_grid"]["half_lives_appearances"]
L2 = {h: f"L2_recent_hl{h}" for h in HL}
MODELS = ["L0_climatology", "L1_player_mean"] + list(L2.values()) + ["L3_minutes_rate", "L4_hier_shrink", "L4_hier_shrink_plugin"]
RUNG = {"L0": "L0_climatology", "L1": "L1_player_mean", "L2": "L2_selected", "L3": "L3_minutes_rate", "L4": "L4_hier_shrink",
        "L4p": "L4_hier_shrink_plugin"}
METRICS = ["crps", "log", "rps", "ae_median", "se_mean", "tw_crps"]
B, BLOCK, SEED = CFG["scoring"]["bootstrap"]["B"], CFG["scoring"]["bootstrap"]["block_days"], CFG["scoring"]["bootstrap"]["seed"]
MPE = 0.01


def load_season(S):
    parts = sorted(glob.glob(os.path.join(OUT, f"scores_{S}_part*.parquet")))
    sco = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    req = pd.read_parquet(os.path.join(OUT, f"requests_{S}.parquet"))
    st = pd.read_parquet(os.path.join(OUT, f"strata_{S}.parquet"))
    key = ["game_id", "player_id"]
    flags = req.drop_duplicates(key)[key + ["phase"]].merge(
        st.drop_duplicates(key)[key + ["low_sample", "role_change_alarm"]], on=key, how="left")
    sco = sco.merge(flags, on=key, how="left")
    sco["low_sample"] = sco["low_sample"].fillna(False).astype(bool)
    sco["role_change_alarm"] = sco["role_change_alarm"].fillna(False).astype(bool)
    sco["regular"] = sco["phase"].eq("regular")
    return sco


def cluster_sums(sco, flag_col=None):
    """Per (season, cluster, mode, target, model[, flag]) sums of every metric and counts."""
    keys = ["season", "cluster", "mode", "target", "model"] + ([flag_col] if flag_col else [])
    agg = {m: "sum" for m in METRICS if m in sco}
    agg["y"] = "size"
    out = sco.groupby(keys, observed=True).agg(agg).rename(columns={"y": "n"}).reset_index()
    return out


def wide(cs, metric, mode, targets, flag=None, flag_val=None, seasons=None):
    """{target: {model: array over clusters}} with a shared cluster index, plus the cluster keys."""
    d = cs[cs["mode"] == mode]
    if flag is not None:
        d = d[d[flag] == flag_val]
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    piv = d.pivot_table(index=["season", "cluster"], columns=["target", "model"], values=metric, aggfunc="sum")
    piv = piv.fillna(0.0).sort_index()
    return piv


def core_skill_boot(piv, ref_models: dict, cand_models: dict, targets, rng):
    """Bootstrap the core aggregate (mean over targets of 1 - cand/ref) over game-day clusters."""
    vals = {}
    for t in targets:
        vals[f"r_{t}"] = piv[(t, ref_models[t])].to_numpy()
        vals[f"c_{t}"] = piv[(t, cand_models[t])].to_numpy()
    cluster = np.arange(len(piv))

    def stat(s):
        return np.mean([1.0 - s[f"c_{t}"].sum(-1) / s[f"r_{t}"].sum(-1) for t in targets], axis=0)

    est, draws = cp.cluster_bootstrap(vals, cluster, stat, B=B, block=BLOCK, rng=rng)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p = (1 + np.sum(draws <= 0)) / (len(draws) + 1)
    return {"effect": float(est), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p), "n_clusters": int(len(piv))}


def target_skill_boot(piv, t, ref, cand, rng):
    r, c = piv[(t, ref)].to_numpy(), piv[(t, cand)].to_numpy()
    cmp_ = cp.compare_losses(r, c, np.arange(len(r)), metric="crps", reference=ref, candidate=cand, B=B, block=BLOCK, rng=rng)
    return {"effect": cmp_.effect, "ci_low": cmp_.ci_low, "ci_high": cmp_.ci_high, "p_value": cmp_.p_value,
            "mean_ref": cmp_.mean_ref, "mean_cand": cmp_.mean_cand}


def main():
    seasons = [s for s in SEL + DEV if os.path.exists(os.path.join(OUT, f"done_{s}.json"))]
    print("seasons found:", seasons)
    cs_all, cs_low, cs_role, pit_rows = [], [], [], []
    for S in seasons:
        sco = load_season(S)
        reg = sco[sco["regular"]]
        cs = cluster_sums(sco.assign(regular=sco["regular"]), "regular")
        cs_all.append(cs)
        cs_low.append(cluster_sums(reg, "low_sample"))
        cs_role.append(cluster_sums(reg, "role_change_alarm"))
        # PIT diagnostics: regular season, per model x target x mode (and low_sample split)
        for (m, t, mode, low), g in reg.groupby(["model", "target", "mode", "low_sample"], observed=True):
            pit = g["pit"].to_numpy()
            pit_rows.append({"season": S, "model": m, "target": t, "mode": mode, "low_sample": bool(low), "n": len(pit),
                             "pit_sum": float(pit.sum()), "pit_sq": float((pit ** 2).sum()),
                             "cov50": float(((pit > .25) & (pit < .75)).sum()), "cov80": float(((pit > .1) & (pit < .9)).sum()),
                             "cov95": float(((pit > .025) & (pit < .975)).sum()),
                             "hist": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
                             "hist1000": np.histogram(pit, bins=1000, range=(0, 1))[0].tolist(),
                             "mean_y": float(g["y"].mean()), "mean_fc": float(g["mean"].mean())})
        print("season", S, "units", len(sco), flush=True)
        del sco, reg
    cs = pd.concat(cs_all, ignore_index=True)
    cs.to_parquet(os.path.join(OUT, "cluster_sums.parquet"), index=False)
    cs_low = pd.concat(cs_low, ignore_index=True)
    cs_role = pd.concat(cs_role, ignore_index=True)
    pits = pd.DataFrame(pit_rows)
    pits.to_parquet(os.path.join(OUT, "pit_summaries.parquet"), index=False)
    res = {"seasons_reported": DEV, "selection_only": SEL}

    # ---- L2 half-life selection (rule fixed in config.json before any outcome) -----------------
    regcs = cs[cs["regular"]]
    tot = regcs.groupby(["season", "mode", "target", "model"])["crps"].sum().unstack("model")
    sel, grid = {}, {}
    for s in DEV:
        prior = [p for p in seasons if p < s]
        crit = {}
        for h, name in L2.items():
            ratios = []
            for t in TARGETS:
                sub = tot.loc[[(p, "U", t) for p in prior if (p, "U", t) in tot.index]]
                ratios.append(sub[name].sum() / sub["L1_player_mean"].sum())
            crit[h] = float(np.mean(ratios))
        sel[s] = min(crit, key=crit.get)
        grid[s] = crit
    res["L2_selection"] = {"half_life_by_season": sel, "criterion_mean_crps_ratio_vs_L1_prior_seasons": grid}

    def l2_name(s):
        return L2[sel[s]]

    # assemble the selected-L2 rows into the cluster-sum tables
    def add_selected(frame):
        rows = []
        for s in DEV:
            r = frame[(frame["season"] == s) & (frame["model"] == l2_name(s))].copy()
            r["model"] = "L2_selected"
            rows.append(r)
        return pd.concat([frame] + rows, ignore_index=True)

    cs, cs_low, cs_role = add_selected(cs), add_selected(cs_low), add_selected(cs_role)
    pit_sel = []
    for s in DEV:
        r = pits[(pits["season"] == s) & (pits["model"] == l2_name(s))].copy()
        r["model"] = "L2_selected"
        pit_sel.append(r)
    pits = pd.concat([pits] + pit_sel, ignore_index=True)

    rng = np.random.default_rng(SEED)
    chain = ["L0", "L1", "L2", "L3", "L4"]
    piv = {mode: wide(cs, "crps", mode, TARGETS, "regular", True, DEV) for mode in ("U", "A")}
    # mean losses per model x target x mode (regular, dev)
    mean_loss = {}
    for mode in ("U", "A"):
        n = wide(cs, "n", mode, TARGETS, "regular", True, DEV)
        for met in ["crps", "log", "rps", "ae_median", "se_mean", "tw_crps"]:
            w = wide(cs, met, mode, TARGETS, "regular", True, DEV)
            for (t, m) in w.columns:
                nn = n[(t, m)].sum()
                val = w[(t, m)].sum() / nn if nn else float("nan")
                if met == "se_mean":
                    val = float(np.sqrt(val))
                mean_loss.setdefault(mode, {}).setdefault(t, {}).setdefault(m, {})[met] = float(val)
            mean_loss[mode]["_n_units"] = {t: int(n[(t, "L0_climatology")].sum()) for t in TARGETS}
    res["mean_loss_regular_dev"] = mean_loss

    # ---- credit chain: immediately simpler and best simpler (development window) --------------
    comps = {}
    for mode in ("U", "A"):
        P = piv[mode]
        core_loss = {r: float(np.mean([P[(t, RUNG[r])].sum() for t in TARGETS])) for r in chain}
        # "best simpler" by core skill vs L0 on the development window
        def core_vs_l0(r):
            return float(np.mean([1 - P[(t, RUNG[r])].sum() / P[(t, RUNG["L0"])].sum() for t in TARGETS]))
        core_rel = {r: core_vs_l0(r) for r in chain}
        out = []
        for i, r in enumerate(chain[1:], 1):
            prev = chain[i - 1]
            best = max(chain[:i], key=lambda k: core_rel[k])
            for ref_r, kind in ((prev, "vs_immediately_simpler"), (best, "vs_best_simpler")):
                if kind == "vs_best_simpler" and ref_r == prev:
                    continue
                c = core_skill_boot(P, {t: RUNG[ref_r] for t in TARGETS}, {t: RUNG[r] for t in TARGETS}, TARGETS, rng)
                per_t = {t: target_skill_boot(P, t, RUNG[ref_r], RUNG[r], rng) for t in TARGETS}
                out.append({"candidate": r, "reference": ref_r, "kind": kind, "core": c, "per_target": per_t})
        # Holm within the immediately-simpler family (4 comparisons), decisions with the default MPE
        fam = [o for o in out if o["kind"] == "vs_immediately_simpler"]
        adj = cp.holm([o["core"]["p_value"] for o in fam])
        for o, pa in zip(fam, adj):
            o["core"]["p_adj"] = float(pa)
            o["core"]["mpe"] = MPE
            o["core"]["decision"] = cp.decide(o["core"]["effect"], o["core"]["ci_low"], o["core"]["ci_high"], MPE, pa)
            # no-harm check on secondary metrics (log score bits, RPS) at the aggregate level
            harm = {}
            for met in ("log", "rps"):
                w = wide(cs, met, mode, TARGETS, "regular", True, DEV)
                dif = []
                for t in TARGETS:
                    if (t, RUNG[o["candidate"]]) not in w.columns:
                        continue
                    r_, c_ = w[(t, RUNG[o["reference"]])].to_numpy(), w[(t, RUNG[o["candidate"]])].to_numpy()
                    cm = cp.compare_losses(r_, c_, np.arange(len(r_)), B=B, block=BLOCK, rng=rng, absolute=(met == "log"))
                    dif.append({"target": t, "effect": cm.effect, "ci_low": cm.ci_low, "ci_high": cm.ci_high})
                harm[met] = dif
            o["secondary"] = harm
            o["core"]["harmful_secondary"] = [f"{met}:{d['target']}" for met, ds in harm.items() for d in ds if d["ci_high"] < 0]
        comps[mode] = {"core_loss_sum_per_cluster_mean": core_loss, "core_skill_vs_L0": core_rel, "comparisons": out}
        # L2 grid and L4-plugin as diagnostics (not in the credit chain)
        comps[mode]["L2_grid_core_skill_vs_L1"] = {
            name: float(np.mean([1 - P[(t, name)].sum() / P[(t, "L1_player_mean")].sum() for t in TARGETS])) for name in L2.values()}
        comps[mode]["L4_plugin_vs_L4_core"] = core_skill_boot(P, {t: RUNG["L4"] for t in TARGETS}, {t: RUNG["L4p"] for t in TARGETS},
                                                               TARGETS, rng)
    res["credit_chain"] = comps

    # ---- calibration (regular season, dev) ----------------------------------------------------
    def pit_stats(df):
        n = df["n"].sum()
        if n == 0:
            return None
        mean = df["pit_sum"].sum() / n
        var = df["pit_sq"].sum() / n - mean ** 2
        h = np.sum(np.stack(df["hist1000"].to_list()), axis=0)
        F = np.cumsum(h) / n
        grid = np.arange(1, 1001) / 1000
        ks = float(max(np.max(np.abs(F - grid)), np.max(np.abs(F - grid + 0.001))))
        return {"n": int(n), "pit_mean": float(mean), "dispersion_ratio": float(12 * var), "ks": ks,
                "cov50": float(df["cov50"].sum() / n), "cov80": float(df["cov80"].sum() / n), "cov95": float(df["cov95"].sum() / n),
                "hist10": (np.sum(np.stack(df["hist"].to_list()), axis=0) / n).round(4).tolist(),
                "bias_mean_fc_minus_y": float((df["mean_fc"] * df["n"]).sum() / n - (df["mean_y"] * df["n"]).sum() / n)}
    cal = {}
    pd_dev = pits[pits["season"].isin(DEV)]
    for (m, t, mode), g in pd_dev.groupby(["model", "target", "mode"]):
        cal.setdefault(mode, {}).setdefault(t, {})[m] = pit_stats(g)
    cal_low = {}
    for (m, t, mode, low), g in pd_dev.groupby(["model", "target", "mode", "low_sample"]):
        cal_low.setdefault(mode, {}).setdefault(t, {}).setdefault(m, {})["low_sample" if low else "rest"] = pit_stats(g)
    res["calibration_regular_dev"] = cal
    res["calibration_by_low_sample"] = cal_low

    # ---- strata (reported, no H6 interaction) ---------------------------------------------------
    strata = {}
    # per season core skill of each rung vs L0 and vs previous rung (mode U, regular)
    per_season = {}
    for s in [x for x in DEV if x in seasons]:
        Ps = wide(cs, "crps", "U", TARGETS, "regular", True, [s])
        d = {"n_clusters": int(len(Ps))}
        for i, r in enumerate(chain):
            d[f"{r}_core_vs_L0"] = float(np.mean([1 - Ps[(t, RUNG[r])].sum() / Ps[(t, RUNG["L0"])].sum() for t in TARGETS]))
            if i:
                d[f"{r}_core_vs_{chain[i-1]}"] = float(np.mean([1 - Ps[(t, RUNG[r])].sum() / Ps[(t, RUNG[chain[i-1]])].sum() for t in TARGETS]))
            d[f"{r}_MIN_U_crps"] = float(Ps[("MIN", RUNG[r])].sum() / wide(cs, "n", "U", ["MIN"], "regular", True, [s])[("MIN", RUNG[r])].sum())
        d["L2_half_life"] = sel[s]
        per_season[s] = d
    strata["per_season_regular_U"] = per_season
    # postseason vs regular
    post = {}
    for mode in ("U",):
        Pp = wide(cs, "crps", mode, TARGETS, "regular", False, DEV)
        post[mode] = {f"{r}_core_vs_{chain[i-1]}": float(np.mean([1 - Pp[(t, RUNG[r])].sum() / Pp[(t, RUNG[chain[i-1]])].sum() for t in TARGETS]))
                      for i, r in enumerate(chain) if i}
        post[mode]["n_clusters"] = int(len(Pp))
    strata["postseason_U"] = post
    # low_sample: L4 vs L3 inside/outside, L4 vs L4-plugin coverage gate
    low = {}
    for mode in ("U", "A"):
        for flag in (True, False):
            Pl = wide(cs_low, "crps", mode, TARGETS, "low_sample", flag, DEV)
            low[f"{mode}_{'low' if flag else 'rest'}"] = {
                "n_clusters": int(len(Pl)),
                "L4_vs_L3_core": core_skill_boot(Pl, {t: RUNG["L3"] for t in TARGETS}, {t: RUNG["L4"] for t in TARGETS}, TARGETS, rng),
                "L4_vs_L3_per_target": {t: float(1 - Pl[(t, RUNG["L4"])].sum() / Pl[(t, RUNG["L3"])].sum()) for t in TARGETS}}
    strata["low_sample"] = low
    # role_change_alarm: L2 vs L1 inside/outside (X-L2-L1 note), no H6 statistic
    role = {}
    for mode in ("U", "A"):
        for flag in (True, False):
            Pr = wide(cs_role, "crps", mode, TARGETS, "role_change_alarm", flag, DEV)
            role[f"{mode}_{'alarm' if flag else 'rest'}"] = {
                "n_clusters": int(len(Pr)),
                "L2_vs_L1_core": core_skill_boot(Pr, {t: RUNG["L1"] for t in TARGETS}, {t: RUNG["L2"] for t in TARGETS}, TARGETS, rng)}
    strata["role_change_alarm"] = role
    res["strata"] = strata

    # ---- gates ---------------------------------------------------------------------------------
    PU = piv["U"]
    l1_vs_l0 = {t: float(1 - PU[(t, RUNG["L1"])].sum() / PU[(t, RUNG["L0"])].sum()) for t in TARGETS}
    l3_vs_l2_min = float(1 - PU[("MIN", RUNG["L3"])].sum() / PU[("MIN", RUNG["L2"])].sum())
    cov_low = {m: cal_low["A"]["PTS"].get(m, {}).get("low_sample", {}) for m in (RUNG["L4"], RUNG["L4p"])}
    gates = {"L1_gg_L0_every_core_target": {"per_target": l1_vs_l0, "passed": all(v > 0.20 for v in l1_vs_l0.values())},
             "L3_ge_L2_on_MIN_U": {"skill": l3_vs_l2_min, "passed": l3_vs_l2_min >= 0},
             "L4plugin_cov80_low_le_L4": {"cov80_low_sample_PTS_A": {m: c.get("cov80") for m, c in cov_low.items()},
                                          "passed": (cov_low[RUNG["L4p"]].get("cov80", 1) <= cov_low[RUNG["L4"]].get("cov80", 0))}}
    res["gates"] = gates
    from constraints import apply as apply_constraints  # frozen no-harm + calibration-floor downgrades (08 section 5)
    res = apply_constraints(res)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res["gates"], indent=1, default=float))
    print("L2 selection:", sel)
    for mode in ("U", "A"):
        print(mode, "core skill vs L0:", {k: round(v, 4) for k, v in res["credit_chain"][mode]["core_skill_vs_L0"].items()})
        for o in res["credit_chain"][mode]["comparisons"]:
            c = o["core"]
            print(f"  {o['candidate']} vs {o['reference']} ({o['kind']}): {c['effect']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] p={c['p_value']:.4f}",
                  c.get("decision", ""))


if __name__ == "__main__":
    main()
