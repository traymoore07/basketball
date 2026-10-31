"""E-HIST summary: pooled held-out bits per event over every prequential month, the pre-registered contrasts
with the frozen game-cluster bootstrap (B = 2000), the frozen diagnosis, per-season contrasts, the three
model families on identical test events, the controls and the hot-hand diagnostic.

Writes results/p2_hist/summary.json and summary_tables.md. Nothing here fits a model."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.experiments.history import MPE_BITS, diagnose_history  # noqa: E402
from hoopslab.metrics.information import usable_information  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_hist")
B = 2000
SEED = 12345
SETS = ["A", "B", "C", "D", "E", "F", "G"]


def load_family(fam):
    fs = sorted(glob.glob(os.path.join(OUT, "bits", f"{fam}_*.parquet")))
    if not fs:
        return None, {}
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    meta = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "bits", f"{fam}_*.json")))}
    return df, meta


def contrast(df, a, b, rng):
    r = usable_information(df[a].to_numpy(), df[b].to_numpy(), df["game_id"].to_numpy(), B=B, rng=rng)
    r["millibits"] = 1000 * r["bits"]
    r["ci_low_millibits"], r["ci_high_millibits"] = 1000 * r["ci_low"], 1000 * r["ci_high"]
    return r


def family_summary(df, rng, prefix="bits_"):
    res = {"n_events": int(len(df)), "n_games": int(df["game_id"].nunique()), "n_months": int(df["month"].nunique()),
           "mean_bits": {s: float(df[f"{prefix}{s}"].mean()) for s in SETS if f"{prefix}{s}" in df}}
    c = {}
    c["history_signal"] = contrast(df, f"{prefix}A", f"{prefix}E", rng)
    c["latent_signal"] = contrast(df, f"{prefix}A", f"{prefix}F", rng)
    c["history_residual"] = contrast(df, f"{prefix}F", f"{prefix}G", rng)
    for k in ("B", "C", "D"):
        c[f"gain_{k}"] = contrast(df, f"{prefix}A", f"{prefix}{k}", rng)
    c["gain_E_vs_D"] = contrast(df, f"{prefix}D", f"{prefix}E", rng)
    c["gain_G_vs_E"] = contrast(df, f"{prefix}E", f"{prefix}G", rng)
    res.update(c)
    res["diagnosis"] = diagnose_history(res)
    if f"{prefix}E_perm" in df:
        res["negative_control_E_perm_vs_A"] = contrast(df, f"{prefix}A", f"{prefix}E_perm", rng)
        res["negative_control_G_perm_vs_F"] = contrast(df, f"{prefix}F", f"{prefix}G_perm", rng)
    if f"{prefix}CANARY" in df:
        res["canary_vs_A"] = contrast(df, f"{prefix}A", f"{prefix}CANARY", rng)
    return res


def hot_hand(seg_files):
    """P(make | previous FGA made) - P(make | previous FGA missed), pooled across games vs within game (same offence)."""
    rows = []
    for f in seg_files:
        P = pd.read_parquet(f)
        P = P[P["season_type"].isin(["regular-season", "playoffs"]) & (P["event_code"] >= 2)].sort_values(["game_id", "off_team_id", "possession_seq"])
        prev = P.groupby(["game_id", "off_team_id"], sort=False)["made"].shift(1)
        d = P.assign(prev=prev).dropna(subset=["prev"])
        rows.append(d[["game_id", "season", "made", "prev"]])
    d = pd.concat(rows, ignore_index=True)
    pooled = float(d.loc[d["prev"] == 1, "made"].mean() - d.loc[d["prev"] == 0, "made"].mean())
    g = d.groupby("game_id").apply(lambda x: (x.loc[x["prev"] == 1, "made"].mean() - x.loc[x["prev"] == 0, "made"].mean(), len(x)) if (x["prev"] == 1).any() and (x["prev"] == 0).any() else (np.nan, 0), include_groups=False)
    wg = pd.DataFrame(g.tolist(), columns=["diff", "n"]).dropna()
    within = float(np.average(wg["diff"], weights=wg["n"]))
    by_season = d.groupby("season").apply(lambda x: x.loc[x["prev"] == 1, "made"].mean() - x.loc[x["prev"] == 0, "made"].mean(), include_groups=False)
    return {"n_pairs": int(len(d)), "pooled_diff": pooled, "within_game_mean_diff": within, "by_season_pooled": {int(k): float(v) for k, v in by_season.items()},
            "note": "within-game differences carry the Miller-Sanjurjo small-sample bias (negative in expectation under no hot hand); the pooled estimate mixes between-game heterogeneity"}


def md_contrast(name, c):
    return f"| {name} | {c['millibits']:+.3f} | [{c['ci_low_millibits']:+.3f}, {c['ci_high_millibits']:+.3f}] | {c['p_value']:.4f} | {c['n']:,} |"


def main():
    rng = np.random.default_rng(SEED)
    res = {"mpe_bits": MPE_BITS, "bootstrap": {"B": B, "cluster": "game", "seed": SEED}, "families": {}}
    L = ["# E-HIST summary (NBA nba_stats segments, development seasons 2000-2020, prequential by month)", ""]
    mnl, meta = load_family("mnl")
    if mnl is not None:
        fam = family_summary(mnl, rng)
        fam["months"] = sorted(mnl["month"].unique().tolist())
        fam["unconverged_fits"] = {m: [k for k, v in d["converged"].items() if not v] for m, d in meta.items() if any(not v for v in d["converged"].values())}
        res["families"]["mnl"] = fam
        per = {}
        for S, d in mnl.groupby("season"):
            r = {"n_events": int(len(d)), "n_games": int(d["game_id"].nunique())}
            for k, (a, b) in {"history_signal": ("A", "E"), "latent_signal": ("A", "F"), "history_residual": ("F", "G"), "gain_B": ("A", "B")}.items():
                r[k] = contrast(d, f"bits_{a}", f"bits_{b}", np.random.default_rng(SEED + int(S)))
            r["diagnosis"] = diagnose_history(r)
            per[int(S)] = r
        res["families"]["mnl"]["per_season"] = per
        L += ["## 1. Frozen multinomial logit (L2 = 1e-4), pooled over all months", "",
              f"events {fam['n_events']:,}, games {fam['n_games']:,}, months {fam['n_months']}; mean held-out bits per event: " +
              ", ".join(f"{s} {v:.4f}" for s, v in fam["mean_bits"].items()), "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|"]
        for k in ("gain_B", "gain_C", "gain_D", "gain_E_vs_D", "history_signal", "latent_signal", "history_residual", "gain_G_vs_E"):
            L.append(md_contrast(k, fam[k]))
        if "negative_control_E_perm_vs_A" in fam:
            L.append(md_contrast("NEGATIVE CONTROL: permuted history (E_perm vs A)", fam["negative_control_E_perm_vs_A"]))
            L.append(md_contrast("NEGATIVE CONTROL: permuted history beyond latent (G_perm vs F)", fam["negative_control_G_perm_vs_F"]))
        if "canary_vs_A" in fam:
            L.append(md_contrast("POSITIVE CONTROL (canary: next segment class)", fam["canary_vs_A"]))
        L += ["", f"**Frozen diagnosis (MPE {1000*MPE_BITS} millibits): {fam['diagnosis']}**", "",
              "### Per season (mnl)", "", "| season | events | games | history_signal (mb) [CI] | latent_signal (mb) | history_residual (mb) [CI] | gain_B (mb) | diagnosis |", "|---|---|---|---|---|---|---|---|"]
        for S, r in per.items():
            h, l, hr, gb = r["history_signal"], r["latent_signal"], r["history_residual"], r["gain_B"]
            L.append(f"| {S} | {r['n_events']:,} | {r['n_games']} | {h['millibits']:+.2f} [{h['ci_low_millibits']:+.2f}, {h['ci_high_millibits']:+.2f}] | {l['millibits']:+.2f} | "
                     f"{hr['millibits']:+.2f} [{hr['ci_low_millibits']:+.2f}, {hr['ci_high_millibits']:+.2f}] | {gb['millibits']:+.2f} | {r['diagnosis']} |")
    for fam_name in ("gbm",):
        df, meta = load_family(fam_name)
        if df is None:
            continue
        fam = family_summary(df, rng)
        res["families"][fam_name] = fam
        L += ["", f"## 2. {fam_name.upper()} replication (fixed parameters), pooled", "",
              f"events {fam['n_events']:,}, months {fam['n_months']}; mean bits: " + ", ".join(f"{s} {v:.4f}" for s, v in fam["mean_bits"].items()), "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|"]
        for k in ("gain_B", "gain_D", "history_signal", "latent_signal", "history_residual"):
            L.append(md_contrast(k, fam[k]))
        if "canary_vs_A" in fam:
            L.append(md_contrast("canary", fam["canary_vs_A"]))
        L += ["", f"diagnosis: {fam['diagnosis']}"]
    gru, _ = load_family("gru")
    if gru is not None and mnl is not None:
        cols = [c for c in ("bits_A_nn", "bits_F_nn", "bits_E_gru", "bits_G_gru") if c in gru]
        j = mnl.merge(gru[["game_id", "possession_seq"] + cols], on=["game_id", "possession_seq"], how="inner").dropna(subset=cols)
        r = {"n_events": int(len(j)), "months": int(j["month"].nunique()),
             "history_signal_nn (A_nn vs E_gru)": contrast(j, "bits_A_nn", "bits_E_gru", rng),
             "latent_signal_nn (A_nn vs F_nn)": contrast(j, "bits_A_nn", "bits_F_nn", rng),
             "history_residual_nn (F_nn vs G_gru)": contrast(j, "bits_F_nn", "bits_G_gru", rng),
             "architecture: A_nn vs mnl A": contrast(j, "bits_A", "bits_A_nn", rng),
             "architecture: E_gru vs mnl E": contrast(j, "bits_E", "bits_E_gru", rng),
             "mean_bits": {"mnl_A": float(j["bits_A"].mean()), "mnl_E": float(j["bits_E"].mean()), "mnl_F": float(j["bits_F"].mean()),
                           "A_nn": float(j["bits_A_nn"].mean()), "F_nn": float(j["bits_F_nn"].mean()), "E_gru": float(j["bits_E_gru"].mean()), "G_gru": float(j["bits_G_gru"].mean())}}
        rr = {"history_signal": r["history_signal_nn (A_nn vs E_gru)"], "latent_signal": r["latent_signal_nn (A_nn vs F_nn)"], "history_residual": r["history_residual_nn (F_nn vs G_gru)"]}
        r["diagnosis_nn"] = diagnose_history(rr)
        res["families"]["gru"] = r
        L += ["", "## 3. GRU sequence model (sets E and G) against the frozen logit on identical test events", "",
              f"events {r['n_events']:,}, months {r['months']}; mean bits: " + ", ".join(f"{k} {v:.4f}" for k, v in r["mean_bits"].items()), "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|"]
        for k in ("history_signal_nn (A_nn vs E_gru)", "latent_signal_nn (A_nn vs F_nn)", "history_residual_nn (F_nn vs G_gru)", "architecture: A_nn vs mnl A", "architecture: E_gru vs mnl E"):
            L.append(md_contrast(k, r[k]))
        L.append("")
        L.append(f"neural-family diagnosis: {r['diagnosis_nn']}")
    att, _ = load_family("attrib")
    if att is not None:
        r = {"n_events": int(len(att)), "months": sorted(att["month"].unique().tolist())}
        r["history_residual_baseline (F vs G)"] = contrast(att, "bits_F", "bits_G", rng)
        for v in ("fouls", "stint", "lineup", "gap", "all"):
            r[f"history_residual_after_{v} (F_{v} vs G_{v})"] = contrast(att, f"bits_F_{v}", f"bits_G_{v}", rng)
            r[f"slow_variable_gain_{v} (F vs F_{v})"] = contrast(att, "bits_F", f"bits_F_{v}", rng)
        if mnl is not None:
            j = att.merge(mnl[["game_id", "possession_seq", "bits_E"]], on=["game_id", "possession_seq"], how="inner")
            r["previous_class_gain (E vs E_PC)"] = contrast(j, "bits_E", "bits_E_PC", rng)
        r["previous_class_gain_beyond_latent_and_slow (G_all vs G_all_PC)"] = contrast(att, "bits_G_all", "bits_G_all_PC", rng)
        r["previous_class_gain_beyond_latent (G vs G_PC)"] = contrast(att, "bits_G", "bits_G_PC", rng)
        res["attribution"] = r
        L += ["", f"## 5. H3b attribution (frozen multinomial logit; every sixth test month: {len(r['months'])} months, {r['n_events']:,} events)", "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|"]
        for k, v in r.items():
            if isinstance(v, dict) and "millibits" in v:
                L.append(md_contrast(k, v))
    can, _ = load_family("canary2")
    if can is not None:
        c = contrast(can, "bits_A", "bits_A_CANARY2", rng)
        res["positive_control_canary2"] = {"n_events": int(len(can)), "months": int(can["month"].nunique()), "gain": c}
        L += ["", f"## 6. Positive control (canary 2: next segment's start type and margin change; {can['month'].nunique()} months, {len(can):,} events)", "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|", md_contrast("A + canary2 vs A", c)]
    ex, _ = load_family("explore_tm")
    if ex is not None:
        r = {"n_events": int(len(ex)), "months": sorted(ex["month"].unique().tolist()), "note": "EXPLORATORY post-hoc block (clock x margin), not pre-registered; no decision rests on it"}
        for k, (a, b) in {"TM_gain_on_state (A vs A_TM)": ("A", "A_TM"), "history_signal_with_TM (A_TM vs G_TM minus latent)": ("A_TM", "G_TM"),
                          "history_residual_baseline (F vs G)": ("F", "G"), "history_residual_with_TM (F_TM vs G_TM)": ("F_TM", "G_TM"),
                          "history_residual_with_TM_and_slow (F_all_TM vs G_all_TM)": ("F_all_TM", "G_all_TM")}.items():
            r[k] = contrast(ex, f"bits_{a}", f"bits_{b}", rng)
        late = ex[ex["late"]]
        r["late_only (4Q/OT): history_residual_with_TM (F_TM vs G_TM)"] = contrast(late, "bits_F_TM", "bits_G_TM", rng)
        r["late_only (4Q/OT): history_residual_baseline (F vs G)"] = contrast(late, "bits_F", "bits_G", rng)
        res["exploratory_time_margin"] = r
        L += ["", f"## 7. EXPLORATORY (not pre-registered): clock x margin block added to the frozen sets ({len(r['months'])} months, {r['n_events']:,} events)", "",
              "| contrast | millibits/event | 95% CI | p | n |", "|---|---|---|---|---|"]
        for k, v in r.items():
            if isinstance(v, dict) and "millibits" in v:
                L.append(md_contrast(k, v))
    seg_files = sorted(glob.glob(os.path.join(OUT, "segments_20[0-2][0-9].parquet")))
    seg_files = [f for f in seg_files if 2000 <= int(os.path.basename(f)[9:13]) <= 2020]
    if seg_files:
        hh = hot_hand(seg_files)
        res["hot_hand_diagnostic"] = hh
        L += ["", "## 4. Hot-hand contrast (diagnostic only)", "",
              f"pairs {hh['n_pairs']:,}; pooled P(make|prev make) - P(make|prev miss) = {hh['pooled_diff']:+.4f}; within-game (game-weighted mean) = {hh['within_game_mean_diff']:+.4f}. {hh['note']}"]
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "summary_tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
