"""E-STATE summary (docs/p2/02_estate_spec.md sections 5-6): pooled held-out bits per set, the add-one chain,
residual history per state, the information decomposition, the previous-class proxy test, controls, per-season
tables, diagnostics of the primary residual, the GBM replication on identical events (chain, leave-one-out,
logit vs GBM per set), the frozen H2 decision and the regime classification. Game-cluster bootstrap B = 2000,
seed 12345. Nothing here fits a model. Writes results/p2_state/summary.json and summary_tables.md."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.metrics.information import usable_information  # noqa: E402
from run_state import months_from_gbm  # noqa: E402
from state_blocks import BLOCK_ORDER  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_state")
B, SEED, MPE = 2000, 12345, 0.5
STATE_BLOCKS = BLOCK_ORDER[:-1]
LABEL = {"FB": "team fouls / bonus", "GAP": "gap since previous segment", "TM": "late clock x margin", "SC": "finer score + early clock x margin",
         "PF": "player fouls", "ROT": "rotation (stints, lineup changes)", "PC": "previous segment class"}


def load_family(fam, months=None):
    fs = sorted(glob.glob(os.path.join(OUT, "bits", f"{fam}_*.parquet")))
    allowed = set(months_from_gbm()) if months is None else set(months)  # the pre-specified months only; the timing month 1999-12 never enters
    fs = [f for f in fs if os.path.basename(f)[len(fam) + 1:-8] in allowed]
    if not fs:
        return None, {}
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    meta = {os.path.basename(f)[len(fam) + 1:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "bits", f"{fam}_*.json")))
            if os.path.basename(f)[len(fam) + 1:-5] in set(df["month"])}
    return df, meta


def contrast(df, a, b, rng, cluster="game_id"):
    r = usable_information(df[f"bits_{a}"].to_numpy(), df[f"bits_{b}"].to_numpy(), df[cluster].to_numpy(), B=B, rng=rng)
    return {"mb": 1000 * r["bits"], "lo": 1000 * r["ci_low"], "hi": 1000 * r["ci_high"], "p": r["p_value"], "n": r["n"]}


def fmt(c):
    return f"{c['mb']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def chain_summary(df, meta, rng, with_bridge):
    res = {"n_events": int(len(df)), "n_games": int(df["game_id"].nunique()), "n_months": int(df["month"].nunique()),
           "mean_bits": {c[5:]: float(df[c].mean()) for c in df.columns if c.startswith("bits_")}}
    cols = {}
    for mt in meta.values():
        for k, v in mt.get("columns", {}).items():
            cols.setdefault(k, []).append(v)
    res["columns_median"] = {k: int(np.median(v)) for k, v in cols.items()}
    conv = {}
    for mt in meta.values():
        for k, v in mt.get("converged", {}).items():
            conv.setdefault(k, []).append(bool(v))
    res["converged_share"] = {k: float(np.mean(v)) for k, v in conv.items()}
    res["gain"] = {f"S{k}": contrast(df, f"S{k-1}", f"S{k}", rng) for k in range(1, 8)}
    res["residual"] = {f"S{k}": contrast(df, f"S{k}", f"S{k}_H", rng) for k in range(8)}
    if with_bridge:
        res["residual_beyond_latent"] = {f"S{k}": contrast(df, f"S{k}_GL", f"S{k}_H", rng) for k in (0, 6, 7) if f"bits_S{k}_GL" in df}
        res["latent_gain"] = {f"S{k}": contrast(df, f"S{k}", f"S{k}_GL", rng) for k in (0, 6, 7) if f"bits_S{k}_GL" in df}
    res["total_S0_to_S7_H"] = contrast(df, "S0", "S7_H", rng)
    res["pc_on_S0"] = contrast(df, "S0", "S0_PC", rng)
    res["pc_on_S6"] = res["gain"]["S7"]
    res["pc_beyond_history_on_S0"] = contrast(df, "S0_H", "S0_PC_H", rng)
    res["S0_PC_vs_S4"] = contrast(df, "S0_PC", "S4", rng)
    res["history_on_S0_PC"] = contrast(df, "S0_PC", "S0_PC_H", rng)
    if "bits_S6_permstate" in df:
        res["control_state_permuted_vs_S0"] = contrast(df, "S0", "S6_permstate", rng)
        res["control_history_permuted_on_S6"] = contrast(df, "S6", "S6_permhist", rng)
    if "bits_S6_no_FB" in df:
        res["leave_one_out_from_S6"] = {b: contrast(df, f"S6_no_{b}", "S6", rng) for b in STATE_BLOCKS}
    # frozen H2: loss of S0 against S7, threshold max(0.5 mb, 10% of I(S7) over the class-frequency model)
    loss = contrast(df, "S0", "S7", rng)
    info = contrast(df, "NULL", "S7", rng)
    thr = max(MPE, 0.1 * info["mb"])
    res["H2_frozen"] = {"loss_S0_vs_S7": loss, "I_S7_over_null": info, "threshold_mb": thr,
                        "decision": "SUPPORTED" if loss["hi"] < thr else ("FALSIFIED" if loss["lo"] > thr else "INCONCLUSIVE")}
    return res


def regime(r):
    if r["hi"] < MPE:
        return "A"
    if r["lo"] >= 2 * MPE:
        return "C"
    return "B"


def per_season(df, rng):
    rows = []
    for s, d in df.groupby("season"):
        row = {"season": int(s), "events": int(len(d)), "games": int(d["game_id"].nunique())}
        for k in range(1, 8):
            row[f"g{k}"] = contrast(d, f"S{k-1}", f"S{k}", rng)["mb"]
        row["r0"] = contrast(d, "S0", "S0_H", rng)
        row["r6"] = contrast(d, "S6", "S6_H", rng)
        row["r7"] = contrast(d, "S7", "S7_H", rng)
        rows.append(row)
    return rows


def diagnostics(df):
    d = df.copy()
    d["r6"] = d["bits_S6"] - d["bits_S6_H"]
    d["r0"] = d["bits_S0"] - d["bits_S0_H"]
    d["phase"] = np.select([d["period"] <= 3, d["period"] >= 4], ["periods 1-3", "4Q/OT"], default="")
    d["margin"] = pd.cut(d["off_margin_start"].abs(), [-1, 5, 10, 15, 999], labels=["0-5", "6-10", "11-15", "16+"])
    out = {}
    for key in ("phase", "margin", "start_type", "y", "def_in_bonus"):
        g = d.groupby(key, observed=True)
        out[str(key)] = {str(k): {"share": float(len(v) / len(d)), "r0_mb": float(1000 * v["r0"].mean()), "r6_mb": float(1000 * v["r6"].mean()),
                                  "r6_contribution_mb": float(1000 * v["r6"].sum() / len(d))} for k, v in g}
    return out


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nba", "wnba"], default="nba", help="wnba: portability run in results/p2_state/wnba (logit only, every month present)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    months = None
    if a.league == "wnba":
        OUT = os.path.join(OUT, "wnba")
        months = sorted(os.path.basename(f)[4:11] for f in glob.glob(os.path.join(OUT, "bits", "mnl_*.parquet")))
    mnl, meta_m = load_family("mnl", months)
    if mnl is None:
        raise SystemExit("no mnl months")
    S = {"mnl": chain_summary(mnl, meta_m, rng, True)}
    S["mnl"]["per_season"] = per_season(mnl, rng)
    S["diagnostics_mnl"] = diagnostics(mnl)
    gbm, meta_g = load_family("gbm", months=set(mnl["month"]))
    if gbm is not None:
        S["gbm"] = chain_summary(gbm, meta_g, rng, False)
        both = mnl.merge(gbm, on=["game_id", "possession_seq", "month"], suffixes=("_mnl", "_gbm"))
        S["identical_events"] = {"n_events": int(len(both)), "n_months": int(both["month"].nunique())}
        for name in [f"S{k}" for k in range(8)] + [f"S{k}_H" for k in range(8)]:
            S["identical_events"][f"logit_minus_gbm_{name}"] = contrast(both.rename(columns={"bits_" + name + "_mnl": "bits_a", "bits_" + name + "_gbm": "bits_b"}),
                                                                        "a", "b", rng)
        S["identical_events"]["gbm_residual_S6"] = contrast(both.rename(columns={"bits_S6_gbm": "bits_a", "bits_S6_H_gbm": "bits_b"}), "a", "b", rng)
        S["identical_events"]["mnl_residual_S6"] = contrast(both.rename(columns={"bits_S6_mnl": "bits_a", "bits_S6_H_mnl": "bits_b"}), "a", "b", rng)
    r6m = S["mnl"]["residual"]["S6"]
    r6g = S["gbm"]["residual"]["S6"] if gbm is not None else None
    reg = {"mnl_r6": regime(r6m), "gbm_r6": regime(r6g) if r6g else None,
           "mnl_r7": regime(S["mnl"]["residual"]["S7"]), "gbm_r7": regime(S["gbm"]["residual"]["S7"]) if gbm is not None else None}
    if r6g is None:
        reg["regime"] = reg["mnl_r6"] + " (logit only)"
    elif reg["mnl_r6"] == reg["gbm_r6"]:
        reg["regime"] = reg["mnl_r6"]
    else:
        reg["regime"] = f"B (families disagree: logit {reg['mnl_r6']}, GBM {reg['gbm_r6']})"
    S["regime"] = reg
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    write_tables(S, mnl, gbm)


def write_tables(S, mnl, gbm):
    L = []
    m = S["mnl"]
    L.append("# E-STATE summary (NBA nba_stats segments, development seasons 2000-2020, prequential by month)\n" if "wnba" not in OUT else
             "# E-STATE portability summary (WNBA wnba_stats segments 2016-2020, test months 2018-05..2020-10, logit only; spec section 7)\n")
    L.append(f"## 1. Frozen multinomial logit: events {m['n_events']:,}, games {m['n_games']:,}, months {m['n_months']}\n")
    L.append("| state | blocks | columns (median) | held-out bits/event | add-one gain (mb) [CI] | residual history r_k (mb) [CI] | converged |")
    L.append("|---|---|---|---|---|---|---|")
    for k in range(8):
        name = f"S{k}"
        g = fmt(m["gain"][name]) if k else "-"
        L.append(f"| {name} | {'S' if k == 0 else '+ ' + LABEL[BLOCK_ORDER[k-1]]} | {m['columns_median'].get(name, '')} | {m['mean_bits'][name]:.4f} | {g} | "
                 f"{fmt(m['residual'][name])} | {m['converged_share'].get(name, float('nan')):.2f} / {m['converged_share'].get(name + '_H', float('nan')):.2f} |")
    L.append("")
    L.append("Bridge to E-HIST (beyond the latent): " + "; ".join(f"{k}: {fmt(v)}" for k, v in m.get("residual_beyond_latent", {}).items()) +
             ". Latent gain: " + "; ".join(f"{k}: {fmt(v)}" for k, v in m.get("latent_gain", {}).items()) + "\n")
    tot = m["total_S0_to_S7_H"]
    L.append(f"## 2. Information decomposition (S0 -> S7 + full history = {fmt(tot)} mb)\n")
    L.append("| component | mb/event [CI] | share of total |")
    L.append("|---|---|---|")
    for k in range(1, 8):
        c = m["gain"][f"S{k}"]
        L.append(f"| {LABEL[BLOCK_ORDER[k-1]]} | {fmt(c)} | {100 * c['mb'] / tot['mb']:.1f}% |")
    c = m["residual"]["S7"]
    L.append(f"| remaining within-game history (GL + L1 + L3 + L10 + EW) after S7 | {fmt(c)} | {100 * c['mb'] / tot['mb']:.1f}% |")
    L.append("")
    L.append("## 3. Previous-class proxy test and short memory\n")
    L.append(f"* PC on S0: {fmt(m['pc_on_S0'])}; PC on S6 (= g7): {fmt(m['pc_on_S6'])}; PC beyond full history on S0: {fmt(m['pc_beyond_history_on_S0'])}")
    L.append(f"* S0 + PC vs S4 (= S0 + fouls + gap + clock x score): {fmt(m['S0_PC_vs_S4'])} (positive = S4 better)")
    L.append(f"* full history on S0 + PC: {fmt(m['history_on_S0_PC'])}; on S6: {fmt(m['residual']['S6'])}; on S7: {fmt(m['residual']['S7'])}\n")
    if "control_state_permuted_vs_S0" in m:
        L.append("## 4. Controls (expected <= 0)\n")
        L.append(f"* S6 with its state blocks permuted within cells vs S0: {fmt(m['control_state_permuted_vs_S0'])}")
        L.append(f"* full history permuted within cells on S6: {fmt(m['control_history_permuted_on_S6'])}\n")
    L.append("## 5. Per season (mb/event)\n")
    L.append("| season | events | games | " + " | ".join(f"g{k} {BLOCK_ORDER[k-1]}" for k in range(1, 8)) + " | r0 | r6 [CI] | r7 [CI] |")
    L.append("|---|---|---|" + "---|" * 7 + "---|---|---|")
    for r in m["per_season"]:
        L.append(f"| {r['season']} | {r['events']:,} | {r['games']} | " + " | ".join(f"{r[f'g{k}']:+.2f}" for k in range(1, 8)) +
                 f" | {r['r0']['mb']:+.2f} | {r['r6']['mb']:+.2f} [{r['r6']['lo']:+.2f}, {r['r6']['hi']:+.2f}] | {r['r7']['mb']:+.2f} [{r['r7']['lo']:+.2f}, {r['r7']['hi']:+.2f}] |")
    L.append("")
    L.append("## 6. Where the primary residual r6 sits (logit, pooled means; diagnostic)\n")
    for key, tab in S["diagnostics_mnl"].items():
        L.append(f"**by {key}**: " + "; ".join(f"{k}: share {v['share']:.3f}, r0 {v['r0_mb']:+.2f}, r6 {v['r6_mb']:+.2f} (contrib {v['r6_contribution_mb']:+.2f})" for k, v in tab.items()))
    L.append("")
    if "gbm" in S:
        g = S["gbm"]
        L.append(f"## 7. LightGBM replication on the same months: events {g['n_events']:,}, months {g['n_months']}\n")
        L.append("| state | bits/event | add-one gain (mb) [CI] | residual r_k (mb) [CI] | leave-one-out from S6 (mb) [CI] |")
        L.append("|---|---|---|---|---|")
        for k in range(8):
            name = f"S{k}"
            loo = fmt(g["leave_one_out_from_S6"][BLOCK_ORDER[k-1]]) if (1 <= k <= 6 and "leave_one_out_from_S6" in g) else "-"
            L.append(f"| {name} | {g['mean_bits'][name]:.4f} | {fmt(g['gain'][name]) if k else '-'} | {fmt(g['residual'][name])} | {loo} |")
        L.append("")
        L.append(f"PC on S0 {fmt(g['pc_on_S0'])}; S0 + PC vs S4 {fmt(g['S0_PC_vs_S4'])}; total S0 -> S7_H {fmt(g['total_S0_to_S7_H'])}\n")
        ie = S["identical_events"]
        L.append(f"### 7.1 Logit vs GBM on identical events ({ie['n_events']:,} events, {ie['n_months']} months; positive = GBM better)\n")
        L.append("| set | logit - GBM (mb) [CI] |")
        L.append("|---|---|")
        for name in [f"S{k}" for k in range(8)] + [f"S{k}_H" for k in range(8)]:
            L.append(f"| {name} | {fmt(ie[f'logit_minus_gbm_{name}'])} |")
        L.append(f"\nResidual r6 on identical events: logit {fmt(ie['mnl_residual_S6'])}, GBM {fmt(ie['gbm_residual_S6'])}\n")
    h = m["H2_frozen"]
    L.append("## 8. Frozen H2 (registry decision_override), S* = S0 against R = S7 (logit)\n")
    L.append(f"loss {fmt(h['loss_S0_vs_S7'])} mb; I(S7) over class frequencies {fmt(h['I_S7_over_null'])} mb; threshold {h['threshold_mb']:.2f} mb -> **{h['decision']}**")
    if "gbm" in S:
        hg = S["gbm"]["H2_frozen"]
        L.append(f"GBM: loss {fmt(hg['loss_S0_vs_S7'])}; threshold {hg['threshold_mb']:.2f} -> {hg['decision']}")
    L.append("")
    L.append("## 9. Regime (docs/p2/02_estate_spec.md section 6)\n")
    L.append(json.dumps(S["regime"]))
    open(os.path.join(OUT, "summary_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
