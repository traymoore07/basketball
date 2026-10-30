"""Render results/p1_ladder/summary.json as markdown tables (results/p1_ladder/summary_tables.md).
Pure presentation: no statistic is computed here that summarize.py did not already compute."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "p1_ladder")
TARGETS = ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]
RUNGS = ["L0_climatology", "L1_player_mean", "L2_selected", "L3_minutes_rate", "L4_hier_shrink", "L4_hier_shrink_plugin"]
GRID = ["L2_recent_hl5", "L2_recent_hl10", "L2_recent_hl20", "L2_recent_hl40"]
SHORT = {"L0_climatology": "L0", "L1_player_mean": "L1", "L2_selected": "L2", "L3_minutes_rate": "L3",
         "L4_hier_shrink": "L4", "L4_hier_shrink_plugin": "L4-plugin", "L2_recent_hl5": "L2 h=5", "L2_recent_hl10": "L2 h=10",
         "L2_recent_hl20": "L2 h=20", "L2_recent_hl40": "L2 h=40"}
METRICS = (("crps", "CRPS (lower is better)"), ("log", "Log score, bits (lower is better)"), ("rps", "RPS on frozen bins"),
           ("ae_median", "MAE of the median"), ("se_mean", "RMSE of the mean"), ("tw_crps", "Upper-tail twCRPS"))


def pct(x):
    return "n/a" if x is None else f"{100 * x:+.2f}%"


def num(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def main():
    r = json.load(open(os.path.join(OUT, "summary.json")))
    L = ["# P1 ladder: summary tables", "",
         f"Seasons scored (development): {r['seasons_reported']}; selection-only (L2 half-life): {r['selection_only']}.", ""]
    # A: losses
    for mode in ("U", "A"):
        ml = r["mean_loss_regular_dev"][mode]
        L += ["", f"## A.{mode} Mean per-unit losses, mode {mode}, regular season, development window", "",
              "Units per target: " + ", ".join(f"{t}={ml['_n_units'][t]:,}" for t in TARGETS if t in ml["_n_units"]), ""]
        for met, nm in METRICS:
            L += ["", f"**{nm}**", "", "| rung | " + " | ".join(TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
            for g in RUNGS + GRID:
                row = []
                for t in TARGETS:
                    v = ml[t].get(g, {}).get(met)
                    if met == "se_mean" and v is not None:
                        v = v ** 0.5
                    row.append(num(v))
                L.append(f"| {SHORT[g]} | " + " | ".join(row) + " |")
    # B: credit chain
    for mode in ("U", "A"):
        cc = r["credit_chain"][mode]
        L += ["", f"## B.{mode} Credit chain, mode {mode} (core = equal-weight mean CRPS skill over 7 targets, regular season)", "",
              "Core skill vs L0: " + ", ".join(f"{k} {pct(v)}" for k, v in cc["core_skill_vs_L0"].items()), "",
              "| candidate | reference | kind | core effect | 95% CI | p (Holm) | clusters | decision | no-harm violations |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in cc["comparisons"]:
            k = c["core"]
            viol = ", ".join(k.get("harmful_secondary", [])) or "none"
            L.append(f"| {c['candidate']} | {c['reference']} | {c['kind']} | {pct(k['effect'])} | [{pct(k['ci_low'])}, {pct(k['ci_high'])}] | "
                     f"{num(k.get('p_adj', k['p_value']))} | {k['n_clusters']} | {k.get('decision', '')} | {viol} |")
        L += ["", "Per-target CRPS skill of candidate vs reference (95% cluster-bootstrap CI):", "",
              "| comparison | " + " | ".join(TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
        for c in cc["comparisons"]:
            cells = [f"{pct(c['per_target'][t]['effect'])} [{pct(c['per_target'][t]['ci_low'])}, {pct(c['per_target'][t]['ci_high'])}]" for t in TARGETS]
            L.append(f"| {c['candidate']} vs {c['reference']} | " + " | ".join(cells) + " |")
        L += ["", "L2 half-life grid, core skill vs L1: " + ", ".join(f"{SHORT[k]} {pct(v)}" for k, v in cc["L2_grid_core_skill_vs_L1"].items())]
        p = cc["L4_plugin_vs_L4_core"]
        L += ["", f"L4-plugin vs L4 core: {pct(p['effect'])} [{pct(p['ci_low'])}, {pct(p['ci_high'])}]", ""]
    # C: L2 selection
    sel = r["L2_selection"]
    L += ["", "## C L2 half-life selected per season (lowest core CRPS on strictly prior scored seasons)", "",
          "| season | selected h | " + " | ".join(f"ratio vs L1, h={h}" for h in (5, 10, 20, 40)) + " |", "|---|---|---|---|---|---|"]
    for s, h in sel["half_life_by_season"].items():
        cr = sel["criterion_mean_crps_ratio_vs_L1_prior_seasons"].get(s, {})
        L.append(f"| {s} | {h} | " + " | ".join(num(cr.get(str(x))) for x in (5, 10, 20, 40)) + " |")
    # D: calibration
    for mode in ("U", "A"):
        cal = r["calibration_regular_dev"][mode]
        L += ["", f"## D.{mode} Calibration (randomised PIT, seed 0), mode {mode}, regular season", "",
              "| target | rung | n | PIT mean | dispersion (12 Var) | KS | cov50 | cov80 | cov95 | bias (mean fc minus y) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in TARGETS:
            for g in RUNGS:
                c = cal[t].get(g)
                if c is None:
                    continue
                L.append(f"| {t} | {SHORT[g]} | {c['n']:,} | {c['pit_mean']:.3f} | {c['dispersion_ratio']:.3f} | {c['ks']:.4f} | "
                         f"{c['cov50']:.3f} | {c['cov80']:.3f} | {c['cov95']:.3f} | {c['bias_mean_fc_minus_y']:+.3f} |")
    # E: low-sample calibration
    cl = r["calibration_by_low_sample"]
    L += ["", "## E Calibration inside the low_sample stratum (coverage 80 / dispersion / KS), both modes", "",
          "| mode | target | rung | n low | cov80 low | disp low | KS low | n rest | cov80 rest | disp rest | KS rest |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in ("U", "A"):
        for t in TARGETS:
            for g in ("L2_selected", "L3_minutes_rate", "L4_hier_shrink", "L4_hier_shrink_plugin"):
                d = cl[mode][t].get(g, {})
                lo, re = d.get("low_sample"), d.get("rest")
                if not lo or not re:
                    continue
                L.append(f"| {mode} | {t} | {SHORT[g]} | {lo['n']:,} | {lo['cov80']:.3f} | {lo['dispersion_ratio']:.3f} | {lo['ks']:.4f} | "
                         f"{re['n']:,} | {re['cov80']:.3f} | {re['dispersion_ratio']:.3f} | {re['ks']:.4f} |")
    # F: strata
    st = r["strata"]
    L += ["", "## F.1 Per season (mode U, regular): core skill vs L0 and vs the previous rung; MIN CRPS per rung", "",
          "| season | clusters | h | L1 vs L0 | L2 vs L1 | L3 vs L2 | L4 vs L3 | L4 vs L0 | MIN CRPS L0/L1/L2/L3/L4 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, d in st["per_season_regular_U"].items():
        mins = "/".join(f"{d[f'{g}_MIN_U_crps']:.3f}" for g in ("L0", "L1", "L2", "L3", "L4"))
        L.append(f"| {s} | {d['n_clusters']} | {d['L2_half_life']} | {pct(d['L1_core_vs_L0'])} | {pct(d['L2_core_vs_L1'])} | "
                 f"{pct(d['L3_core_vs_L2'])} | {pct(d['L4_core_vs_L3'])} | {pct(d['L4_core_vs_L0'])} | {mins} |")
    L += ["", "## F.2 Postseason (mode U): core skill vs previous rung", ""]
    for mode, d in st["postseason_U"].items():
        L.append(f"- mode {mode}, clusters {d['n_clusters']}: " + ", ".join(f"{k} {pct(v)}" for k, v in d.items() if k != "n_clusters"))
    L += ["", "## F.3 low_sample stratum: L4 vs L3", "",
          "| mode/stratum | clusters | core effect | 95% CI | " + " | ".join(TARGETS) + " |", "|---|---|---|---|" + "---|" * len(TARGETS)]
    for k, d in st["low_sample"].items():
        c = d["L4_vs_L3_core"]
        L.append(f"| {k} | {d['n_clusters']} | {pct(c['effect'])} | [{pct(c['ci_low'])}, {pct(c['ci_high'])}] | " +
                 " | ".join(pct(d["L4_vs_L3_per_target"][t]) for t in TARGETS) + " |")
    L += ["", "## F.4 role_change_alarm stratum: L2 vs L1 (X-L2-L1 note only; no H6 statistic)", "",
          "| mode/stratum | clusters | core effect | 95% CI | " + " | ".join(TARGETS) + " |", "|---|---|---|---|" + "---|" * len(TARGETS)]
    for k, d in st["role_change_alarm"].items():
        c = d["L2_vs_L1_core"]
        L.append(f"| {k} | {d['n_clusters']} | {pct(c['effect'])} | [{pct(c['ci_low'])}, {pct(c['ci_high'])}] | " +
                 " | ".join(pct(d.get("L2_vs_L1_per_target", {}).get(t)) for t in TARGETS) + " |")
    # G: gates
    L += ["", "## G P1 gates", ""]
    for k, d in r["gates"].items():
        L.append(f"- {k}: {'PASS' if d['passed'] else 'FAIL'}; " + json.dumps({x: y for x, y in d.items() if x != 'passed'}))
    open(os.path.join(OUT, "summary_tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote summary_tables.md", len(L), "lines")


if __name__ == "__main__":
    main()
