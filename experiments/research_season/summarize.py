"""R1 summary (research_season/01_carryover_spec.md sections 3-5): per-season selection on prior seasons, paired
contrasts of the selected grid point against A (the frozen L2) on identical units, by phase of the season,
by stratum and by season; family comparison; the frozen decision; the artefact rule. Nothing here fits a model.

Pass 1 reads every season's score parts and keeps per (season, target, model, phase bucket) sums; pass 2 reloads
only the rows of A and of the models selected for that season. Game-day block bootstrap (B = 2000, block 7 days,
seed 12345), Holm over the seven targets, MPE = P1 CRPS skill 0.01.
Writes results/research_season/summary.json and summary_tables.md."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.metrics.compare import apply_decisions, compare_losses  # noqa: E402
from hoopslab.models.carryover import GRID, GRID_IDS, TARGETS  # noqa: E402

OUT = os.path.join(ROOT, "results", "research_season")
DEV = list(range(2000, 2021))
SEL_ONLY = [1998, 1999]
MPE, B, BLOCK, SEED = 0.01, 2000, 7, 12345
CORE = ["PTS", "REB", "AST", "MIN"]
PHASES = ["game 1", "games 2-5", "games 6-10", "games 11-20", "rest"]
FAMILY = {g["id"]: g["family"] for g in GRID}


def phase_of(idx: np.ndarray) -> np.ndarray:
    return np.select([idx <= 1, idx <= 5, idx <= 10, idx <= 20], PHASES[:4], default="rest")


def load_season(S: int, models=None) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(OUT, f"scores_{S}_part*.parquet")))
    if not fs:
        return None
    parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=["model", "target", "game_id", "player_id", "crps", "log", "pit"])
        if models is not None:
            d = d[d["model"].isin(models)]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    ctx = pd.read_parquet(os.path.join(OUT, f"context_{S}.parquet"))
    ctx = ctx.drop_duplicates(["game_id", "player_id"])
    d = d.merge(ctx[["game_id", "player_id", "forecast_time", "team_game_index", "n_cur", "has_prev_season", "team_changed",
                     "role_changed", "high_turnover", "starter_prev"]], on=["game_id", "player_id"], how="left")
    d["phase"] = phase_of(d["team_game_index"].fillna(99).to_numpy())
    d["early"] = d["team_game_index"].fillna(99).to_numpy() <= 20
    d["cluster"] = pd.to_datetime(d["forecast_time"], utc=True).dt.floor("D")
    sf = os.path.join(OUT, f"strata_{S}.parquet")
    if os.path.exists(sf):
        st = pd.read_parquet(sf).drop_duplicates(["game_id", "player_id"])
        d = d.merge(st[["game_id", "player_id", "own_move_21d", "roster_change_21d", "low_sample", "role_change_alarm"]],
                    on=["game_id", "player_id"], how="left")
    d["season"] = S
    return d


def pass1(seasons):
    """per (season, target, model, early) mean CRPS; used for selection only."""
    rows = []
    for S in seasons:
        d = load_season(S)
        if d is None:
            continue
        g = d.groupby(["target", "model", "early"], observed=True)["crps"].agg(["sum", "count"]).reset_index()
        g["season"] = S
        rows.append(g)
        print("pass1", S, len(d), flush=True)
    return pd.concat(rows, ignore_index=True)


def select(tab, S, target):
    """mean CRPS over ALL units of scored seasons < S (selection-only seasons included): best point incl. A, best non-A."""
    prior = tab[(tab["season"] < S) & (tab["target"] == target)]
    if not len(prior):
        return None, None
    m = prior.groupby("model", observed=True)[["sum", "count"]].sum()
    m = (m["sum"] / m["count"]).sort_values()
    best = m.index[0]
    best_non_a = m.drop("A").index[0] if "A" in m.index else m.index[0]
    return str(best), str(best_non_a)


def paired(d, cand, target, mask=None, rng=None):
    a = d[(d["model"] == "A") & (d["target"] == target)]
    c = d[(d["model"] == cand) & (d["target"] == target)]
    k = ["game_id", "player_id"]
    m = a[k + ["cluster", "crps"]].merge(c[k + ["crps"]], on=k, suffixes=("_ref", "_cand"))
    if mask is not None:
        mm = d[(d["model"] == "A") & (d["target"] == target)][k + [mask]].drop_duplicates(k)
        m = m.merge(mm, on=k)
        m = m[m[mask].astype(bool)]
    if len(m) < 50:
        return None
    r = compare_losses(m["crps_ref"].to_numpy(), m["crps_cand"].to_numpy(), m["cluster"].to_numpy(), metric="crps",
                       reference="A", candidate=cand, B=B, block=BLOCK, rng=rng)
    return r


def main():
    rng = np.random.default_rng(SEED)
    done = {int(os.path.basename(f)[5:9]) for f in glob.glob(os.path.join(OUT, "done_????.json"))}
    seasons = sorted(s for s in SEL_ONLY + DEV if s in done)
    tab = pass1(seasons)
    S_out = {"seasons": seasons, "selection": {}, "mpe": MPE}
    # selection per reported season and target
    sel = {}
    for S in [s for s in seasons if s in DEV]:
        sel[S] = {t: select(tab, S, t) for t in TARGETS}
    S_out["selection"] = {str(S): {t: {"selected_incl_A": v[0], "selected_non_A": v[1]} for t, v in d.items()} for S, d in sel.items()}
    # pass 2: rows of A, the selected non-A point and the best point per family (selected on prior seasons too)
    fam_sel = {}
    for S in sel:
        fam_sel[S] = {}
        for t in TARGETS:
            prior = tab[(tab["season"] < S) & (tab["target"] == t)]
            m = prior.groupby("model", observed=True)[["sum", "count"]].sum()
            m = (m["sum"] / m["count"])
            fam_sel[S][t] = {f: str(m[[i for i in m.index if FAMILY[i] == f]].idxmin()) for f in ("B", "C", "D", "E") if any(FAMILY[i] == f for i in m.index)}
    frames = []
    for S in sel:
        need = {"A"} | {sel[S][t][1] for t in TARGETS if sel[S][t][1]} | {v for t in TARGETS for v in fam_sel[S][t].values()}
        d = load_season(S, models=need)
        # relabel: the per-target selected model becomes "SEL"; family bests become "F_B" .. "F_E" per target
        parts = [d[d["model"] == "A"]]
        for t in TARGETS:
            dt = d[d["target"] == t]
            s = dt[dt["model"] == sel[S][t][1]].copy()
            s["model"] = "SEL"
            parts.append(s)
            for f, mid in fam_sel[S][t].items():
                x = dt[dt["model"] == mid].copy()
                x["model"] = f"F_{f}"
                parts.append(x)
        frames.append(pd.concat(parts, ignore_index=True))
        print("pass2", S, flush=True)
    D = pd.concat(frames, ignore_index=True)
    D["model"] = D["model"].astype(str)
    res = {}
    # primary: SEL vs A, games 1-20, pooled; Holm over the 7 targets
    comps = [paired(D, "SEL", t, mask="early", rng=rng) for t in TARGETS]
    comps = [c for c in comps if c is not None]
    apply_decisions(comps, [MPE] * len(comps))
    res["primary_games_1_20"] = {t: c.to_dict() for t, c in zip(TARGETS, comps)}
    # rest of season (must not lose beyond the MPE)
    D["rest"] = ~D["early"]
    rest = [paired(D, "SEL", t, mask="rest", rng=rng) for t in TARGETS]
    res["rest_of_season"] = {t: (c.to_dict() if c else None) for t, c in zip(TARGETS, rest)}
    # by phase
    res["by_phase"] = {}
    for ph in PHASES:
        D["_ph"] = D["phase"] == ph
        res["by_phase"][ph] = {t: (lambda c: c.to_dict() if c else None)(paired(D, "SEL", t, mask="_ph", rng=rng)) for t in TARGETS}
    # phase where the gain first falls below the MPE
    res["phase_gain_below_mpe"] = {}
    for t in TARGETS:
        first = None
        for ph in PHASES:
            c = res["by_phase"][ph][t]
            if c is not None and c["effect"] < MPE:
                first = ph
                break
        res["phase_gain_below_mpe"][t] = first
    # strata (games 1-20)
    strata = {"same_team": ("team_changed", False), "new_team": ("team_changed", True), "stable_role": ("role_changed", False),
              "changed_role": ("role_changed", True), "starter_prev": ("starter_prev", True), "bench_prev": ("starter_prev", False),
              "high_turnover": ("high_turnover", True), "low_turnover": ("high_turnover", False),
              "own_move_21d": ("own_move_21d", True), "roster_change_21d": ("roster_change_21d", True), "low_sample": ("low_sample", True),
              "role_change_alarm": ("role_change_alarm", True), "no_current_history": ("n_cur", 0)}
    res["strata_games_1_20"] = {}
    for name, (col, val) in strata.items():
        if col not in D:
            continue
        if isinstance(val, bool):
            m = D[col].fillna(False).astype(bool) == val
        else:
            m = D[col] == val
        D["_m"] = D["early"] & m
        res["strata_games_1_20"][name] = {t: (lambda c: c.to_dict() if c else None)(paired(D, "SEL", t, mask="_m", rng=rng)) for t in TARGETS}
    res["no_current_history_share_games_1_20"] = float((D[(D["model"] == "A") & D["early"]]["n_cur"] == 0).mean())
    # families (games 1-20)
    res["families_games_1_20"] = {}
    for f in ("B", "C", "D", "E"):
        res["families_games_1_20"][f] = {t: (lambda c: c.to_dict() if c else None)(paired(D, f"F_{f}", t, mask="early", rng=rng)) for t in TARGETS}
    # per season (games 1-20, SEL vs A)
    res["per_season_games_1_20"] = {}
    for S in sorted(D["season"].unique()):
        D["_s"] = D["early"] & (D["season"] == S)
        res["per_season_games_1_20"][int(S)] = {t: (lambda c: {"effect": c.effect, "ci_low": c.ci_low, "ci_high": c.ci_high, "n": c.n_units} if c else None)(
            paired(D, "SEL", t, mask="_s", rng=rng)) for t in TARGETS}
    # decision (spec section 4)
    prim = res["primary_games_1_20"]
    core_ok = all(prim.get(t, {}).get("decision") == "SUPPORTED" for t in CORE)
    rest_ok = all((res["rest_of_season"][t] is None) or (res["rest_of_season"][t]["effect"] > -MPE and res["rest_of_season"][t]["decision"] != "HARMFUL") for t in CORE)
    res["decision"] = {"core_targets_supported_games_1_20": core_ok, "rest_of_season_not_lost_beyond_mpe": rest_ok,
                       "artefact": "SEASON-AWARE-L2-v0.2: YES" if (core_ok and rest_ok) else "NO ARTEFACT (baseline stands)"}
    # the artefact's weighting rule: the grid point selected on ALL development seasons, per target
    allsel = {}
    for t in TARGETS:
        prior = tab[(tab["season"] <= max(DEV)) & (tab["target"] == t)]
        m = prior.groupby("model", observed=True)[["sum", "count"]].sum()
        m = (m["sum"] / m["count"]).sort_values()
        allsel[t] = {"best_incl_A": str(m.index[0]), "best_non_A": str(m.drop("A").index[0]) if "A" in m.index else str(m.index[0]),
                     "mean_crps_A": float(m.get("A", np.nan)), "mean_crps_best": float(m.iloc[0])}
    res["final_selection_all_dev_seasons"] = allsel
    res["selection_by_season"] = S_out["selection"]
    res["n_units_games_1_20"] = int(((D["model"] == "A") & D["early"]).sum() // len(TARGETS))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    write_tables(res)


def fmt(c):
    if c is None:
        return "n/a"
    return f"{100*c['effect']:+.2f}% [{100*c['ci_low']:+.2f}, {100*c['ci_high']:+.2f}]" + (f" {c['decision']}" if c.get("decision") else "")


def write_tables(res):
    L = ["# R1 SEASON-CARRYOVER summary (NBA nba_stats, development seasons 2000-2020, mode A, CRPS relative skill of the selected grid point vs A = frozen L2)\n"]
    L.append(f"Units in games 1-20 (per target): {res['n_units_games_1_20']:,}; MPE {res['mpe']} (P1 CRPS skill)\n")
    L.append("## 1. Primary: selected (on prior seasons) non-A point vs A, team games 1-20, pooled, Holm over targets\n")
    L.append("| target | skill vs A [CI] decision | rest of season |")
    L.append("|---|---|---|")
    for t in TARGETS:
        L.append(f"| {t} | {fmt(res['primary_games_1_20'].get(t))} | {fmt(res['rest_of_season'].get(t))} |")
    L.append("\n## 2. By phase of the season\n")
    L.append("| target | " + " | ".join(PHASES) + " | first phase with gain < MPE |")
    L.append("|---|" + "---|" * (len(PHASES) + 1))
    for t in TARGETS:
        L.append(f"| {t} | " + " | ".join(fmt(res["by_phase"][ph][t]).split(" SUP")[0].split(" FAL")[0].split(" PRO")[0].split(" INC")[0].split(" HAR")[0] for ph in PHASES)
                 + f" | {res['phase_gain_below_mpe'][t]} |")
    L.append("\n## 3. Families (best point of each family on prior seasons) vs A, games 1-20\n")
    L.append("| target | B hard reset | C soft discount | D decaying | E context |")
    L.append("|---|---|---|---|---|")
    for t in TARGETS:
        L.append(f"| {t} | " + " | ".join(fmt(res["families_games_1_20"][f][t]) for f in ("B", "C", "D", "E")) + " |")
    L.append("\n## 4. Strata, games 1-20 (selected vs A)\n")
    L.append("| stratum | " + " | ".join(TARGETS) + " |")
    L.append("|---|" + "---|" * len(TARGETS))
    for name, d in res["strata_games_1_20"].items():
        L.append(f"| {name} | " + " | ".join(fmt(d[t]) for t in TARGETS) + " |")
    L.append(f"\nShare of games-1-20 units with no current-season appearance: {100*res['no_current_history_share_games_1_20']:.1f}%\n")
    L.append("## 5. Per season (games 1-20, selected vs A, skill %)\n")
    L.append("| season | " + " | ".join(TARGETS) + " |")
    L.append("|---|" + "---|" * len(TARGETS))
    for S, d in res["per_season_games_1_20"].items():
        L.append(f"| {S} | " + " | ".join((f"{100*d[t]['effect']:+.2f}" if d[t] else "n/a") for t in TARGETS) + " |")
    L.append("\n## 6. Selection on all development seasons (the artefact's weighting rule if adopted)\n")
    for t, v in res["final_selection_all_dev_seasons"].items():
        L.append(f"* {t}: best incl. A = {v['best_incl_A']}, best non-A = {v['best_non_A']} (mean CRPS A {v['mean_crps_A']:.4f}, best {v['mean_crps_best']:.4f})")
    L.append("\n## 7. Decision (spec section 4)\n")
    L.append(json.dumps(res["decision"], indent=1))
    open(os.path.join(OUT, "summary_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
