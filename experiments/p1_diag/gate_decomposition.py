"""P1 diagnostic mission (2026-10-30): decomposition of the two failed sanity gates.

Reads only the P1 per-unit scores, strata, requests and forecasts already produced (O-dev, exposed in
EXPOSURE_LOG rows 8-10). No model is run, nothing is re-fit, no threshold is introduced. Groupings that
are not frozen strata are labelled EXPLORATORY (post-outcome) in the output and never replace a frozen
result.

Writes results/p1_diag/gate_decomposition.json and .md.
"""
import glob
import json
import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
P1 = os.path.join(ROOT, "results", "p1_ladder").replace("\\", "/")
OUT = os.path.join(ROOT, "results", "p1_diag")
os.makedirs(OUT, exist_ok=True)
TARGETS = ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]
DEV = list(range(2000, 2021))
PRE = [2000, 2001, 2002, 2003]
res = {}
con = duckdb.connect()
con.execute("SET threads TO 16")
SC = f"'{P1}/scores_*_part*.parquet'"
ST = f"'{P1}/strata_*.parquet'"
RQ = f"'{P1}/requests_*.parquet'"
CS = f"'{P1}/cluster_sums.parquet'"


def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(L)


# ---- 1. L1 vs L0 per target, pre / post DI-2, regular season (frozen strata: season) ---------
q1 = con.execute(f"""
    SELECT CASE WHEN season <= 2003 THEN '2000-2003' ELSE '2004-2020' END AS period, mode, target,
           1 - sum(CASE WHEN model='L1_player_mean' THEN crps END) / sum(CASE WHEN model='L0_climatology' THEN crps END) AS skill_L1_vs_L0,
           sum(CASE WHEN model='L0_climatology' THEN n END) AS units
    FROM {CS} WHERE regular AND season BETWEEN 2000 AND 2020 AND model IN ('L0_climatology','L1_player_mean')
    GROUP BY 1,2,3 ORDER BY 2,1,3""").df()
res["1_L1_vs_L0_by_period"] = q1.to_dict("records")

# ---- 2. L1 vs L0 in mode U split by whether the player appeared (outcome split: EXPLORATORY) --
q2 = con.execute(f"""
    WITH u AS (
      SELECT season, game_id, player_id, target, model, crps, forecast_time
      FROM {SC} WHERE mode='U' AND model IN ('L0_climatology','L1_player_mean') AND season BETWEEN 2000 AND 2020),
    a AS (SELECT DISTINCT season, game_id, player_id FROM {SC} WHERE mode='A' AND model='L0_climatology' AND target='MIN'),
    r AS (SELECT game_id, player_id, phase FROM {RQ}),
    j AS (
      SELECT u.season, u.target, u.model, u.crps, (a.game_id IS NOT NULL) AS appeared
      FROM u LEFT JOIN a USING (season, game_id, player_id) JOIN r USING (game_id, player_id) WHERE r.phase='regular')
    SELECT CASE WHEN season <= 2003 THEN '2000-2003' ELSE '2004-2020' END AS period, appeared, target,
           count(*) FILTER (WHERE model='L0_climatology') AS units,
           sum(crps) FILTER (WHERE model='L0_climatology') AS crps_L0,
           sum(crps) FILTER (WHERE model='L1_player_mean') AS crps_L1
    FROM j GROUP BY 1,2,3 ORDER BY 3,1,2""").df()
q2["skill_L1_vs_L0"] = 1 - q2["crps_L1"] / q2["crps_L0"]
tot = q2.groupby(["period", "target"])[["crps_L0", "crps_L1"]].transform("sum")
q2["share_of_L0_loss"] = q2["crps_L0"] / tot["crps_L0"]
q2["share_of_gain"] = (q2["crps_L0"] - q2["crps_L1"]) / (tot["crps_L0"] - tot["crps_L1"])
res["2_L1_vs_L0_U_by_appearance_EXPLORATORY"] = q2.to_dict("records")

# ---- 3. L3 vs L2 on MIN: per-unit join with frozen strata + exploratory groupings -------------
base = f"""
    WITH s AS (
      SELECT season, game_id, player_id, forecast_time, mode, model, crps, pit, y, mean, ae_median
      FROM {SC} WHERE target='MIN' AND model IN ('L2_recent_hl5','L3_minutes_rate') AND season BETWEEN 2000 AND 2020),
    w AS (
      SELECT season, game_id, player_id, forecast_time, mode,
             max(CASE WHEN model='L2_recent_hl5' THEN crps END) AS crps_L2, max(CASE WHEN model='L3_minutes_rate' THEN crps END) AS crps_L3,
             max(CASE WHEN model='L2_recent_hl5' THEN pit END) AS pit_L2, max(CASE WHEN model='L3_minutes_rate' THEN pit END) AS pit_L3,
             max(CASE WHEN model='L2_recent_hl5' THEN mean END) AS mean_L2, max(CASE WHEN model='L3_minutes_rate' THEN mean END) AS mean_L3,
             max(CASE WHEN model='L2_recent_hl5' THEN ae_median END) AS ae_L2, max(CASE WHEN model='L3_minutes_rate' THEN ae_median END) AS ae_L3,
             max(y) AS y
      FROM s GROUP BY 1,2,3,4,5),
    st AS (SELECT season, game_id, player_id, low_sample, role_change_alarm, own_move_21d, roster_change_21d FROM {ST}),
    rq AS (SELECT season, game_id, player_id, phase FROM {RQ}),
    ss AS (SELECT season, min(forecast_time) AS season_start FROM {RQ} GROUP BY 1),
    j AS (
      SELECT w.*, st.low_sample, st.role_change_alarm, st.own_move_21d, st.roster_change_21d, rq.phase,
             date_diff('day', ss.season_start, w.forecast_time) AS days_into_season,
             (w.mode = 'A') AS appeared_row
      FROM w JOIN st USING (season, game_id, player_id) JOIN rq USING (season, game_id, player_id) JOIN ss USING (season))
"""
def grp(expr, label, where="TRUE"):
    df = con.execute(base + f"""
        SELECT mode, {expr} AS grp, count(*) AS units,
               sum(crps_L2) AS crps_L2, sum(crps_L3) AS crps_L3,
               avg(mean_L2 - y) AS bias_L2, avg(mean_L3 - y) AS bias_L3,
               12*var_pop(pit_L2) AS disp_L2, 12*var_pop(pit_L3) AS disp_L3,
               avg(ae_L2) AS mae_L2, avg(ae_L3) AS mae_L3, avg(y) AS mean_y
        FROM j WHERE {where} GROUP BY 1,2 ORDER BY 1,2""").df()
    df["skill_L3_vs_L2"] = 1 - df["crps_L3"] / df["crps_L2"]
    t = df.groupby("mode")[["crps_L2", "crps_L3"]].transform("sum")
    df["share_of_units"] = df["units"] / df.groupby("mode")["units"].transform("sum")
    df["share_of_L3_minus_L2_loss"] = (df["crps_L3"] - df["crps_L2"]) / (t["crps_L3"] - t["crps_L2"])
    df.insert(0, "grouping", label)
    return df

groups = [
    ("phase", "phase (frozen)"),
    ("low_sample", "low_sample (frozen)"),
    ("role_change_alarm", "role_change_alarm (frozen)"),
    ("own_move_21d", "own_move_21d (frozen)"),
    ("roster_change_21d", "roster_change_21d (frozen)"),
    ("CASE WHEN mode='U' THEN (y > 0) ELSE TRUE END", "appeared outcome in U (EXPLORATORY, outcome split)"),
    ("CASE WHEN days_into_season < 14 THEN 'a:<14d' WHEN days_into_season < 60 THEN 'b:14-59d' ELSE 'c:60d+' END", "days into season (EXPLORATORY, as-of)"),
    ("CASE WHEN mean_L2 < 8 THEN 'a:<8' WHEN mean_L2 < 16 THEN 'b:8-16' WHEN mean_L2 < 26 THEN 'c:16-26' ELSE 'd:26+' END", "L2 forecast mean minutes bin (EXPLORATORY, as-of forecast)"),
    ("CASE WHEN y = 0 THEN 'a:0' WHEN y < 10 THEN 'b:1-9' WHEN y < 20 THEN 'c:10-19' WHEN y < 30 THEN 'd:20-29' ELSE 'e:30+' END", "realised minutes bin (EXPLORATORY, outcome split)"),
    ("CASE WHEN season <= 2003 THEN '2000-2003' ELSE '2004-2020' END", "period pre/post DI-2 (frozen strata: season)"),
]
g3 = pd.concat([grp(e, l) for e, l in groups], ignore_index=True)
res["3_L3_vs_L2_MIN_decomposition"] = g3.to_dict("records")
# per season, regular, both modes
g3s = con.execute(base + """
    SELECT mode, season, count(*) AS units, 1 - sum(crps_L3)/sum(crps_L2) AS skill_L3_vs_L2,
           avg(mean_L2 - y) AS bias_L2, avg(mean_L3 - y) AS bias_L3, 12*var_pop(pit_L2) AS disp_L2, 12*var_pop(pit_L3) AS disp_L3
    FROM j WHERE phase='regular' GROUP BY 1,2 ORDER BY 1,2""").df()
res["3b_L3_vs_L2_MIN_by_season"] = g3s.to_dict("records")

# ---- 4. p_appear identity across rungs (from the saved forecasts) ---------------------------
ident = {}
for S in DEV:
    pas = {}
    for m in ("L0_climatology", "L1_player_mean", "L2_recent_hl5", "L3_minutes_rate", "L4_hier_shrink"):
        f = os.path.join(P1, f"forecasts_{S}_{m}.npz")
        if os.path.exists(f):
            pas[m] = np.load(f)["p_appear"]
    if len(pas) == 5:
        ident[S] = {"max_abs_diff_L1_L2": float(np.max(np.abs(pas["L1_player_mean"] - pas["L2_recent_hl5"]))),
                    "max_abs_diff_L2_L3": float(np.max(np.abs(pas["L2_recent_hl5"] - pas["L3_minutes_rate"]))),
                    "max_abs_diff_L3_L4": float(np.max(np.abs(pas["L3_minutes_rate"] - pas["L4_hier_shrink"]))),
                    "mean_p_appear_L0": float(pas["L0_climatology"].mean()), "mean_p_appear_L1": float(pas["L1_player_mean"].mean()),
                    "n": int(len(pas["L1_player_mean"]))}
res["4_p_appear_identity_by_season"] = ident

# ---- 5. MIN bias in U and A per season per model (from pit summaries, all phases) -------------
ps = pd.read_parquet(os.path.join(P1, "pit_summaries.parquet"))
ps = ps[(ps["target"] == "MIN") & ps["season"].isin(DEV)]
b = ps.groupby(["mode", "model", "season"]).apply(lambda d: pd.Series({"n": d["n"].sum(), "bias": (d["mean_fc"] * d["n"]).sum() / d["n"].sum() - (d["mean_y"] * d["n"]).sum() / d["n"].sum()}), include_groups=False).reset_index()
res["5_MIN_bias_by_season"] = b.to_dict("records")

# ---- 6. U-bias decomposition: mean p_appear vs realised appearance rate per season -------------
q6 = con.execute(f"""
    WITH u AS (SELECT season, game_id, player_id, y, mean FROM {SC} WHERE mode='U' AND target='MIN' AND model='L2_recent_hl5' AND season BETWEEN 2000 AND 2020),
         a AS (SELECT season, game_id, player_id, y AS y_a, mean AS mean_a FROM {SC} WHERE mode='A' AND target='MIN' AND model='L2_recent_hl5')
    SELECT u.season, count(*) AS units, avg(CASE WHEN a.game_id IS NOT NULL THEN 1 ELSE 0 END) AS realised_appear,
           avg(u.mean) AS mean_fc_U, avg(u.y) AS mean_y_U, avg(a.mean_a) AS mean_fc_A_given_appear, avg(a.y_a) AS mean_y_A
    FROM u LEFT JOIN a USING (season, game_id, player_id) GROUP BY 1 ORDER BY 1""").df()
for S in DEV:
    if S in ident:
        q6.loc[q6["season"] == S, "mean_p_appear_L2"] = ident[S]["mean_p_appear_L1"]  # L1 == L2 p_appear (checked in 4)
q6["p_appear_minus_realised"] = q6["mean_p_appear_L2"] - q6["realised_appear"]
q6["bias_U"] = q6["mean_fc_U"] - q6["mean_y_U"]
q6["bias_A"] = q6["mean_fc_A_given_appear"] - q6["mean_y_A"]
res["6_U_bias_vs_p_appear_by_season_L2"] = q6.to_dict("records")

json.dump(res, open(os.path.join(OUT, "gate_decomposition.json"), "w"), indent=1, default=str)
L = ["# Gate decomposition (diagnostic only; EXPLORATORY groupings are post-outcome and never replace a frozen result)", ""]
L += ["## 1. L1 vs L0 per target, pre / post DI-2 (regular season, frozen season stratum)", "", md_table(q1), ""]
L += ["## 2. L1 vs L0 in mode U by appearance outcome (EXPLORATORY)", "", md_table(q2[["period", "appeared", "target", "units", "skill_L1_vs_L0", "share_of_L0_loss", "share_of_gain"]]), ""]
L += ["## 3. L3 vs L2 on MIN: decomposition", "", md_table(g3[["grouping", "mode", "grp", "units", "share_of_units", "skill_L3_vs_L2", "share_of_L3_minus_L2_loss", "bias_L2", "bias_L3", "disp_L2", "disp_L3", "mae_L2", "mae_L3", "mean_y"]]), ""]
L += ["## 3b. L3 vs L2 on MIN per season (regular)", "", md_table(g3s), ""]
L += ["## 4. p_appear identity across rungs", "", md_table(pd.DataFrame(ident).T.reset_index().rename(columns={"index": "season"})), ""]
L += ["## 5. MIN bias (mean forecast minus mean outcome) per season and model", "", md_table(b.pivot_table(index=["mode", "season"], columns="model", values="bias").reset_index()), ""]
L += ["## 6. U bias vs appearance probability (L2), per season", "", md_table(q6), ""]
open(os.path.join(OUT, "gate_decomposition.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
