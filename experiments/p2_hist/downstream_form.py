"""Downstream check (EXPLORATORY, post-outcome; not a hypothesis decision): does between-game history beyond
L2-A's compact current state improve player-stat distributions?

Within-game history cannot enter a pre-game forecast; its only downstream carrier would be a simulator,
which this mission does not build. The pre-game analogue of "history beyond the compact state" is
between-game form: the player's recent residuals (outcome minus L2-A forecast mean over his previous
three appearances), which L2's exponentially weighted mean (half-life 5) has already partly absorbed.

Diagnostic model L2-A + FORM: the frozen L2 (h = 5, mode A) conditional pmf shifted by delta = beta_t *
mean(last 3 residuals), beta_t fitted by least squares on the PREVIOUS season's appearing units (prequential by
season, 2001-2020; 2000 fits beta for 2001). The shift is applied as a mixture of the two integer shifts
around delta. L2-A itself is never changed; the comparison is the frozen paired CRPS skill with the P1
game-day cluster bootstrap (B = 2000, block 7). Inputs: P1 run-2 artifacts only (requests, L2_recent_hl5
forecasts, per-unit scores). Writes results/p2_hist/downstream_form.json / .md."""
import glob
import json
import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.metrics import compare as cp  # noqa: E402
from hoopslab.metrics import scoring as sc  # noqa: E402

P1 = os.path.join(ROOT, "results", "p1_ladder").replace("\\", "/")
OUT = os.path.join(ROOT, "results", "p2_hist")
TARGETS = ["MIN", "PTS", "REB", "AST", "PRA"]
SEASONS = list(range(2000, 2021))
B, BLOCK, SEED = 2000, 7, 12345


def shift_pmf(pmf, delta):
    """Shift a discrete pmf by a real delta: mixture of shifts floor(delta) and floor(delta)+1 with weights (1-f, f)."""
    K = pmf.shape[1]
    f0 = np.floor(delta).astype(int)
    w = delta - f0
    out = np.zeros_like(pmf)
    for s_add, wt in ((0, 1 - w), (1, w)):
        s = f0 + s_add
        for sv in np.unique(s):
            rows = np.where(s == sv)[0]
            if not len(rows):
                continue
            shifted = np.zeros((len(rows), K))
            if sv >= 0:
                shifted[:, sv:] = pmf[rows, :K - sv] if sv < K else 0
                shifted[:, -1] += pmf[rows, K - sv:].sum(1) if 0 < sv < K else 0
            else:
                shifted[:, :K + sv] = pmf[rows, -sv:]
                shifted[:, 0] += pmf[rows, :-sv].sum(1)
            out[rows] += wt[rows, None] * shifted
    return out / out.sum(1, keepdims=True)


def main():
    con = duckdb.connect()
    units = {}
    for S in SEASONS:
        req = pd.read_parquet(os.path.join(P1, f"requests_{S}.parquet"))
        fz = np.load(os.path.join(P1, f"forecasts_{S}_L2_recent_hl5.npz"))
        sc_df = con.execute(f"""SELECT game_id, player_id, target, y, crps, cluster FROM '{P1}/scores_{S}_part*.parquet'
                                 WHERE mode='A' AND model='L2_recent_hl5' AND target IN ({','.join("'"+t+"'" for t in TARGETS)})""").df()
        req = req.reset_index().rename(columns={"index": "row"})
        u = sc_df.merge(req[["row", "game_id", "player_id", "tip", "phase"]], on=["game_id", "player_id"], how="inner")
        for t in TARGETS:
            m = u["target"] == t
            idx = u.loc[m, "row"].to_numpy()
            pm = fz[f"cond_{t}"][idx].astype(np.float64)
            u.loc[m, "mean_fc"] = pm @ np.arange(pm.shape[1])
        u["season"] = S
        units[S] = u
    all_u = pd.concat(units.values(), ignore_index=True)
    all_u = all_u.sort_values(["target", "player_id", "tip"], kind="stable")
    all_u["resid"] = all_u["y"] - all_u["mean_fc"]
    g = all_u.groupby(["target", "player_id"], sort=False)["resid"]
    all_u["form3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    all_u["form3"] = all_u["form3"].fillna(0.0)
    res = {"design": "L2-A + FORM: pmf shifted by beta_t * mean(last 3 residuals), beta fitted on the previous season; prequential 2001-2020; mode A; regular season scored",
           "per_target": {}, "per_season": {}}
    rng = np.random.default_rng(SEED)
    L = ["# Downstream form diagnostic (EXPLORATORY): L2-A + between-game form vs L2-A", "",
         "| target | units | seasons | beta range | CRPS skill vs L2-A | 95% CI | p | mean |form3| |", "|---|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        d = all_u[all_u["target"] == t]
        ref, cand, clus, betas = [], [], [], {}
        for S in SEASONS[1:]:
            prev = d[(d["season"] == S - 1) & (d["phase"] == "regular")]
            cur = d[(d["season"] == S) & (d["phase"] == "regular")]
            if not len(prev) or not len(cur):
                continue
            x, yv = prev["form3"].to_numpy(), prev["resid"].to_numpy()
            beta = float((x * yv).sum() / max((x * x).sum(), 1e-9))
            betas[S] = beta
            req = pd.read_parquet(os.path.join(P1, f"requests_{S}.parquet"))
            fz = np.load(os.path.join(P1, f"forecasts_{S}_L2_recent_hl5.npz"))
            idx = cur.index
            rows = cur["row"].to_numpy() if "row" in cur else None
            pm = fz[f"cond_{t}"][cur["row"].to_numpy()].astype(np.float64)
            delta = beta * cur["form3"].to_numpy()
            pm2 = shift_pmf(pm, delta)
            yv = cur["y"].to_numpy().astype(int)
            ref.append(sc.crps_pmf(pm, yv)); cand.append(sc.crps_pmf(pm2, yv))
            clus.append(pd.to_datetime(cur["cluster"]).dt.floor("D").astype("int64").to_numpy() // 86_400_000_000_000)
        ref, cand, clus = np.concatenate(ref), np.concatenate(cand), np.concatenate(clus)
        c = cp.compare_losses(ref, cand, clus, B=B, block=BLOCK, rng=rng)
        res["per_target"][t] = {"units": int(len(ref)), "crps_L2A": float(ref.mean()), "crps_L2A_form": float(cand.mean()), "skill": c.effect,
                                "ci_low": c.ci_low, "ci_high": c.ci_high, "p_value": c.p_value, "beta_by_season": betas,
                                "mean_abs_form3": float(np.abs(d.loc[d["season"] >= 2001, "form3"]).mean())}
        L.append(f"| {t} | {len(ref):,} | 2001-2020 | [{min(betas.values()):+.3f}, {max(betas.values()):+.3f}] | {100*c.effect:+.3f}% | [{100*c.ci_low:+.3f}%, {100*c.ci_high:+.3f}%] | {c.p_value:.4f} | {res['per_target'][t]['mean_abs_form3']:.3f} |")
        print(L[-1], flush=True)
    json.dump(res, open(os.path.join(OUT, "downstream_form.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "downstream_form.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
