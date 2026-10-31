"""score_locked_forecast(): attach reality to a locked forecast AFTER the game is final, with the frozen metrics.

Separate from forecasting by construction: this module never imports the model, never alters a
package, and only appends to the ledger. Outcomes are read from the final box score the executor
supplies; the research lane never sees them (docs/shadow/00_firewall.md).

Evaluation populations (prospective scoring policy, docs/shadow/01):
    STATUS_CONFIRMED  availability_state CONFIRMED_ACTIVE at the cutoff and the player appeared   (PRIMARY)
    STATUS_EXPECTED   EXPECTED_ACTIVE and appeared                                                (secondary)
    UNKNOWN           QUESTIONABLE_UNKNOWN and appeared (stored; excluded from the primary score)
    NOT_APPEARED      forecast issued but seconds == 0 or no box row: mode A has no score; counted as a scratch
    OUT               no distribution was issued; counted only
Mode A never scores a non-appearance, so no appearance probability is ever inferred from a status.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..metrics import scoring as sc
from ..targets import BINS, PLAYER_TARGETS, player_outcomes
from .package import verify_locked

TAIL = {"PTS": 25, "PRA": 40, "REB": 10, "AST": 8}
POPULATION = {"CONFIRMED_ACTIVE": "STATUS_CONFIRMED", "EXPECTED_ACTIVE": "STATUS_EXPECTED",
              "QUESTIONABLE_UNKNOWN": "UNKNOWN", "OUT": "OUT"}
LEDGER_COLUMNS = ["forecast_id", "game_id", "model_version", "forecast_time", "information_cutoff", "lock_hash", "result_time",
                  "score_revision", "player_id", "team_id", "availability_state", "population", "target", "y", "crps", "log",
                  "rps", "pit", "ae_median", "se_mean", "tw_crps", "mean_fc", "p10", "p90", "in_central_80"]


def score_locked_forecast(pkg: dict, box_score: pd.DataFrame, result_time, rng_seed: int = 0, score_revision: int = 1) -> dict:
    """Score one locked package against the final box score.

    box_score: PLAYER_GAME-shaped rows of the finished game (game_id, player_id, team_id, seconds, pts, fgm, fga, fg3m,
    fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf). result_time: when the box became final/available.
    Returns {"rows": [ledger rows], "summary": {...}}. Nothing is written here; see Ledger.append.
    """
    if not verify_locked(pkg):
        raise ValueError("package is not a valid locked forecast (hash mismatch)")
    gid = pkg["game"]["game_id"]
    box = box_score[box_score["game_id"] == gid].copy()
    if box.duplicated(["game_id", "player_id"]).any():
        raise ValueError("box score has duplicate (game, player) rows; resolve deterministically before scoring")
    outc = player_outcomes(box).set_index("player_id") if len(box) else pd.DataFrame()
    rng = np.random.default_rng(rng_seed)
    rows, counts = [], {"STATUS_CONFIRMED": 0, "STATUS_EXPECTED": 0, "UNKNOWN": 0, "NOT_APPEARED": 0, "OUT": 0,
                        "OUT_but_appeared": 0}
    for p in pkg["players"]:
        pid = p["player_id"]
        state = p["availability_state"]
        appeared = bool(len(outc) and pid in outc.index and outc.loc[pid, "appeared"])
        if p["distributions"] is None:
            counts["OUT"] += 1
            if appeared:
                counts["OUT_but_appeared"] += 1
            continue
        if not appeared:
            counts["NOT_APPEARED"] += 1
            continue
        pop = POPULATION[state]
        counts[pop] += 1
        for t, d in p["distributions"].items():
            pmf = np.asarray(d["pmf"], dtype=np.float64)[None, :]
            y = np.asarray([int(outc.loc[pid, t])])
            pit = float(sc.randomized_pit(pmf, y, rng)[0])
            row = {"forecast_id": pkg["forecast_id"], "game_id": gid, "model_version": pkg["model_version"],
                   "forecast_time": pkg["forecast_created_at"], "information_cutoff": pkg["information_cutoff"],
                   "lock_hash": pkg["lock_hash"], "result_time": str(pd.Timestamp(result_time)), "score_revision": score_revision,
                   "player_id": pid, "team_id": p["team_id"], "availability_state": state, "population": pop, "target": t,
                   "y": int(y[0]), "crps": float(sc.crps_pmf(pmf, y)[0]), "log": float(sc.log_score(pmf, y)[0]),
                   "rps": float(sc.rps(sc.bin_pmf(pmf, BINS[t]), sc.bin_outcome(y, BINS[t]))[0]) if t in BINS else None,
                   "pit": pit, "ae_median": int(sc.abs_error_of_median(pmf, y)[0]), "se_mean": float(sc.sq_error_of_mean(pmf, y)[0]),
                   "tw_crps": float(sc.tw_crps_upper(pmf, y, TAIL[t])[0]) if t in TAIL else None,
                   "mean_fc": float(d["mean"]), "p10": int(d["p10"]), "p90": int(d["p90"]),
                   "in_central_80": bool(0.1 < pit < 0.9)}
            rows.append(row)
    return {"rows": rows, "counts": counts, "game_id": gid, "forecast_id": pkg["forecast_id"], "result_time": str(pd.Timestamp(result_time))}


class Ledger:
    """Append-only JSONL ledger of per-(forecast, player, target) scores + a derived cumulative table."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "ledger"), exist_ok=True)
        self.path = os.path.join(root, "ledger", "scores.jsonl")
        self.events = os.path.join(root, "ledger", "events.jsonl")

    def append(self, scored: dict) -> int:
        """Append rows; refuse if this (forecast_id, score_revision) was already scored."""
        key = (scored["forecast_id"], scored["rows"][0]["score_revision"] if scored["rows"] else 1)
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if (r["forecast_id"], r["score_revision"]) == key:
                        raise ValueError(f"{key} already in the ledger; corrections use a new score_revision")
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            for r in scored["rows"]:
                fh.write(json.dumps({c: r.get(c) for c in LEDGER_COLUMNS}, sort_keys=True) + "\n")
        with open(self.events, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"event": "scored", "forecast_id": scored["forecast_id"], "game_id": scored["game_id"],
                                 "result_time": scored["result_time"], "counts": scored["counts"], "rows": len(scored["rows"])},
                                sort_keys=True) + "\n")
        return len(scored["rows"])

    def frame(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        return pd.read_json(self.path, lines=True)

    def cumulative(self) -> pd.DataFrame:
        """Cumulative metrics by model version, target, population and month (of the scheduled tip's forecast time).
        Only the latest score_revision of each forecast_id counts. Derived from the ledger, never edited by hand."""
        df = self.frame()
        if not len(df):
            return df
        latest = df.groupby("forecast_id")["score_revision"].transform("max")
        df = df[df["score_revision"] == latest].copy()
        df["month"] = pd.to_datetime(df["forecast_time"], utc=True).dt.strftime("%Y-%m")
        g = df.groupby(["model_version", "target", "population", "month"])
        out = g.agg(n=("crps", "size"), crps=("crps", "mean"), log=("log", "mean"), rps=("rps", "mean"),
                    mae_median=("ae_median", "mean"), rmse_mean=("se_mean", lambda s: float(np.sqrt(np.mean(s)))),
                    tw_crps=("tw_crps", "mean"), cov80=("in_central_80", "mean"),
                    pit_dispersion=("pit", lambda s: float(np.var(s) * 12)), pit_ks=("pit", lambda s: sc.pit_ks(s.to_numpy())),
                    bias=("mean_fc", "mean")).reset_index()
        ymean = g["y"].mean().reset_index(name="y_mean")
        out = out.merge(ymean, on=["model_version", "target", "population", "month"])
        out["bias"] = out["bias"] - out["y_mean"]
        path = os.path.join(self.root, "ledger", "cumulative.csv")
        out.to_csv(path, index=False)
        return out
