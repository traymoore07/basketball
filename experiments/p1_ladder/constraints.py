"""Frozen decision constraints (docs/prereg/08_metric_spec.md section 5, "Additional binding constraints"):

  1. No-harm: a SUPPORTED claim is downgraded to PROMISING if any secondary metric for the same
     comparison is HARMFUL at the aggregate level.
  2. Calibration floor: a SUPPORTED claim is downgraded if the candidate's PIT-KS distance or
     |coverage80 - 0.8| is worse than the reference's by more than 0.02 in the same comparison.

summarize.py computes every ingredient (harmful_secondary lists; PIT-KS and coverage80 per model,
target and mode on the regular-season development window) but its first version labelled the
decision with compare.decide alone. This module applies the two constraints to the credit-chain
decisions, as a pure function of summary.json: no statistic is recomputed and no threshold is
introduced (0.02 is the frozen value). The raw label is kept as decision_raw.

Reading used for the calibration floor on a core (7-target) comparison: the floor is checked for
every core target of the comparison, in the comparison's conditioning mode, on the regular-season
development window; a violation on any target downgrades the core claim. The targets that triggered
it are listed under downgrade_reasons.

Run standalone (python constraints.py) to apply in place to results/p1_ladder/summary.json, or call
apply(summary_dict) from summarize.py before it writes the file.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "p1_ladder")
FLOOR = 0.02
RUNG = {"L0": "L0_climatology", "L1": "L1_player_mean", "L2": "L2_selected", "L3": "L3_minutes_rate", "L4": "L4_hier_shrink",
        "L4p": "L4_hier_shrink_plugin"}


def floor_violations(cal_mode, cand, ref, targets):
    """Targets on which the candidate's PIT-KS or |cov80 - 0.8| is worse than the reference's by more than FLOOR."""
    bad = []
    for t in targets:
        c, r = cal_mode.get(t, {}).get(RUNG[cand]), cal_mode.get(t, {}).get(RUNG[ref])
        if c is None or r is None:
            continue
        ks_worse = c["ks"] - r["ks"] > FLOOR
        cov_worse = abs(c["cov80"] - 0.8) - abs(r["cov80"] - 0.8) > FLOOR
        if ks_worse or cov_worse:
            bad.append({"target": t, "ks": [r["ks"], c["ks"]], "cov80": [r["cov80"], c["cov80"]],
                        "ks_worse": bool(ks_worse), "cov80_worse": bool(cov_worse)})
    return bad


def apply(res):
    targets = [t for t in res["mean_loss_regular_dev"]["U"] if not t.startswith("_")]
    for mode in ("U", "A"):
        cal = res["calibration_regular_dev"][mode]
        for o in res["credit_chain"][mode]["comparisons"]:
            k = o["core"]
            if "decision" not in k:
                continue
            raw = k.get("decision_raw", k["decision"])
            k["decision_raw"] = raw
            reasons = []
            if raw == "SUPPORTED":
                if k.get("harmful_secondary"):
                    reasons.append({"constraint": "no-harm", "harmful_secondary": k["harmful_secondary"]})
                viol = floor_violations(cal, o["candidate"], o["reference"], targets)
                if viol:
                    reasons.append({"constraint": "calibration-floor", "targets": viol})
            k["downgrade_reasons"] = reasons
            k["decision"] = "PROMISING" if (raw == "SUPPORTED" and reasons) else raw
            k["constraints_applied"] = ["no-harm", "calibration-floor"]
    res["constraints"] = {"applied": True, "floor": FLOOR, "source": "docs/prereg/08_metric_spec.md section 5 items 1-2",
                          "floor_reading": "any core target of the comparison, same mode, regular season, development window"}
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "summary.json")
    res = apply(json.load(open(path)))
    json.dump(res, open(path, "w"), indent=1, default=float)
    for mode in ("U", "A"):
        for o in res["credit_chain"][mode]["comparisons"]:
            k = o["core"]
            if "decision" in k:
                print(mode, o["candidate"], "vs", o["reference"], k["decision_raw"], "->", k["decision"],
                      json.dumps(k["downgrade_reasons"])[:200])
    print("applied to", path)


if __name__ == "__main__":
    main()
