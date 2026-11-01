"""P2 E-STATE preflight (the E-HIST preflight extended to the state lane): registry, attestation, holdout,
lockbox never parsed in either lane, segment-parser gate (NBA from E-HIST, WNBA from the portability parse),
state-variable foul-tag gate (>= 0.98 per season, both leagues), hashes of the frozen event modules, of the
E-HIST modules the state lane imports and of the E-STATE scripts, environment. Writes results/p2_state/preflight.json."""
import ast
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

import duckdb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from hoopslab.adapters.corpus_v102 import DEFAULT_DB  # noqa: E402

HIST = os.path.join(ROOT, "results", "p2_hist")
OUT = os.path.join(ROOT, "results", "p2_state")
WNBA = os.path.join(OUT, "wnba")
os.makedirs(OUT, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def parser_gates(d):
    g = {}
    for f in sorted(glob.glob(os.path.join(d, "segments_*_validation.json"))):
        v = json.load(open(f))
        g[int(os.path.basename(f).split("_")[1])] = {"points_match_rate": v["points_match_rate"], "fga_match_rate": v["fga_match_rate"],
                                                     "segments": v["segments_kept"], "dropped_no_terminal": v["dropped_no_terminal"]}
    return g


def tag_gates(log):
    g = {}
    if os.path.exists(log):
        for line in open(log, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{") and "'tag_agreement'" in line:
                d = ast.literal_eval(line)
                g[int(d["season"])] = {"tag_agreement": d["tag_agreement"], "personal_fouls": d["personal_fouls"], "tagged": d["tagged"]}
    return g


out = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "checks": {}}
v = run("python -m hoopslab.registry verify")
out["checks"]["registry_verify"] = {"output": v.splitlines()[-1] if v else "", "passed": "INTACT" in v}
att = json.load(open("registry/ADAPTER_ATTESTATION.json", encoding="utf-8"))
mism = {f: (h, sha(f)) for f, h in att["files"].items() if sha(f) != h}
out["checks"]["adapter_attestation"] = {"attested_at": att["attested_at_utc"], "mismatches": mism, "passed": not mism}
con = duckdb.connect(DEFAULT_DB, read_only=True)
hold = {t: con.execute(f"SELECT count(*) FROM {t} WHERE season >= 2023").fetchone()[0] for t in ("possessions", "play_by_play", "player_game_box")}
lock = {t: con.execute(f"SELECT count(*) FROM {t} WHERE league='nba' AND source='nba_stats' AND season BETWEEN 2021 AND 2022").fetchone()[0]
        for t in ("possessions", "play_by_play")}
wn = {t: con.execute(f"SELECT count(*) FROM {t} WHERE league='wnba' AND source='wnba_stats' AND season >= 2021").fetchone()[0]
      for t in ("possessions", "play_by_play")}
con.close()
out["checks"]["holdout_sealed"] = {"rows_season_ge_2023": hold, "passed": all(x == 0 for x in hold.values())}
late = sorted(glob.glob(os.path.join(HIST, "segments_202[12].parquet")) + glob.glob(os.path.join(OUT, "state_202[12].parquet")) +
              glob.glob(os.path.join(WNBA, "*_202[1-9].parquet")))
out["checks"]["lockbox_present_but_never_parsed"] = {"rows_2021_2022_in_package_nba": lock, "rows_ge_2021_in_package_wnba": wn,
                                                     "files_after_2020_in_either_lane": late, "passed": not late}
nba_g, wnba_g = parser_gates(HIST), parser_gates(WNBA)
rule = "points_match_rate >= 0.99 and fga_match_rate >= 0.99 in every parsed season"
out["checks"]["segment_parser_gate_nba"] = {"per_season": nba_g, "rule": rule,
                                            "passed": bool(nba_g) and all(g["points_match_rate"] >= 0.99 and g["fga_match_rate"] >= 0.99 for g in nba_g.values())}
out["checks"]["segment_parser_gate_wnba"] = {"per_season": wnba_g, "rule": rule + " (portability, spec section 7)",
                                             "passed": bool(wnba_g) and all(g["points_match_rate"] >= 0.99 and g["fga_match_rate"] >= 0.99 for g in wnba_g.values())}
nba_t, wnba_t = tag_gates(os.path.join(OUT, "state_build_log.txt")), tag_gates(os.path.join(WNBA, "prepare_log.txt"))
trule = "share of tagged personal fouls whose '(P#.' tag equals the running count computed here >= 0.98 per season (spec section 3)"
out["checks"]["state_foul_tag_gate_nba"] = {"per_season": nba_t, "rule": trule, "seasons": sorted(nba_t),
                                            "passed": set(range(1997, 2021)) <= set(nba_t) and all(g["tag_agreement"] >= 0.98 for g in nba_t.values())}
out["checks"]["state_foul_tag_gate_wnba"] = {"per_season": wnba_t, "rule": trule,
                                             "passed": bool(wnba_t) and all(g["tag_agreement"] >= 0.98 for g in wnba_t.values())}
reg = json.load(open(os.path.join(OUT, "parser_regression_2019.txt"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "parser_regression_2019.txt")) and \
    open(os.path.join(OUT, "parser_regression_2019.txt"), encoding="utf-8").read().lstrip().startswith("{") else \
    open(os.path.join(OUT, "parser_regression_2019.txt"), encoding="utf-8").read().strip() if os.path.exists(os.path.join(OUT, "parser_regression_2019.txt")) else None
out["checks"]["nba_parser_regression_after_league_generalisation"] = {"record": reg, "passed": reg is not None and "identical" in str(reg).lower()}
frozen = ["src/hoopslab/models/event.py", "src/hoopslab/experiments/history.py", "src/hoopslab/metrics/information.py", "src/hoopslab/metrics/compare.py"]
hist_mod = ["experiments/p2_hist/segments.py", "experiments/p2_hist/slow_vars.py", "experiments/p2_hist/run_hist.py"]
scripts = sorted(glob.glob("experiments/p2_state/*.py"))
out["checks"]["hashes"] = {"frozen_event_modules": {f: sha(f) for f in frozen}, "e_hist_modules_imported": {f: sha(f) for f in hist_mod},
                           "e_state_scripts": {f: sha(f) for f in scripts},
                           "frozen_event_modules_unchanged_since_e_hist": {f: sha(f) == h for f, h in json.load(open(os.path.join(HIST, "preflight.json")))["checks"]["hashes"]["frozen_event_modules"].items()}}
out["checks"]["hashes"]["passed"] = all(out["checks"]["hashes"]["frozen_event_modules_unchanged_since_e_hist"].values())
import numpy, pandas, scipy  # noqa: E402
env = {"git_head": run("git rev-parse HEAD"), "git_status_short": run("git status --short"), "python": sys.version.split()[0],
       "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}
for m in ("sklearn", "lightgbm", "torch"):
    try:
        env[m] = __import__(m).__version__
    except Exception:  # noqa: BLE001
        env[m] = None
out["checks"]["environment"] = env
out["design_fixed_before_outcomes"] = {
    "spec": "docs/p2/02_estate_spec.md (written 2026-10-31 before any E-STATE fit; sections 3-7)",
    "states": "S0 = E-HIST set A; S1 +FB; S2 +GAP; S3 +TM; S4 +SC; S5 +PF; S6 +ROT; S7 +PC (experiments/p2_state/state_blocks.py BLOCK_ORDER)",
    "fits": "mnl 23 sets + 2 controls per month (MultinomialLogit L2=1e-4, maxiter 3000 per POSTFREEZE C23/C24); gbm 24 sets per month (run_hist fixed parameters)",
    "months": "the 62 E-HIST GBM months (compute decision in results/p2_state/launch_log.txt, before any test-month outcome); timing month 1999-12 excluded",
    "protocol": "prequential by month over development seasons 2000-2020; training window = 730 days before the test month (burn-in 1997-1999 allowed)",
    "mpe_bits": 0.0005, "bootstrap": "game cluster, B=2000, seed 12345",
    "decision": "spec section 6: regime A/B/C by the primary residual r6 (S6 vs S6 + GL + H) in logit and GBM on identical events; r7 secondary; PC proxy rule; frozen H2 S0 vs S7",
    "portability": "WNBA wnba_stats 2016-2020, test months 2018-05..2020-10, logit only, no pooling; NCAA not attempted",
}
out["all_passed"] = all(c.get("passed", True) for c in out["checks"].values() if isinstance(c, dict))
json.dump(out, open(os.path.join(OUT, "preflight.json"), "w"), indent=1, default=str)
print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in out["checks"].items()}, indent=1))
print("ALL PASSED" if out["all_passed"] else "FAILED")
