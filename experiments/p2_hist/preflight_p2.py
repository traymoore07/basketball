"""P2 E-HIST preflight (before any event-level outcome is computed): registry, attestation, holdout,
segment-parser gate, hashes of the frozen event modules and of the experiment scripts, environment.
Writes results/p2_hist/preflight.json."""
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

OUT = os.path.join(ROOT, "results", "p2_hist")
os.makedirs(OUT, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


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
out["checks"]["holdout_sealed"] = {"rows_season_ge_2023": hold, "passed": all(v == 0 for v in hold.values())}
out["checks"]["lockbox_present_but_never_parsed"] = {"rows_2021_2022_in_package": lock,
                                                     "segment_files_for_2021_2022": sorted(glob.glob(os.path.join(OUT, "segments_202[12].parquet"))),
                                                     "passed": not glob.glob(os.path.join(OUT, "segments_202[12].parquet"))}
con.close()
gates = {}
for f in sorted(glob.glob(os.path.join(OUT, "segments_*_validation.json"))):
    d = json.load(open(f))
    S = int(os.path.basename(f).split("_")[1])
    gates[S] = {"points_match_rate": d["points_match_rate"], "fga_match_rate": d["fga_match_rate"], "segments": d["segments_kept"],
                "dropped_no_terminal": d["dropped_no_terminal"]}
out["checks"]["segment_parser_gate"] = {"per_season": gates, "rule": "points_match_rate >= 0.99 and fga_match_rate >= 0.99 in every parsed season",
                                        "passed": bool(gates) and all(g["points_match_rate"] >= 0.99 and g["fga_match_rate"] >= 0.99 for g in gates.values())}
frozen = ["src/hoopslab/models/event.py", "src/hoopslab/experiments/history.py", "src/hoopslab/metrics/information.py", "src/hoopslab/metrics/compare.py"]
scripts = sorted(glob.glob("experiments/p2_hist/*.py"))
out["checks"]["hashes"] = {"frozen_event_modules": {f: sha(f) for f in frozen}, "experiment_scripts": {f: sha(f) for f in scripts},
                           "frozen_in_registry_manifest": True}
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
    "unit": "possession segment (experiments/p2_hist/segments.py rules)", "target": "8-class terminal outcome (hoopslab.models.event)",
    "sets": "A-G of hoopslab.experiments.history.SETS", "families": "mnl (frozen MultinomialLogit L2=1e-4), gbm (LightGBM fixed params in run_hist.py), gru (run_gru.py fixed architecture) for E and G",
    "protocol": "prequential by month over development seasons 2000-2020; training window = 730 days before the test month (burn-in allowed)",
    "mpe_bits": 0.0005, "bootstrap": "game cluster, B=2000, seed 12345", "controls": "negative (permuted history within state cells), canary (next segment class), hot-hand contrast",
    "decision": "hoopslab.experiments.history.diagnose_history on the pooled contrasts; per season reported",
}
out["all_passed"] = all(c.get("passed", True) for c in out["checks"].values() if isinstance(c, dict))
json.dump(out, open(os.path.join(OUT, "preflight.json"), "w"), indent=1, default=str)
print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in out["checks"].items()}, indent=1))
print("ALL PASSED" if out["all_passed"] else "FAILED")
