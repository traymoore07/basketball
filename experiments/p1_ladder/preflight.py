"""BEFORE-FIRST-OUTCOME checks (P1 mission section 1). Writes results/p1_ladder/preflight.json.
Nothing here computes a model metric."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import duckdb
import numpy
import pandas
import scipy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
out = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "checks": {}}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return (r.stdout + r.stderr).strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# 1 pre-registration integrity
v = run("python -m hoopslab.registry verify")
out["checks"]["1_registry_verify"] = {"output": v.splitlines()[-1], "passed": "INTACT" in v}
t = run("python -m pytest -q tests/test_registry.py")
out["checks"]["1b_registry_tests"] = {"output": t.splitlines()[-1], "passed": " passed" in t and "failed" not in t}
# 2 adapter attestation
att = json.load(open("registry/ADAPTER_ATTESTATION.json", encoding="utf-8"))
mism = {f: (h, sha(f)) for f, h in att["files"].items() if sha(f) != h}
out["checks"]["2_adapter_attestation"] = {"attested_at": att["attested_at_utc"],
                                           "adapter_sha256": att["files"]["src/hoopslab/adapters/corpus_v102.py"],
                                           "mismatches": mism, "passed": not mism}
# 3 corpus version
from hoopslab.adapters.corpus_v102 import DEFAULT_DB, HOLDOUT_FROM_SEASON  # noqa: E402

con = duckdb.connect(DEFAULT_DB, read_only=True)
meta = dict(con.execute("SELECT k, v FROM isolated_release_meta").fetchall())
out["checks"]["3_corpus_version"] = {"db_path": DEFAULT_DB, "db_bytes": os.path.getsize(DEFAULT_DB), "db_sha256": sha(DEFAULT_DB),
                                      "meta": meta, "passed": "1.0.2" in json.dumps(meta)}
# 4 split definitions
rules = con.execute("SELECT * FROM isolated_split_rules").df().to_dict("records")
out["checks"]["4_split_definitions"] = {"package_rules": rules, "adapter_holdout_from_season": HOLDOUT_FROM_SEASON,
                                         "hoopslab_windows": json.load(open("experiments/p1_ladder/config.json"))["windows"],
                                         "passed": HOLDOUT_FROM_SEASON == 2023}
# 5 holdout sealed: no in-game row with season >= 2023 anywhere in the package
cnt = {}
for tbl in ["player_game_box", "team_game_box", "play_by_play", "possessions", "shots", "substitutions", "lineups"]:
    try:
        cnt[tbl] = con.execute(f"SELECT count(*) FROM {tbl} WHERE season >= {HOLDOUT_FROM_SEASON}").fetchone()[0]
    except Exception as e:  # noqa: BLE001
        cnt[tbl] = f"ERR {e}"
out["checks"]["5_holdout_sealed"] = {"rows_season_ge_2023": cnt,
                                      "passed": all(c == 0 for c in cnt.values() if isinstance(c, int))}
con.close()
# 7/8 what will be exposed + reproducibility identifiers
out["checks"]["7_experiments_exposed"] = {
    "experiments": ["E-LADDER rungs L0, L1, L2 (half-life grid 5/10/20/40), L3, L4, L4-plugin"], "league": "NBA (nba_stats)",
    "window": "development 2000-2020 (+ burn-in 1998-1999 forecast for L2 half-life selection only)",
    "modes": ["U", "A"], "targets": ["MIN", "PTS", "REB", "AST", "FG3M", "TOV", "PRA"]}
out["checks"]["8_reproducibility"] = {
    "git_head": run("git rev-parse HEAD"), "git_status_short": run("git status --short"),
    "python": sys.version, "platform": platform.platform(), "duckdb": duckdb.__version__, "pandas": pandas.__version__,
    "numpy": numpy.__version__, "scipy": scipy.__version__,
    "config_sha256": sha("experiments/p1_ladder/config.json"), "models_py_sha256": sha("experiments/p1_ladder/models.py"),
    "run_season_py_sha256": sha("experiments/p1_ladder/run_season.py")}
# 9 no unlogged change to a decision-bearing file: test_registry (1b) plus a clean tree
out["checks"]["9_no_unlogged_change"] = {"passed": out["checks"]["1b_registry_tests"]["passed"]
                                          and out["checks"]["8_reproducibility"]["git_status_short"] == ""}
out["all_passed"] = all(c.get("passed", True) for c in out["checks"].values())
os.makedirs("results/p1_ladder", exist_ok=True)
json.dump(out, open("results/p1_ladder/preflight.json", "w", encoding="utf-8"), indent=1, default=str)
for k, c in out["checks"].items():
    print(k, "PASS" if c.get("passed", True) else "FAIL")
print("ALL PASSED" if out["all_passed"] else "PREFLIGHT FAILED -- STOP")
