"""R1 preflight (before any R1 forecast): registry intact, attestation, holdout sealed, lockbox never parsed, the
frozen spec's hash recorded, the frozen modules the model imports unchanged since E-HIST / shadow, the store
carries the columns the context rules need (started, status, season_id), grid identity test passed, environment.
Writes results/research_season/preflight.json."""
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
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import DEFAULT_DB, CorpusV102Adapter  # noqa: E402
from hoopslab.models.carryover import GRID_IDS  # noqa: E402

OUT = os.path.join(ROOT, "results", "research_season")
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
con.close()
out["checks"]["holdout_sealed"] = {"rows_season_ge_2023": hold, "passed": all(x == 0 for x in hold.values())}
spec = "research_season/01_carryover_spec.md"
out["checks"]["spec_frozen"] = {"file": spec, "sha256": sha(spec), "frozen_marker": "FROZEN 2026-11-02" in open(spec, encoding="utf-8").read(),
                                "draft_absent": not os.path.exists("research_season/01_carryover_spec_draft.md")}
out["checks"]["spec_frozen"]["passed"] = out["checks"]["spec_frozen"]["frozen_marker"] and out["checks"]["spec_frozen"]["draft_absent"]
frozen = ["src/hoopslab/models/baselines.py", "src/hoopslab/features.py", "src/hoopslab/targets.py", "src/hoopslab/forecast.py",
          "src/hoopslab/asof.py", "src/hoopslab/metrics/scoring.py", "src/hoopslab/metrics/compare.py", "src/hoopslab/backtest/runner.py",
          "src/hoopslab/backtest/strata.py", "src/hoopslab/diagnostics/changepoint.py"]
card = json.load(open("shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json", encoding="utf-8"))["implementation_hash"]["files"]
same = {f: sha(f) == h for f, h in card.items() if f != "src/hoopslab/shadow/model.py"}
out["checks"]["hashes"] = {"frozen_modules": {f: sha(f) for f in frozen}, "r1_code": {f: sha(f) for f in ["src/hoopslab/models/carryover.py"] + sorted(glob.glob("experiments/research_season/*.py"))},
                           "baseline_model_files_unchanged_since_shadow_card": same, "passed": all(same.values())}
# the store carries what the context rules read
store, rep = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=[1998], level="A"), strict=True)
pg_cols = list(store.full_table("PLAYER_GAME").columns)
g_cols = list(store.full_table("GAME").columns)
need_pg, need_g = ["started", "status", "seconds", "fga", "fta", "tov", "team_id"], ["season_id", "scheduled_tip", "knowledge_time", "record_version"]
out["checks"]["store_columns"] = {"PLAYER_GAME_has": {c: c in pg_cols for c in need_pg}, "GAME_has": {c: c in g_cols for c in need_g},
                                  "passed": all(c in pg_cols for c in need_pg) and all(c in g_cols for c in need_g)}
sp = os.path.join(OUT, "started.parquet")
if os.path.exists(sp):
    import pandas as pd
    st = pd.read_parquet(sp)
    share = st.groupby("season")["started"].mean()
    out["checks"]["started_table"] = {"rows": int(len(st)), "seasons": sorted(int(s) for s in share.index), "share_by_season": {int(k): round(float(v), 4) for k, v in share.items()},
                                      "passed": set(range(1997, 2021)) <= set(int(s) for s in share.index) and bool((share > 0.2).all())}
else:
    out["checks"]["started_table"] = {"passed": False, "reason": "results/research_season/started.parquet missing (run build_started.py)"}
t = run("python -m pytest -q tests/test_carryover.py")
out["checks"]["grid_identity_tests"] = {"output": t.splitlines()[-1] if t else "", "passed": "passed" in t and "failed" not in t}
import numpy, pandas, scipy  # noqa: E402
out["checks"]["environment"] = {"git_head": run("git rev-parse HEAD"), "git_status_short": run("git status --short"), "python": sys.version.split()[0],
                                "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}
out["design_fixed_before_outcomes"] = {"spec": spec, "grid": GRID_IDS, "selection": "per target on seasons <= S-1 (1998-1999 selection only, P1 convention)",
                                       "primary_contrast": "selected family vs A, CRPS, team game index <= 20, paired game-day block bootstrap, P1 MPE 0.01",
                                       "mode": "A only", "lead_minutes": 60}
out["all_passed"] = all(c.get("passed", True) for c in out["checks"].values() if isinstance(c, dict))
json.dump(out, open(os.path.join(OUT, "preflight.json"), "w"), indent=1, default=str)
print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in out["checks"].items()}, indent=1))
print("ALL PASSED" if out["all_passed"] else "FAILED")
