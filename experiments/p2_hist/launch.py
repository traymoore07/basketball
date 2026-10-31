"""Spawn detached worker processes for the E-HIST runs (one per group of test seasons), logging to results/p2_hist/.
    python experiments/p2_hist/launch.py --family mnl --threads 2 --groups 2000-2001 2002-2003 ...
    python experiments/p2_hist/launch.py --family gbm --threads 4 --groups ...
    python experiments/p2_hist/launch.py --family gru --threads 4 --groups ...
Workers skip months whose output already exists, so a relaunch resumes."""
import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "p2_hist")
ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["mnl", "gbm", "gru"], required=True)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--groups", nargs="+", required=True, help="season ranges like 2000-2001")
ap.add_argument("--extra", default="", help="extra worker arguments as one string, e.g. \"--every 3 --controls\"")
a = ap.parse_args()
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
pids = []
for g in a.groups:
    lo, hi = (int(x) for x in g.split("-"))
    seasons = [str(s) for s in range(lo, hi + 1)]
    script = "run_gru.py" if a.family == "gru" else "run_hist.py"
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "p2_hist", script), "--test-seasons", *seasons]
    if a.family != "gru":
        cmd += ["--family", a.family]
    cmd += shlex.split(a.extra)
    log = open(os.path.join(OUT, f"stdout_{a.family}_{lo}_{hi}.txt"), "a")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    pids.append((g, p.pid))
print("launched", pids)
