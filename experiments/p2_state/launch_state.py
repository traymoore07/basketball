"""Spawn detached E-STATE workers, each on a contiguous chunk of months (memory: a worker loads only the seasons
its months need). Months default to the 62 E-HIST GBM months (docs/p2/02_estate_spec.md section 2).
    python experiments/p2_state/launch_state.py --family mnl --workers 10 --threads 2 [--months ...] [--chunk-index 0 --chunks 2]
--chunks/--chunk-index split the month list across machines first (this machine takes chunk-index of chunks)."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_state import OUT, months_from_gbm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["mnl", "gbm"], required=True)
ap.add_argument("--workers", type=int, required=True)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--months", nargs="*", default=None)
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--chunk-index", type=int, default=0)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
months = a.months or months_from_gbm()
done = {os.path.basename(f)[len(a.family) + 1:-8] for f in __import__("glob").glob(os.path.join(OUT, "bits", f"{a.family}_*.parquet"))}
months = [m for m in months if m not in done]
months = [m for i, m in enumerate(months) if (i * a.chunks) // len(months) == a.chunk_index] if a.chunks > 1 else months
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
n = len(months)
pids = []
for w in range(a.workers):
    chunk = [m for i, m in enumerate(months) if (i * a.workers) // max(n, 1) == w]
    if not chunk:
        continue
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "p2_state", "run_state.py"), "--family", a.family, "--months", *chunk]
    if a.dry:
        print(w, chunk)
        continue
    log = open(os.path.join(OUT, f"stdout_{a.family}_{chunk[0]}_{chunk[-1]}.txt"), "a")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    pids.append((chunk[0], chunk[-1], p.pid))
print("months", n, "launched", pids)
