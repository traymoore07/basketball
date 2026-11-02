"""Run run_season.py for a list of seasons with N parallel processes (every forecast time refits from scratch,
so splitting by season is exact; P1 pattern). Skips seasons whose done_<S>.json exists.
    python experiments/research_season/launch.py --seasons 1998 1999 2000 ... --workers 4"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "research_season")
ap = argparse.ArgumentParser()
ap.add_argument("--seasons", nargs="+", type=int, required=True)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
seasons = [s for s in a.seasons if not os.path.exists(os.path.join(OUT, f"done_{s}.json"))]


def run(s):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_season.py"), "--season", str(s)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    open(os.path.join(OUT, f"stdout_{s}.txt"), "w").write(r.stdout + "\n" + r.stderr)
    print(f"season {s} exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
    return r.returncode


os.makedirs(OUT, exist_ok=True)
print("seasons", seasons, flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    codes = list(ex.map(run, seasons))
print("ALL DONE codes", codes, flush=True)
