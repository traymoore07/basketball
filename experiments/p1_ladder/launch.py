"""Run run_season.py for a list of seasons with N parallel processes. Every model refits from
scratch at every forecast time, so splitting the backtest by season is exact."""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--seasons", nargs="+", type=int, required=True)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()


def run(s):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_season.py"), "--season", str(s)], cwd=ROOT,
                       capture_output=True, text=True)
    open(os.path.join(ROOT, "results", "p1_ladder", f"stdout_{s}.txt"), "w").write(r.stdout + "\n" + r.stderr)
    print(f"season {s} exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
    return r.returncode


with ThreadPoolExecutor(a.workers) as ex:
    codes = list(ex.map(run, a.seasons))
print("ALL DONE codes", codes, flush=True)
