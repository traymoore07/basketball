"""E-STATE portability preparation (docs/p2/02_estate_spec.md section 7): WNBA wnba_stats seasons with lineups
(2016-2020) through the same segment parser (regulation clock rescaled from 600-s to 720-s periods; overtime
unchanged), the same gate (points and field-goal attempts reconcile >= 0.99), the E-HIST slow variables and the
E-STATE state variables (last-two-minutes window = 144 rescaled seconds = 2 real minutes). Writes
results/p2_state/wnba/{segments,slow,state}_{S}.parquet and validation json. No model is fitted here."""
import json
import os
import sys
import time

import duckdb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "p2_hist"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.adapters.corpus_v102 import DEFAULT_DB  # noqa: E402
import segments  # noqa: E402
import slow_vars  # noqa: E402
import state_vars  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_state", "wnba")
SEASONS = [2016, 2017, 2018, 2019, 2020]


def main():
    os.makedirs(OUT, exist_ok=True)
    segments.configure("wnba", "wnba_stats", "wnba", 600.0)
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    log = open(os.path.join(OUT, "prepare_log.txt"), "a")
    for S in SEASONS:
        t0 = time.time()
        P, st = segments.parse_season(S, con)
        P.to_parquet(os.path.join(OUT, f"segments_{S}.parquet"), index=False)
        st["seconds"] = round(time.time() - t0)
        json.dump(st, open(os.path.join(OUT, f"segments_{S}_validation.json"), "w"), indent=1)
        msg = f"{S} segments={st['segments_kept']} possessions={st['possessions']} points_match={st['points_match_rate']} fga_match={st['fga_match_rate']} dropped={st['dropped_no_terminal']}"
        print(msg, file=log, flush=True)
        print(msg, flush=True)
        slow_vars.build_slow(S, con, league="wnba", source="wnba_stats", seg_dir=OUT, out_dir=OUT)
        out, info = state_vars.build_season(S, con, league="wnba", source="wnba_stats", prefix="wnba", hist_dir=OUT, l2m_s=144.0)
        out.to_parquet(os.path.join(OUT, f"state_{S}.parquet"), index=False)
        print(info, file=log, flush=True)
        print(info, flush=True)


if __name__ == "__main__":
    main()
