"""Realised starters for the R1 context rules (research_season/01_carryover_spec.md section 2).

At data level A the nba_stats box has no starter flag (PLAYER_GAME.started is all False); the adapter derives
it at level C from the opening lineup stint of each team (corpus_v102.py, `starters_from_opening_lineup_stint`).
Loading level C for every season in every R1 process is too heavy (EVENT, POSSESSION, SHOT tables), so this
script builds level C one season at a time and keeps only (game_id, player_id, started). run_season.py patches
the level-A store's PLAYER_GAME.started column from this table; `started` is an outcome field of the same box
row, so its knowledge time (tip + 4 h) is unchanged and the as-of view exposes it exactly as the row itself.
Writes results/research_season/started.parquet and started_log.txt (per-season share of started rows)."""
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.adapters.base import build_store  # noqa: E402
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter  # noqa: E402

OUT = os.path.join(ROOT, "results", "research_season")
SEASONS = list(range(1997, 2021))


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "started_log.txt"), "a")
    parts = []
    for S in SEASONS:
        t0 = time.time()
        store, rep = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=[S], level="C"), strict=True)
        pg = store.full_table("PLAYER_GAME")[["game_id", "player_id", "started", "seconds"]].copy()
        pg["season"] = S
        share = float(pg["started"].astype(bool).mean())
        per_team_game = pg.groupby("game_id")["started"].sum().mean() if len(pg) else 0
        msg = f"{S} rows={len(pg)} started_share={share:.4f} starters_per_game={per_team_game:.2f} seconds={time.time()-t0:.0f}"
        print(msg, file=log, flush=True)
        print(msg, flush=True)
        parts.append(pg[["game_id", "player_id", "started", "season"]])
    out = pd.concat(parts, ignore_index=True)
    out["started"] = out["started"].astype(bool)
    out.to_parquet(os.path.join(OUT, "started.parquet"), index=False)
    print("wrote", len(out), "rows", file=log, flush=True)


if __name__ == "__main__":
    main()
