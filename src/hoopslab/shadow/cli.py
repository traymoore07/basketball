"""Executor command line for the shadow lane. No model code is touched; every input is a file.

    python -m hoopslab.shadow.cli forecast --snapshot DIR --game-id ID [--created-at ISO] [--lane SHADOW_HOME]
        DIR holds game.csv|parquet, player_game.csv|parquet, eligible.csv|parquet (docs/shadow/01). The cutoff is
        taken from the game's scheduled tip (T-60). Builds the snapshot, forecasts, locks; prints forecast_id and hash.
    python -m hoopslab.shadow.cli score --lane SHADOW_HOME --game-id ID --forecast-id FID --box FILE --result-time ISO
                                        [--revision N]
        FILE holds the final PLAYER_GAME rows of that game. Scores the locked package, appends to the ledger.
    python -m hoopslab.shadow.cli void --lane SHADOW_HOME --forecast-id FID --reason {postponed,cancelled,tip_changed}
    python -m hoopslab.shadow.cli event --lane SHADOW_HOME --forecast-id FID --kind degraded --note TEXT
    python -m hoopslab.shadow.cli report --lane SHADOW_HOME
        Cumulative table only (no game ids, no per-player rows): the only output meant to travel toward research.
    python -m hoopslab.shadow.cli verify --lane SHADOW_HOME
        Re-hashes every locked package; prints any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from ..asof import to_utc
from . import model as M
from .package import LockStore, build_snapshot, make_forecast_package, verify_locked
from .scoring import Ledger, score_locked_forecast

DEFAULT_LANE = os.environ.get("SHADOW_HOME", os.path.join(os.path.expanduser("~"), "Documents", "Claude", "shadow_lane"))


def _read(d: str, name: str) -> pd.DataFrame:
    for ext, fn in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
        p = os.path.join(d, f"{name}.{ext}")
        if os.path.exists(p):
            return fn(p)
    raise FileNotFoundError(f"{name}.parquet or {name}.csv not found in {d}")


class FileProvider:
    """SnapshotProvider over the three files of a snapshot directory (what an executor writes)."""

    def __init__(self, d: str):
        self.d = d
        self.provider_id = f"files:{os.path.basename(os.path.abspath(d))}"

    def schedule(self, cutoff):
        return _read(self.d, "game")

    def appearance_history(self, cutoff):
        return _read(self.d, "player_game")

    def eligible_players(self, cutoff, game_ids):
        e = _read(self.d, "eligible")
        return e[e["game_id"].isin(game_ids)]


def cmd_forecast(a):
    prov = FileProvider(a.snapshot)
    g = prov.schedule(None)
    row = g[g["game_id"] == a.game_id]
    if not len(row):
        sys.exit(f"game {a.game_id} not in {a.snapshot}/game.*")
    tip = to_utc(row["scheduled_tip"].iloc[0])
    cutoff = tip - pd.Timedelta(minutes=M.LEAD_MINUTES)
    snap = build_snapshot(prov, [a.game_id], cutoff, history_start=a.history_start)
    pkg = make_forecast_package(snap, a.game_id, created_at=a.created_at, allow_created_before_cutoff=a.allow_early)
    locked = LockStore(a.lane).lock(pkg)
    print(json.dumps({"forecast_id": locked["forecast_id"], "lock_hash": locked["lock_hash"], "information_cutoff": locked["information_cutoff"],
                      "forecast_created_at": locked["forecast_created_at"], "data_snapshot_id": locked["data_snapshot_id"],
                      "players_issued": sum(p["distributions"] is not None for p in locked["players"]),
                      "availability_counts": locked["availability_counts"]}, indent=1))


def cmd_score(a):
    store, led = LockStore(a.lane), Ledger(a.lane)
    pkg = store.load(a.game_id, a.forecast_id)
    box = pd.read_parquet(a.box) if a.box.endswith(".parquet") else pd.read_csv(a.box)
    scored = score_locked_forecast(pkg, box, result_time=a.result_time, score_revision=a.revision)
    n = led.append(scored)
    print(json.dumps({"forecast_id": a.forecast_id, "rows_appended": n, "counts": scored["counts"]}, indent=1))


def _event(lane, rec):
    led = Ledger(lane)
    with open(led.events, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps(rec))


def cmd_void(a):
    _event(a.lane, {"event": "voided", "forecast_id": a.forecast_id, "reason": a.reason, "at": str(pd.Timestamp.now(tz="UTC"))})


def cmd_event(a):
    _event(a.lane, {"event": a.kind, "forecast_id": a.forecast_id, "note": a.note, "at": str(pd.Timestamp.now(tz="UTC"))})


def cmd_report(a):
    cum = Ledger(a.lane).cumulative()
    if not len(cum):
        print("ledger empty")
        return
    print(cum.to_string(index=False))


def cmd_verify(a):
    root = os.path.join(a.lane, "forecasts")
    bad, n = [], 0
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".json"):
                n += 1
                pkg = json.load(open(os.path.join(d, f), encoding="utf-8"))
                if not verify_locked(pkg):
                    bad.append(os.path.join(d, f))
    print(json.dumps({"packages": n, "hash_mismatches": bad}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoopslab.shadow.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast"); f.add_argument("--snapshot", required=True); f.add_argument("--game-id", required=True)
    f.add_argument("--created-at", default=None); f.add_argument("--lane", default=DEFAULT_LANE); f.add_argument("--history-start", default=None)
    f.add_argument("--allow-early", action="store_true"); f.set_defaults(fn=cmd_forecast)
    s = sub.add_parser("score"); s.add_argument("--lane", default=DEFAULT_LANE); s.add_argument("--game-id", required=True)
    s.add_argument("--forecast-id", required=True); s.add_argument("--box", required=True); s.add_argument("--result-time", required=True)
    s.add_argument("--revision", type=int, default=1); s.set_defaults(fn=cmd_score)
    v = sub.add_parser("void"); v.add_argument("--lane", default=DEFAULT_LANE); v.add_argument("--forecast-id", required=True)
    v.add_argument("--reason", required=True, choices=["postponed", "cancelled", "tip_changed"]); v.set_defaults(fn=cmd_void)
    e = sub.add_parser("event"); e.add_argument("--lane", default=DEFAULT_LANE); e.add_argument("--forecast-id", required=True)
    e.add_argument("--kind", required=True); e.add_argument("--note", default=""); e.set_defaults(fn=cmd_event)
    r = sub.add_parser("report"); r.add_argument("--lane", default=DEFAULT_LANE); r.set_defaults(fn=cmd_report)
    y = sub.add_parser("verify"); y.add_argument("--lane", default=DEFAULT_LANE); y.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
