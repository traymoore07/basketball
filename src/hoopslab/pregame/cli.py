"""X1 CONTEXT-SHADOW executor CLI (skeleton; shadow_context/README.md).

    python -m hoopslab.pregame.cli versions                      list the version plan and which versions are forecastable
    python -m hoopslab.pregame.cli validate --snapshot DIR       check an L1 pregame-context snapshot against the contract
    python -m hoopslab.pregame.cli eligible --snapshot DIR --out FILE
                                                                 derive the frozen v0.1 ELIGIBLE table (one way; UNKNOWN never ACTIVE)
    python -m hoopslab.pregame.cli forecast --version V --snapshot DIR --shadow-snapshot DIR --game-id G
                                                                 LIVE LANE ONLY: run a FROZEN version on one game into CONTEXT_HOME/live_results/V
    python -m hoopslab.pregame.cli score / report / verify       LIVE LANE ONLY: per-version ledgers (delegated to hoopslab.shadow.cli per version lane)

The live-lane guard: HOOPSLAB_LANE must be "live" and CONTEXT_HOME must be set to a directory outside this
repository; otherwise forecast / score / report / verify refuse. Research Claude runs only versions,
validate and eligible, on fixtures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import contract as C
from . import versions as V

DEFAULT_CONTEXT_HOME = r"C:\Users\Tray\Documents\Claude\context_lane"


def context_home() -> str:
    return os.environ.get("CONTEXT_HOME", DEFAULT_CONTEXT_HOME)


def assert_live_lane():
    lane = os.environ.get("HOOPSLAB_LANE", "research")
    home = os.path.abspath(context_home())
    root = os.path.abspath(V.repo_root())
    if lane != "live":
        raise SystemExit("refused: HOOPSLAB_LANE is not 'live'. Research Claude may not forecast, score or report prospective games "
                         "(docs/lanes/00_lane_registry.md section 2).")
    if home == root or home.startswith(root + os.sep):
        raise SystemExit(f"refused: CONTEXT_HOME {home} lies inside the research repository; prospective results must live outside it.")


def cmd_versions(a):
    plan = V.load_plan()
    for v in plan["versions"]:
        ok, probs = V.verify_version(v["model_version"])
        print(f"{v['model_version']:28s} {v['status']:8s} {v['role']:40s} forecastable={ok}" + (f"  ({'; '.join(probs)})" if probs else ""))


def cmd_validate(a):
    snap = C.load_and_validate(a.snapshot, a.cutoff)
    rep = {"snapshot": a.snapshot, "cutoff": None if snap.cutoff is None or str(snap.cutoff) == "NaT" else snap.cutoff.isoformat(),
           "ok": snap.ok, "problems": snap.problems, "counts": snap.counts, "table_hashes": snap.hashes,
           "snapshot_id": snap.snapshot_id if snap.ok else None}
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if snap.ok else 2


def cmd_eligible(a):
    snap = C.load_and_validate(a.snapshot, a.cutoff)
    if not snap.ok:
        print("snapshot invalid:", "; ".join(snap.problems))
        return 2
    el = C.to_eligible_v01(snap)
    el.to_csv(a.out, index=False)
    counts = el["availability_state"].value_counts().to_dict()
    print(f"wrote {a.out}: {len(el)} rows {counts}; snapshot_id {snap.snapshot_id}")
    return 0


def cmd_forecast(a):
    assert_live_lane()
    ok, probs = V.verify_version(a.version)
    if not ok:
        raise SystemExit("refused: " + "; ".join(probs))
    snap = C.load_and_validate(a.snapshot, a.cutoff)
    if not snap.ok:
        raise SystemExit("refused: snapshot invalid: " + "; ".join(snap.problems))
    lane = os.path.join(context_home(), "live_results", a.version)
    os.makedirs(lane, exist_ok=True)
    if a.version == "L2-A-SHADOW-v0.1":
        # the frozen baseline, run unchanged through its own CLI; only the ELIGIBLE table comes from the L1 snapshot
        el_path = os.path.join(a.shadow_snapshot, "eligible.csv")
        C.to_eligible_v01(snap).to_csv(el_path, index=False)
        json.dump({"pregame_snapshot": a.snapshot, "pregame_snapshot_id": snap.snapshot_id, "eligible_mapping": "to_eligible_v01"},
                  open(os.path.join(a.shadow_snapshot, "pregame_manifest.json"), "w", encoding="utf-8"), indent=1)
        from ..shadow.cli import main as shadow_main
        argv = ["forecast", "--snapshot", a.shadow_snapshot, "--game-id", a.game_id, "--lane", lane]
        if a.allow_early:
            argv.append("--allow-early")
        return shadow_main(argv)
    raise SystemExit(f"refused: {a.version} has no executor implementation yet (status {V.get_version(a.version)['status']})")


def _delegate(a, cmd):
    assert_live_lane()
    lane = os.path.join(context_home(), "live_results", a.version)
    from ..shadow.cli import main as shadow_main
    argv = [cmd, "--lane", lane] + list(a.rest)
    return shadow_main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoopslab.pregame.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("versions").set_defaults(fn=cmd_versions)
    v = sub.add_parser("validate"); v.add_argument("--snapshot", required=True); v.add_argument("--cutoff"); v.add_argument("--out"); v.set_defaults(fn=cmd_validate)
    e = sub.add_parser("eligible"); e.add_argument("--snapshot", required=True); e.add_argument("--cutoff"); e.add_argument("--out", required=True); e.set_defaults(fn=cmd_eligible)
    f = sub.add_parser("forecast"); f.add_argument("--version", required=True); f.add_argument("--snapshot", required=True)
    f.add_argument("--shadow-snapshot", required=True, help="directory with game.* and player_game.* for the frozen baseline")
    f.add_argument("--game-id", required=True); f.add_argument("--cutoff"); f.add_argument("--allow-early", action="store_true"); f.set_defaults(fn=cmd_forecast)
    for c in ("score", "report", "verify"):
        p = sub.add_parser(c); p.add_argument("--version", required=True); p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(fn=lambda a, c=c: _delegate(a, c))
    a = ap.parse_args(argv)
    rc = a.fn(a)
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
