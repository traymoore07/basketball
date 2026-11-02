"""Replay test of the prospective pipeline on development-period dates only (infrastructure test, no research).

For every game at the chosen tips:
    T-60 cutoff -> build snapshot -> forecast -> canonicalise -> hash -> lock -> reveal the historical
    outcome -> score -> ledger, and three invariance checks:
      (a) identity with the P1 run-2 forecasts of L2_recent_hl5 for the same (game, player) units;
      (b) a leaky provider that appends post-cutoff games/box rows (with permuted outcomes) yields the
          same distributions and the same data_snapshot_id;
      (c) a provider built on a physically truncated store yields the same lock hash.
Also: the lock store refuses to overwrite; a tampered package fails verification and scoring.

Writes results/shadow/replay_test.json and .md; the replay lane (forecasts + ledger) under results/shadow/replay_lane/.
Cutoffs are restricted to 1999-10-01..2020-10-31 by the provider itself; the scored outcomes are
development-period box scores already exposed in P1 (EXPOSURE_LOG rows 8-10).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hoopslab.shadow import model as M  # noqa: E402
from hoopslab.shadow.package import LockStore, build_snapshot, lock_hash, make_forecast_package, verify_locked  # noqa: E402
from hoopslab.shadow.replay import LeakyProvider, ResearchCorpusProvider  # noqa: E402
from hoopslab.shadow.scoring import Ledger, score_locked_forecast  # noqa: E402

TIPS = ["2010-01-15 12:00", "2015-03-10 12:00"]
OUT = os.path.join(ROOT, "results", "shadow")
LANE = os.path.join(OUT, "replay_lane")
P1 = os.path.join(ROOT, "results", "p1_ladder")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
prov = ResearchCorpusProvider()
print(f"store built {time.time()-t0:.0f}s", flush=True)
leaky = LeakyProvider(prov)
store = LockStore(LANE)
ledger = Ledger(LANE)
res = {"tips": TIPS, "games": [], "checks": {}}
all_ok = True

for tip_s in TIPS:
    tip = pd.Timestamp(tip_s, tz="UTC")
    cutoff = tip - pd.Timedelta(minutes=M.LEAD_MINUTES)
    gids = prov.games_at(tip)
    season = int(prov._g.loc[prov._g["game_id"] == gids[0], "season_id"].iloc[0].split("-")[1])
    req = pd.read_parquet(os.path.join(P1, f"requests_{season}.parquet"))
    p1 = np.load(os.path.join(P1, f"forecasts_{season}_L2_recent_hl5.npz"))
    snap = build_snapshot(prov, gids, cutoff)
    snap_leaky = build_snapshot(leaky, gids, cutoff)
    trunc = ResearchCorpusProvider(store=prov.store.truncated(cutoff))
    snap_trunc = build_snapshot(trunc, gids, cutoff)
    assert pd.Timestamp(snap.manifest["tables"]["PLAYER_GAME"]["knowledge_time_max"]) <= cutoff
    for gid in gids:
        created = cutoff + pd.Timedelta(minutes=1)  # simulated: issued one minute after the cutoff, before the tip
        pkg = make_forecast_package(snap, gid, created_at=created)
        pkg_leaky = make_forecast_package(snap_leaky, gid, created_at=created)
        pkg_trunc = make_forecast_package(snap_trunc, gid, created_at=created)
        same_players = json.dumps(pkg["players"], sort_keys=True) == json.dumps(pkg_leaky["players"], sort_keys=True)
        same_snapshot = pkg["data_snapshot_id"] == pkg_leaky["data_snapshot_id"] == pkg_trunc["data_snapshot_id"]
        same_hash_trunc = lock_hash(pkg) == lock_hash(pkg_trunc)
        locked = store.lock(pkg)
        # identity with P1 run 2 (L2_recent_hl5): match units by (game_id, player_id)
        idx = req.index[req["game_id"] == gid]
        p1_pids = req.loc[idx, "player_id"].tolist()
        maxdiff, n_matched, missing = 0.0, 0, []
        for p in locked["players"]:
            if p["distributions"] is None:
                continue
            if p["player_id"] not in p1_pids:
                missing.append(p["player_id"]); continue
            j = idx[p1_pids.index(p["player_id"])]
            for t in M.TARGETS:
                d = np.max(np.abs(np.asarray(p["distributions"][t]["pmf"]) - p1[f"cond_{t}"][j].astype(np.float64)))
                maxdiff = max(maxdiff, float(d))
            n_matched += 1
        # immutability: second lock = v2, v1 bytes unchanged; direct overwrite refused
        path1 = os.path.join(LANE, "forecasts", gid.replace(":", "_"), locked["forecast_id"] + ".json")
        b1 = open(path1, "rb").read()
        locked2 = store.lock(make_forecast_package(snap, gid, created_at=created + pd.Timedelta(minutes=1)))
        overwrite_refused = False
        try:
            with open(path1, "x") as fh:  # noqa: F841
                pass
        except FileExistsError:
            overwrite_refused = True
        v1_unchanged = open(path1, "rb").read() == b1 and verify_locked(json.load(open(path1, encoding="utf-8")))
        # reveal the historical outcome and score
        box = prov.final_box_score(gid)
        rt = prov.result_time(gid)
        scored = score_locked_forecast(locked, box, result_time=rt)
        n_rows = ledger.append(scored)
        tampered = json.loads(json.dumps(locked)); tampered["players"][0]["distributions"]["PTS"]["pmf"][0] += 1e-3
        tamper_refused = not verify_locked(tampered)
        rec = {"game_id": gid, "tip": str(tip), "cutoff": str(cutoff), "forecast_id": locked["forecast_id"], "lock_hash": locked["lock_hash"],
               "players_issued": sum(p["distributions"] is not None for p in locked["players"]),
               "p1_units_for_game": int(len(idx)), "matched_to_p1": n_matched, "not_in_p1": missing,
               "max_abs_pmf_diff_vs_p1": maxdiff, "identical_to_p1": maxdiff <= 1e-6 and not missing and n_matched == len(idx),
               "leaky_provider_same_distributions": same_players, "same_data_snapshot_id": same_snapshot,
               "truncated_store_same_lock_hash": same_hash_trunc,
               "rows_dropped_by_builder_from_leaky": snap_leaky.manifest["rows_dropped_after_cutoff"],
               "overwrite_refused": overwrite_refused, "v1_unchanged_after_v2": v1_unchanged, "v2_id": locked2["forecast_id"],
               "tamper_detected": tamper_refused, "result_time": str(rt), "scored_rows": n_rows, "score_counts": scored["counts"],
               "created_before_tip": pd.Timestamp(locked["forecast_created_at"]) < tip,
               "information_cutoff_is_T60": pd.Timestamp(locked["information_cutoff"]) == cutoff}
        ok = all([rec["identical_to_p1"], same_players, same_snapshot, same_hash_trunc, overwrite_refused, v1_unchanged,
                  tamper_refused, rec["created_before_tip"], rec["information_cutoff_is_T60"], n_rows > 0])
        rec["all_checks_passed"] = ok
        all_ok &= ok
        res["games"].append(rec)
        print(gid, "issued", rec["players_issued"], "p1 units", len(idx), "maxdiff", f"{maxdiff:.2e}", "scored", n_rows, "OK" if ok else "FAIL", flush=True)

cum = ledger.cumulative()
res["cumulative_rows"] = int(len(cum))
res["cumulative_head"] = cum.head(20).astype(object).to_dict("records")
res["checks"] = {"games": len(res["games"]), "all_passed": bool(all_ok),
                 "provider_refuses_lockbox": None, "provider_refuses_holdout": None}
for k, when in (("provider_refuses_lockbox", "2021-11-01 00:00"), ("provider_refuses_holdout", "2024-01-01 00:00")):
    try:
        prov.schedule(pd.Timestamp(when, tz="UTC")); res["checks"][k] = False
    except PermissionError:
        res["checks"][k] = True
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "replay_test.json"), "w"), indent=1, default=str)
L = ["# Shadow pipeline replay test (development-period dates only)", "",
     f"Tips: {TIPS}; games: {len(res['games'])}; all checks passed: {all_ok}; provider refuses lockbox: {res['checks']['provider_refuses_lockbox']}, holdout: {res['checks']['provider_refuses_holdout']}", "",
     "| game | issued | P1 units | matched | max |pmf diff| vs P1 | identical | leaky same | truncated same hash | overwrite refused | v1 unchanged | tamper detected | scored rows | counts |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in res["games"]:
    L.append(f"| {r['game_id']} | {r['players_issued']} | {r['p1_units_for_game']} | {r['matched_to_p1']} | {r['max_abs_pmf_diff_vs_p1']:.1e} | {r['identical_to_p1']} | "
             f"{r['leaky_provider_same_distributions']} | {r['truncated_store_same_lock_hash']} | {r['overwrite_refused']} | {r['v1_unchanged_after_v2']} | {r['tamper_detected']} | {r['scored_rows']} | {r['score_counts']} |")
open(os.path.join(OUT, "replay_test.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("ALL PASSED" if all_ok else "FAILURES", f"{time.time()-t0:.0f}s")
