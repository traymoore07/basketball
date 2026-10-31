"""Snapshot building, forecast packages, canonical hashing and the immutable lock store.

Flow (one game, one cutoff):
    build_snapshot(provider, game_ids, cutoff)  -> Snapshot (rows after the cutoff dropped and counted)
    make_forecast_package(snapshot, game_id, created_at)  -> dict (one game: metadata, players, pmfs, quantiles)
    canonical_json(package) / lock_hash(package)  -> deterministic bytes and SHA-256
    LockStore(root).lock(package)  -> writes forecasts/<game_id>/<forecast_id>.json + .sha256; refuses to overwrite

A locked package is never modified. A correction is a new version: lock() assigns v(n+1) with a new
forecast_created_at and a new hash; the earlier file stays.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess

import numpy as np
import pandas as pd

from ..asof import to_utc
from .interface import GAME_COLUMNS, ELIGIBLE_COLUMNS, PLAYER_GAME_COLUMNS, Snapshot, SnapshotProvider, validate_tables
from . import model as M

PACKAGE_SCHEMA_VERSION = "shadow-forecast-package/1"


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def _table_hash(df: pd.DataFrame, cols: list[str]) -> str:
    """Order-independent sha256 of a table: rows serialised canonically and sorted."""
    d = df[cols].copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = _utc(d[c]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    lines = sorted(json.dumps([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row],
                              separators=(",", ":"), default=str) for row in d.itertuples(index=False, name=None))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build_snapshot(provider: SnapshotProvider, game_ids: list[str], cutoff, history_start=None) -> Snapshot:
    """Pull the three tables from the provider, enforce the cutoff, validate, and write the manifest.

    Rows with knowledge_time > cutoff are DROPPED (never used) and counted in the manifest, so a provider
    that returns too much cannot leak into the forecast. ELIGIBLE rows must come from the provider's own
    cutoff-time knowledge; their status_time is checked against the cutoff.
    """
    cutoff = to_utc(cutoff)
    game = provider.schedule(cutoff).copy()
    pg = provider.appearance_history(cutoff).copy()
    el = provider.eligible_players(cutoff, list(game_ids)).copy()
    game["knowledge_time"] = _utc(game["knowledge_time"])
    game["scheduled_tip"] = _utc(game["scheduled_tip"])
    pg["knowledge_time"] = _utc(pg["knowledge_time"])
    dropped = {}
    for name, df in (("GAME", game), ("PLAYER_GAME", pg)):
        late = df["knowledge_time"] > cutoff
        dropped[name] = int(late.sum())
    game = game[game["knowledge_time"] <= cutoff].reset_index(drop=True)
    pg = pg[pg["knowledge_time"] <= cutoff].reset_index(drop=True)
    if history_start is not None:
        hs = to_utc(history_start)
        tips = game.set_index("game_id")["scheduled_tip"]
        keep = pg["game_id"].map(tips) >= hs
        dropped["PLAYER_GAME_before_history_start"] = int((~keep).sum())
        pg = pg[keep].reset_index(drop=True)
    if "status_time" in el.columns and len(el):
        st = _utc(el["status_time"])
        late_status = st > cutoff
        if late_status.any():
            raise ValueError(f"ELIGIBLE: {int(late_status.sum())} availability rows carry status_time after the cutoff")
    missing_games = sorted(set(game_ids) - set(game["game_id"]))
    if missing_games:
        raise ValueError(f"GAME rows not known at the cutoff for {missing_games}")
    errs = validate_tables(game, pg, el)
    if errs:
        raise ValueError("snapshot contract violations: " + "; ".join(errs))
    manifest = {
        "provider_id": getattr(provider, "provider_id", "unknown"),
        "cutoff": str(cutoff), "forecast_game_ids": sorted(game_ids),
        "tables": {
            "GAME": {"rows": int(len(game)), "sha256": _table_hash(game, GAME_COLUMNS),
                     "knowledge_time_max": str(game["knowledge_time"].max()) if len(game) else None},
            "PLAYER_GAME": {"rows": int(len(pg)), "sha256": _table_hash(pg, PLAYER_GAME_COLUMNS),
                            "knowledge_time_max": str(pg["knowledge_time"].max()) if len(pg) else None,
                            "earliest_tip": str(game.set_index("game_id")["scheduled_tip"].reindex(pg["game_id"]).min()) if len(pg) else None},
            "ELIGIBLE": {"rows": int(len(el)), "sha256": _table_hash(el, ELIGIBLE_COLUMNS)},
        },
        "rows_dropped_after_cutoff": dropped,
        "history_policy": M.__dict__.get("HISTORY_POLICY", None) or "see hoopslab.shadow.interface.HISTORY_POLICY",
    }
    manifest["data_snapshot_id"] = hashlib.sha256(json.dumps(
        {k: manifest["tables"][k]["sha256"] for k in manifest["tables"]} | {"cutoff": str(cutoff)}, sort_keys=True).encode()).hexdigest()
    return Snapshot(cutoff=cutoff, game=game, player_game=pg, eligible=el, manifest=manifest)


def environment_metadata() -> dict:
    import numpy, pandas, scipy  # noqa
    root = M.repo_root()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(), "hoopslab_commit": commit}


def make_forecast_package(snap: Snapshot, game_id: str, created_at=None, allow_created_before_cutoff: bool = False) -> dict:
    """One game's forecast package (unlocked). created_at defaults to now (UTC).

    Rules enforced: cutoff = scheduled_tip - 60 min; created_at < scheduled_tip; created_at >= cutoff unless the
    caller declares an earlier information state (then information_cutoff is created_at, never later).
    """
    g = snap.game.set_index("game_id").loc[game_id]
    tip = to_utc(g["scheduled_tip"])
    cutoff = tip - pd.Timedelta(minutes=M.LEAD_MINUTES)
    created = to_utc(created_at) if created_at is not None else pd.Timestamp.now(tz="UTC")
    if created >= tip:
        raise ValueError(f"forecast created at {created} is not before the scheduled tip {tip}: refused")
    if snap.cutoff != cutoff:
        raise ValueError(f"snapshot cutoff {snap.cutoff} is not the game's T-60 cutoff {cutoff}")
    if created < cutoff and not allow_created_before_cutoff:
        raise ValueError(f"forecast created at {created} before the information cutoff {cutoff}; pass allow_created_before_cutoff "
                         "to issue an early forecast (information_cutoff is then the creation time)")
    info_cutoff = min(cutoff, created)
    fc = M.forecast(snap)
    players = []
    for p in fc["players"]:
        if p["game_id"] != game_id:
            continue
        row = {k: p[k] for k in ("player_id", "team_id", "availability_state", "status_source", "status_time",
                                 "history_appearances", "history_fallback")}
        if p["pmf"] is None:
            row["distributions"] = None
        else:
            row["distributions"] = {t: {"pmf": [float(np.float32(x)) for x in p["pmf"][t]],
                                        "support": f"0..{len(p['pmf'][t]) - 1}",
                                        **M.derived_quantities(p["pmf"][t], t)} for t in M.TARGETS}
        players.append(row)
    players.sort(key=lambda r: (r["team_id"], r["player_id"]))
    el = snap.eligible[snap.eligible["game_id"] == game_id]
    return {
        "schema": PACKAGE_SCHEMA_VERSION,
        "forecast_id": None,            # assigned by LockStore.lock
        "forecast_version": None,
        "model_version": M.MODEL_VERSION,
        "forecast_mode": "A - CONDITIONAL ON APPEARANCE",
        "interpretation": "every distribution is P(stat | the player appears); nothing here is an unconditional probability",
        "game": {"game_id": game_id, "league": str(g["league"]), "season_id": str(g["season_id"]), "phase": str(g["phase"]),
                 "scheduled_tip": str(tip), "home_team_id": str(g["home_team_id"]), "away_team_id": str(g["away_team_id"]),
                 "status_at_cutoff": str(g["status"])},
        "forecast_created_at": str(created), "scheduled_tip": str(tip), "information_cutoff": str(info_cutoff),
        "data_snapshot_id": snap.manifest["data_snapshot_id"], "input_data_manifest": snap.manifest,
        "model": {"version": M.MODEL_VERSION, "half_life_appearances": M.HALF_LIFE_APPEARANCES, "mode": M.FORECAST_MODE,
                  "targets": M.TARGETS, "implementation_hash": M.model_implementation_hash(),
                  "pmf_construction": M.PMF_CONSTRUCTION, "dispersion_r_fitted": fc["dispersion_r"],
                  "league_appearances_in_history": fc["league_appearances_in_history"]},
        "availability_counts": {s: int((el["availability_state"].astype(str) == s).sum())
                                for s in sorted(el["availability_state"].astype(str).unique())},
        "players": players,
        "environment": environment_metadata(),
        "lock_hash": None,
    }


def canonical_json(package: dict) -> bytes:
    """Deterministic serialisation: sorted keys, no whitespace, floats as shortest round-trip repr."""
    body = {k: v for k, v in package.items() if k != "lock_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def lock_hash(package: dict) -> str:
    return hashlib.sha256(canonical_json(package)).hexdigest()


class LockStore:
    """Append-only forecast store: forecasts/<game_id>/<forecast_id>.json. Existing files are never rewritten."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "forecasts"), exist_ok=True)

    def _dir(self, game_id: str) -> str:
        d = os.path.join(self.root, "forecasts", game_id.replace(":", "_"))
        os.makedirs(d, exist_ok=True)
        return d

    def versions(self, game_id: str, model_version: str) -> list[str]:
        d = self._dir(game_id)
        pref = model_version + "__"
        return sorted(f for f in os.listdir(d) if f.startswith(pref) and f.endswith(".json"))

    def lock(self, package: dict) -> dict:
        """Assign forecast_id/version, compute the hash, write once. Returns the locked package."""
        gid, mv = package["game"]["game_id"], package["model_version"]
        n = len(self.versions(gid, mv)) + 1
        pkg = dict(package)
        pkg["forecast_version"] = n
        pkg["forecast_id"] = f"{mv}__{gid.replace(':', '_')}__v{n}"  # file-system safe on every platform
        pkg["lock_hash"] = lock_hash(pkg)
        path = os.path.join(self._dir(gid), pkg["forecast_id"] + ".json")
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists: locked forecasts are never overwritten")
        with open(path, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(pkg, sort_keys=True, indent=1, allow_nan=False))
        with open(path[:-5] + ".sha256", "x", encoding="utf-8", newline="\n") as fh:
            fh.write(pkg["lock_hash"] + "\n")
        try:
            os.chmod(path, 0o444)
            os.chmod(path[:-5] + ".sha256", 0o444)
        except OSError:
            pass
        return pkg

    def load(self, game_id: str, forecast_id: str) -> dict:
        path = os.path.join(self._dir(game_id), forecast_id + ".json")
        pkg = json.load(open(path, encoding="utf-8"))
        if lock_hash(pkg) != pkg["lock_hash"]:
            raise ValueError(f"{forecast_id}: content does not match its lock hash (tampered or corrupted)")
        return pkg


def verify_locked(pkg: dict) -> bool:
    return lock_hash(pkg) == pkg.get("lock_hash")


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
