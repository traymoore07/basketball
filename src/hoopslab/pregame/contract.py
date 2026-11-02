"""L1 PREGAME-CONTEXT-LIVE input contract (live_inputs/01_pregame_context_contract.md) and the R2 output
contract for minutes distributions.

A snapshot is one directory per information cutoff holding `manifest.json` and the tables below as CSV or
parquet. Every timestamp is UTC. The validator enforces the contract; `to_eligible_v01` derives the frozen
L2-A-SHADOW-v0.1 ELIGIBLE table (four states) one way, and UNKNOWN is never turned into ACTIVE anywhere.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from ..shadow.interface import AvailabilityState

CONTRACT_VERSION = "pregame-context/1"


class Availability(str, Enum):
    ACTIVE = "ACTIVE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    UNKNOWN = "UNKNOWN"


class SourceKind(str, Enum):
    OFFICIAL = "official"          # league / team official report or active list
    STARTERS_FEED = "starters_feed"
    BEAT_WRITER = "beat_writer"
    VENDOR = "vendor"
    INFERRED = "inferred"          # the executor's own inference; never "official"
    NONE = "none"


class RosterStatus(str, Enum):
    STANDARD = "standard"
    TWO_WAY = "two_way"
    EXHIBIT_10 = "exhibit_10"
    HARDSHIP = "hardship"
    UNKNOWN = "unknown"


class Assignment(str, Enum):
    NBA = "nba"
    G_LEAGUE = "g_league"
    UNKNOWN = "unknown"


class LineupStatus(str, Enum):
    CONFIRMED_STARTER = "CONFIRMED_STARTER"
    EXPECTED_STARTER = "EXPECTED_STARTER"
    BENCH = "BENCH"
    UNKNOWN = "UNKNOWN"


class SubRole(str, Enum):
    STARTER = "starter"
    FIRST_SUB = "first_sub"
    ROTATION = "rotation"
    DEEP = "deep"
    UNKNOWN = "unknown"


# R2 role taxonomy (research_rotation/01_opportunity_spec_draft.md section 3); "unknown" allowed live
ROLES = ["PRIMARY_CREATOR", "SECONDARY_CREATOR", "HIGH_USAGE_WING", "STARTING_CENTER", "STARTING_GUARD",
         "STARTING_OTHER", "ROTATION_BENCH", "DEEP_BENCH", "unknown"]

# table -> (required columns, timestamp columns that must be <= cutoff, enum columns)
TABLES = {
    "availability": (["game_id", "player_id", "team_id", "availability", "source", "source_kind", "official",
                      "published_at", "retrieved_at"], ["published_at", "retrieved_at"],
                     {"availability": Availability, "source_kind": SourceKind}),
    "roster": (["team_id", "player_id", "roster_status", "assignment", "as_of"], ["as_of", "added_at", "removed_at"],
               {"roster_status": RosterStatus, "assignment": Assignment}),
    "lineup": (["game_id", "team_id", "player_id", "lineup_status", "source", "source_kind", "published_at", "retrieved_at"],
               ["published_at", "retrieved_at"], {"lineup_status": LineupStatus, "source_kind": SourceKind}),
    "recent_role": (["player_id", "team_id", "as_of", "n_games", "minutes_mean", "minutes_last", "starts", "sub_role",
                     "rotation_position"], ["as_of"], {"sub_role": SubRole}),
    "team_absences": (["game_id", "team_id", "player_id", "availability", "role_removed", "source", "published_at"],
                      ["published_at"], {"availability": Availability}),
}
OPTIONAL = {"availability": ["confidence", "reason_category", "note", "cutoff_note"],
            "roster": ["added_at", "removed_at", "note"], "lineup": ["note"], "recent_role": ["note"], "team_absences": ["note"]}
REQUIRED_TABLES = ["availability", "roster"]      # lineup, recent_role, team_absences may be empty or absent (recorded)
MANIFEST_FIELDS = ["contract_version", "cutoff", "provider", "retrieved_at", "games"]


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    manifest: dict
    tables: dict[str, pd.DataFrame]
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def snapshot_id(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.cutoff.isoformat()).encode())
        for k in sorted(self.hashes):
            h.update(f"{k}:{self.hashes[k]}".encode())
        return h.hexdigest()


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _table_hash(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return hashlib.sha256(b"").hexdigest()
    d = df.astype(str).sort_values(list(df.columns), kind="stable").reset_index(drop=True)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def read_snapshot_dir(path: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    mf = os.path.join(path, "manifest.json")
    manifest = json.load(open(mf, encoding="utf-8")) if os.path.exists(mf) else {}
    tables = {}
    for name in TABLES:
        for f in glob.glob(os.path.join(path, f"{name}.*")):
            if f.endswith(".parquet"):
                tables[name] = pd.read_parquet(f)
            elif f.endswith(".csv"):
                tables[name] = pd.read_csv(f)
    return manifest, tables


def validate(manifest: dict, tables: dict[str, pd.DataFrame], cutoff=None) -> Snapshot:
    """Contract check. Never mutates the tables; never converts a state. Returns the problems list."""
    problems = []
    for k in MANIFEST_FIELDS:
        if k not in manifest:
            problems.append(f"manifest: missing {k}")
    if manifest.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
        problems.append(f"manifest: contract_version {manifest.get('contract_version')} != {CONTRACT_VERSION}")
    cut = pd.Timestamp(cutoff) if cutoff is not None else (pd.Timestamp(manifest["cutoff"]) if "cutoff" in manifest else None)
    if cut is None:
        problems.append("no cutoff")
        return Snapshot(pd.NaT, manifest, tables, problems)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    counts, hashes = {}, {}
    for name, (req, tcols, enums) in TABLES.items():
        df = tables.get(name)
        if df is None:
            if name in REQUIRED_TABLES:
                problems.append(f"{name}: table missing")
            counts[name] = 0
            hashes[name] = _table_hash(None)
            continue
        counts[name] = int(len(df))
        hashes[name] = _table_hash(df)
        miss = [c for c in req if c not in df.columns]
        if miss:
            problems.append(f"{name}: missing columns {miss}")
            continue
        for c in tcols:
            if c in df.columns:
                t = _utc(df[c])
                if c in req and t.isna().any():
                    problems.append(f"{name}.{c}: unparsable or null timestamps")
                late = t.notna() & (t > cut)
                if late.any():
                    problems.append(f"{name}.{c}: {int(late.sum())} rows after the cutoff {cut.isoformat()}")
        for c, en in enums.items():
            vals = set(df[c].astype(str))
            bad = sorted(vals - {e.value for e in en})
            if bad:
                problems.append(f"{name}.{c}: values outside the contract {bad}")
    av = tables.get("availability")
    if av is not None and all(c in av.columns for c in TABLES["availability"][0]):
        if av.duplicated(["game_id", "player_id"]).any():
            problems.append("availability: duplicate (game_id, player_id) rows")
        off = av["official"].astype(str).str.lower().isin(["true", "1"])
        inferred = av["source_kind"].astype(str) == SourceKind.INFERRED.value
        if (off & inferred).any():
            problems.append("availability: an inferred state cannot be official")
        if (off & (av["availability"].astype(str) == Availability.UNKNOWN.value)).any():
            problems.append("availability: an official row cannot be UNKNOWN")
        games = set(manifest.get("games", []))
        if games and not set(av["game_id"].astype(str)) <= games:
            problems.append("availability: game_id not in manifest.games")
    ta = tables.get("team_absences")
    if ta is not None and av is not None and "availability" in ta.columns and "player_id" in ta.columns:
        if not set(ta["availability"].astype(str)) <= {Availability.OUT.value, Availability.DOUBTFUL.value}:
            problems.append("team_absences: only OUT or DOUBTFUL players may be listed as absences")
        if "role_removed" in ta.columns and not set(ta["role_removed"].astype(str)) <= set(ROLES):
            problems.append("team_absences.role_removed: values outside the R2 role taxonomy")
        key = set(zip(av["game_id"].astype(str), av["player_id"].astype(str)))
        dangling = [k for k in zip(ta["game_id"].astype(str), ta["player_id"].astype(str)) if k not in key]
        if dangling:
            problems.append(f"team_absences: {len(dangling)} rows without an availability row")
    ln = tables.get("lineup")
    if ln is not None and "lineup_status" in ln.columns and "game_id" in ln.columns:
        conf = ln[ln["lineup_status"].astype(str) == LineupStatus.CONFIRMED_STARTER.value]
        over = conf.groupby(["game_id", "team_id"]).size()
        if (over > 5).any():
            problems.append("lineup: more than five confirmed starters for a team")
    return Snapshot(cut, manifest, tables, problems, counts, hashes)


def load_and_validate(path: str, cutoff=None) -> Snapshot:
    manifest, tables = read_snapshot_dir(path)
    return validate(manifest, tables, cutoff)


# one-way mapping to the frozen L2-A-SHADOW-v0.1 ELIGIBLE states (docs/shadow/01 section 1 and 02 section 1)
V01_MAP = {
    (Availability.ACTIVE, True): AvailabilityState.CONFIRMED_ACTIVE,
    (Availability.ACTIVE, False): AvailabilityState.EXPECTED_ACTIVE,
    (Availability.PROBABLE, True): AvailabilityState.EXPECTED_ACTIVE,
    (Availability.PROBABLE, False): AvailabilityState.EXPECTED_ACTIVE,
    (Availability.QUESTIONABLE, True): AvailabilityState.QUESTIONABLE_UNKNOWN,
    (Availability.QUESTIONABLE, False): AvailabilityState.QUESTIONABLE_UNKNOWN,
    (Availability.DOUBTFUL, True): AvailabilityState.QUESTIONABLE_UNKNOWN,
    (Availability.DOUBTFUL, False): AvailabilityState.QUESTIONABLE_UNKNOWN,
    (Availability.UNKNOWN, True): AvailabilityState.QUESTIONABLE_UNKNOWN,   # never ACTIVE
    (Availability.UNKNOWN, False): AvailabilityState.QUESTIONABLE_UNKNOWN,
    (Availability.OUT, True): AvailabilityState.OUT,
    (Availability.OUT, False): AvailabilityState.OUT,
}


def to_eligible_v01(snap: Snapshot) -> pd.DataFrame:
    """ELIGIBLE table for the frozen baseline: game_id, player_id, team_id, availability_state, status_source,
    status_time. Deterministic, one way; a row's state can only be as strong as its source (official flag)."""
    if not snap.ok:
        raise ValueError("snapshot does not pass the contract: " + "; ".join(snap.problems))
    av = snap.tables["availability"]
    off = av["official"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    states = [V01_MAP[(Availability(a), bool(o))].value for a, o in zip(av["availability"].astype(str), off)]
    out = pd.DataFrame({"game_id": av["game_id"].astype(str), "player_id": av["player_id"].astype(str),
                        "team_id": av["team_id"].astype(str), "availability_state": states,
                        "status_source": av["source"].astype(str), "status_time": _utc(av["published_at"])})
    assert not ((av["availability"].astype(str) == Availability.UNKNOWN.value).to_numpy() &
                (out["availability_state"] != AvailabilityState.QUESTIONABLE_UNKNOWN.value).to_numpy()).any()
    return out.sort_values(["game_id", "team_id", "player_id"], kind="stable").reset_index(drop=True)


# R2 output contract: a minutes distribution, never a single number
MINUTES_QUANTILES = ("p10", "p25", "p50", "p75", "p90")


@dataclass
class MinutesDistribution:
    mean: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    pmf: np.ndarray | None = None   # over 0..70 integer minutes when available

    @property
    def median(self) -> float:
        return self.p50

    def validate(self) -> list[str]:
        q = [getattr(self, k) for k in MINUTES_QUANTILES]
        p = []
        if any(np.isnan(v) for v in q + [self.mean]):
            p.append("nan")
        if any(b < a for a, b in zip(q, q[1:])):
            p.append("quantiles not monotone")
        if min(q) < 0 or max(q) > 70 or not (0 <= self.mean <= 70):
            p.append("outside 0..70 minutes")
        if self.pmf is not None:
            if len(self.pmf) != 71 or abs(float(np.sum(self.pmf)) - 1) > 1e-6 or (np.asarray(self.pmf) < 0).any():
                p.append("pmf must be 71 non-negative values summing to 1")
        return p

    def as_dict(self) -> dict:
        d = {"mean": float(self.mean), "median": float(self.p50)}
        d.update({k: float(getattr(self, k)) for k in MINUTES_QUANTILES})
        if self.pmf is not None:
            d["pmf"] = [float(x) for x in self.pmf]
        return d


def minutes_distribution_from_pmf(pmf: np.ndarray) -> MinutesDistribution:
    pmf = np.asarray(pmf, float)
    pmf = pmf / pmf.sum()
    cdf = np.cumsum(pmf)
    q = {k: float(np.searchsorted(cdf, p / 100.0, side="left")) for k, p in zip(MINUTES_QUANTILES, (10, 25, 50, 75, 90))}
    return MinutesDistribution(mean=float(np.sum(np.arange(len(pmf)) * pmf)), pmf=pmf, **q)
