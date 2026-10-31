"""Load the data contract and validate adapter output against it."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

CONTRACT_PATH = Path(__file__).with_name("data_contract.yaml")
LEVELS = ["A", "B", "C", "D", "E"]

# Tables whose presence defines each data level (cumulative). Each entry is a list
# of alternatives; the level needs at least one alternative per entry.
LEVEL_TABLES = {
    "A": [["RULESET"], ["GAME"], ["TEAM_GAME"], ["PLAYER_GAME"], ["ROSTER"], ["PLAYER"]],
    "B": [["EVENT", "POSSESSION"]],
    "C": [["SUBSTITUTION", "LINEUP_STINT", "POSSESSION.off_lineup"], ["STARTING_LINEUP"]],
    "D": [["MATCHUP", "TRACKING_FRAME"]],
    "E": [["PROPRIETARY_LOAD"]],
}
# Strongly recommended at level A. Without it, availability-conditional
# experiments are impossible and forecasts must be declared unconditional.
LEVEL_A_RECOMMENDED = ["INJURY_REPORT"]


@dataclass
class Field:
    name: str
    type: str
    level: str
    required: bool
    time_role: str
    values: list | None = None
    desc: str = ""


@dataclass
class Table:
    name: str
    level: str
    primary_key: list[str]
    fields: list[Field] = field(default_factory=list)
    description: str = ""

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def required_fields(self, level: str = "E") -> list[Field]:
        lv = LEVELS.index(level)
        return [f for f in self.fields if f.required and LEVELS.index(f.level) <= lv]


def _flatten(items):
    for it in items:
        if isinstance(it, list):
            yield from _flatten(it)
        else:
            yield it


@lru_cache(maxsize=1)
def load_contract(path: str | None = None) -> dict[str, Table]:
    raw = yaml.safe_load(Path(path or CONTRACT_PATH).read_text())
    tables = {}
    for name, spec in raw["tables"].items():
        fields = [Field(**{k: v for k, v in f.items() if k in Field.__dataclass_fields__})
                  for f in _flatten(spec["fields"])]
        tables[name] = Table(name=name, level=spec["level"], primary_key=spec["primary_key"],
                             fields=fields, description=spec.get("description", ""))
    return tables


def contract_meta(path: str | None = None) -> dict:
    raw = yaml.safe_load(Path(path or CONTRACT_PATH).read_text())
    return {k: v for k, v in raw.items() if k != "tables"}


def validate_table(name: str, df: pd.DataFrame, level: str = "A") -> list[str]:
    """Return a list of contract violations (empty list = valid at `level`)."""
    t = load_contract()[name]
    issues = []
    for f in t.required_fields(level):
        if f.name not in df.columns:
            issues.append(f"{name}: missing required column '{f.name}' (level {f.level})")
            continue
        if df[f.name].isna().any():
            issues.append(f"{name}: nulls in required column '{f.name}'")
        if f.type == "enum" and f.values:
            bad = set(df[f.name].dropna().unique()) - set(f.values)
            if bad:
                issues.append(f"{name}: invalid values in '{f.name}': {sorted(map(str, bad))[:5]}")
    if "knowledge_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["knowledge_time"]):
        issues.append(f"{name}: knowledge_time is not a timestamp dtype")
    if "ingested_time" in df.columns and "knowledge_time" in df.columns:
        same = (df["ingested_time"] == df["knowledge_time"]).mean()
        if len(df) > 0 and same > 0.99:
            issues.append(f"{name}: knowledge_time equals ingested_time for {same:.0%} of rows "
                          "(suspect: ingestion time used as knowledge time)")
    pk = [c for c in t.primary_key if c in df.columns]
    if pk and len(pk) == len(t.primary_key) and df.duplicated(pk).any():
        issues.append(f"{name}: duplicate primary keys {t.primary_key}")
    return issues


def _present(tables: dict[str, pd.DataFrame], spec: str) -> bool:
    name, _, col = spec.partition(".")
    if name not in tables or len(tables[name]) == 0:
        return False
    return (not col) or (col in tables[name].columns and tables[name][col].notna().any())


def available_level(tables: dict[str, pd.DataFrame]) -> str | None:
    """Highest level L such that the requirements of all levels <= L are present."""
    best = None
    for lv in LEVELS:
        if all(any(_present(tables, alt) for alt in entry) for entry in LEVEL_TABLES[lv]):
            best = lv
        else:
            break
    return best
