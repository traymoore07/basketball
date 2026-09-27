"""Adapter interface between an external dataset and the canonical data contract.

The experiment system never reads raw storage directly. An adapter's only job is to
return canonical tables (see contract/data_contract.yaml) with honest knowledge_time
values. Everything downstream (as-of views, models, backtests) is adapter-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..asof import AsOfStore
from ..contract.schema import available_level, load_contract, validate_table


class DataAdapter(ABC):
    name: str = "adapter"

    @abstractmethod
    def load_tables(self) -> dict[str, pd.DataFrame]:
        """Return {canonical_table_name: DataFrame} conforming to the contract."""

    def provenance(self) -> dict:
        """Free-form description of sources and how knowledge_time was determined."""
        return {}


def build_store(adapter: DataAdapter, strict: bool = True):
    """Validate adapter output against the contract and wrap it in an AsOfStore.

    Returns (store, report). With strict=True, any contract violation raises.
    """
    tables = adapter.load_tables()
    contract = load_contract()
    level = available_level(tables)
    issues = {}
    for name, df in tables.items():
        if name not in contract:
            continue
        v = validate_table(name, df, level or "A")
        if v:
            issues[name] = v
    report = {"adapter": adapter.name, "level": level, "issues": issues,
              "rows": {n: len(df) for n, df in tables.items()},
              "unknown_tables": sorted(set(tables) - set(contract)),
              "provenance": adapter.provenance()}
    if strict and issues:
        raise ValueError(f"contract violations: {issues}")
    served = {n: df for n, df in tables.items() if n in contract}
    return AsOfStore(served), report
