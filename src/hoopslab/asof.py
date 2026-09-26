"""As-of (point-in-time) data access.

Every model and feature builder receives an `AsOfView`, never the raw store. A
view at time t exposes exactly the record versions whose knowledge_time <= t.
Where a record has several visible versions, it exposes the latest one.

Tables are sorted by knowledge_time once, so creating a view is a binary search
plus a prefix slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contract.schema import load_contract


def _record_key(table: str) -> list[str] | None:
    """Columns identifying one logical record whose versions supersede each other."""
    contract = load_contract()
    if table not in contract:
        return None
    pk = contract[table].primary_key
    if "record_version" not in pk:
        return None  # every row is its own record (snapshots, events, spells)
    return [c for c in pk if c not in ("record_version", "knowledge_time")]


def to_utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _as_utc_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def _ns(s: pd.Series) -> np.ndarray:
    return s.dt.as_unit("ns").astype("int64").to_numpy()


class AsOfStore:
    """Holds canonical tables and produces point-in-time views."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables: dict[str, pd.DataFrame] = {}
        self._kt: dict[str, np.ndarray] = {}
        for name, df in tables.items():
            if "knowledge_time" not in df.columns:
                raise ValueError(f"table {name} has no knowledge_time; refusing to serve it")
            if df["knowledge_time"].isna().any():
                raise ValueError(f"table {name} has null knowledge_time values")
            df = df.copy()
            df["knowledge_time"] = _as_utc_series(df["knowledge_time"])
            contract = load_contract()
            if name in contract:  # normalise every contract timestamp column to UTC
                for f in contract[name].fields:
                    if f.type == "timestamp" and f.name in df.columns and f.name != "knowledge_time":
                        df[f.name] = _as_utc_series(df[f.name])
            d = df.sort_values("knowledge_time", kind="stable").reset_index(drop=True)
            self._tables[name] = d
            self._kt[name] = _ns(d["knowledge_time"])

    @property
    def table_names(self) -> list[str]:
        return list(self._tables)

    def view(self, t) -> "AsOfView":
        return AsOfView(self, to_utc(t))

    def _visible(self, name: str, t: pd.Timestamp) -> pd.DataFrame:
        df = self._tables[name]
        idx = np.searchsorted(self._kt[name], t.as_unit("ns").value, side="right")
        out = df.iloc[:idx]
        key = _record_key(name)
        if key and len(out):
            out = out.drop_duplicates(subset=key, keep="last")  # sorted by knowledge_time
        return out

    # ---- tools used ONLY by the evaluation harness and leakage tests ----------

    def full_table(self, name: str) -> pd.DataFrame:
        """Unrestricted access, for scoring outcomes after the fact. Never pass to models."""
        return self._tables[name]

    def truncated(self, t) -> "AsOfStore":
        """A store in which every record not yet known at t has been physically deleted."""
        t = to_utc(t)
        return AsOfStore({n: self._tables[n][self._tables[n]["knowledge_time"] <= t] for n in self._tables})

    def perturbed_future(self, t, rng: np.random.Generator, protect: tuple[str, ...] = ()) -> "AsOfStore":
        """Shuffle numeric outcome columns among rows not yet known at t.

        A leak-free forecaster's output at time t is invariant to this.
        """
        t = to_utc(t)
        contract = load_contract()
        new = {}
        for n, df in self._tables.items():
            df = df.copy()
            fut = (df["knowledge_time"] > t).to_numpy()
            if fut.sum() > 1 and n in contract:
                for f in contract[n].fields:
                    if f.time_role == "outcome" and f.name in df.columns and f.name not in protect \
                            and pd.api.types.is_numeric_dtype(df[f.name]):
                        vals = df.loc[fut, f.name].to_numpy()
                        df.loc[fut, f.name] = rng.permutation(vals)
            new[n] = df
        return AsOfStore(new)


class AsOfView:
    """Point-in-time view. Deliberately exposes no path back to the raw store."""

    __slots__ = ("_AsOfView__store", "t", "accessed", "cache")

    def __init__(self, store: AsOfStore, t: pd.Timestamp):
        self.__store = store
        self.t = t
        self.accessed: set[str] = set()
        self.cache: dict = {}  # memo for derived as-of features (valid only for this t)

    def table(self, name: str) -> pd.DataFrame:
        self.accessed.add(name)
        if name not in self.__store.table_names:
            raise KeyError(f"table {name} not available in this dataset")
        return self.__store._visible(name, self.t)

    def has(self, name: str) -> bool:
        return name in self.__store.table_names
