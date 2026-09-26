"""Generic file adapter for the real dataset (placeholder mapping, no storage assumptions).

Configure with a mapping from canonical table -> source file and column renames. The
external collection project can publish parquet/CSV in any layout; only this mapping
changes. Example (YAML or dict):

    PLAYER_GAME:
      path: box/player_games.parquet
      rename: {PLAYER_ID: player_id, GAME_ID: game_id, MIN_SEC: seconds}
      timestamps: [knowledge_time]
      constants: {source: nba_stats, knowledge_time_quality: bounded}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataAdapter


class FileAdapter(DataAdapter):
    name = "files"

    def __init__(self, root: str | Path, mapping: dict, provenance: dict | None = None):
        self.root = Path(root)
        self.mapping = mapping
        self._prov = provenance or {}

    def _read(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in (".csv", ".gz"):
            return pd.read_csv(path)
        raise ValueError(f"unsupported file type: {path}")

    def load_tables(self):
        out = {}
        for table, spec in self.mapping.items():
            df = self._read(self.root / spec["path"]).rename(columns=spec.get("rename", {}))
            for c in spec.get("timestamps", []):
                df[c] = pd.to_datetime(df[c], utc=True)
            for c, v in spec.get("constants", {}).items():
                df[c] = v
            out[table] = df
        return out

    def provenance(self):
        return self._prov
