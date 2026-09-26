"""Adapter exposing a synthetic World through the data contract."""

from __future__ import annotations

from ..synthetic.world import ScenarioConfig, World, generate_world
from .base import DataAdapter


class SyntheticAdapter(DataAdapter):
    name = "synthetic"

    def __init__(self, world: World | None = None, config: ScenarioConfig | None = None):
        self.world = world or generate_world(config or ScenarioConfig())

    def load_tables(self):
        return {k: v.copy() for k, v in self.world.tables.items()}

    def provenance(self):
        return {"scenario": self.world.config.name, "seed": self.world.config.seed,
                "knowledge_time": "exact by construction"}
