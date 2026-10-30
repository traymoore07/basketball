"""Smoke-test the corpus adapter (S1/S2 only: contract validation, T4, T6, determinism).

    python experiments/onboarding/smoke_adapter.py nba espn 2015,2016 C
"""
import hashlib
import sys
import time

import pandas as pd

from hoopslab.adapters.base import build_store
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter
from hoopslab.backtest.leakage import knowledge_time_audit, population_check
from hoopslab.backtest.runner import BacktestSpec, forecast_points, make_requests
from hoopslab.contract.schema import available_level, validate_table
from hoopslab.targets import player_outcomes

league, family, seasons, level = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")], sys.argv[4]
t0 = time.time()
ad = CorpusV102Adapter(league=league, family=family, seasons=seasons, level=level)
tables = ad.load_tables()
print(f"loaded in {time.time()-t0:.0f}s:", {k: len(v) for k, v in tables.items()})
print("available_level:", available_level(tables))
viol = {}
for name, df in tables.items():
    iss = validate_table(name, df, level)
    if iss:
        viol[name] = iss
        print("  VIOLATIONS", name, iss[:4])
print("T4:", knowledge_time_audit(tables))
store, report = build_store(ad, strict=False)
print("report level:", report["level"], "| issue tables:", list(report["issues"]))
print("excluded seasons:", ad._excluded_seasons, "| dropped:", report["provenance"]["rows_dropped"])
g = tables["GAME"]
mid = g[(g.record_version == 1)]["scheduled_tip"].sort_values().iloc[len(g) // 4]
spec = BacktestSpec(start=mid, end=mid + pd.Timedelta(days=2))
pts = list(forecast_points(store, spec))
print("forecast points in a 2-day window:", len(pts))
t, games = pts[0]
view = store.view(t)
req = make_requests(view, games)
print("requests at", t, ":", len(req), "players for", games["game_id"].nunique(), "games")
outc = player_outcomes(store.full_table("PLAYER_GAME").sort_values("knowledge_time").drop_duplicates(["game_id", "player_id"], keep="last"))
print("T6:", population_check(req, outc))
h1 = hashlib.sha256(pd.util.hash_pandas_object(tables["PLAYER_GAME"], index=False).values.tobytes()).hexdigest()
tables2 = CorpusV102Adapter(league=league, family=family, seasons=seasons, level=level).load_tables()
h2 = hashlib.sha256(pd.util.hash_pandas_object(tables2["PLAYER_GAME"], index=False).values.tobytes()).hexdigest()
print("deterministic PLAYER_GAME:", h1 == h2)
print("GAME versions:", g.record_version.value_counts().to_dict(), "| statuses:", g.status.value_counts().to_dict())
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(tables["ROSTER"].head(4).to_string())
for name in ("EVENT", "POSSESSION", "SUBSTITUTION", "LINEUP_STINT", "STARTING_LINEUP"):
    if name in tables:
        print(f"-- {name}"); print(tables[name].head(3).to_string())
