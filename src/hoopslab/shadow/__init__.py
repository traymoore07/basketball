"""Prospective shadow lane: package the frozen P1 L2 (mode A) so an external executor can forecast games
that have not happened yet, lock each forecast before tip-off, and attach outcomes afterwards.

Research lane never imports anything that reads live or 2023+ data; see docs/shadow/00_firewall.md.
"""
from .model import MODEL_VERSION, forecast, model_implementation_hash  # noqa: F401
from .package import LockStore, build_snapshot, canonical_json, lock_hash, make_forecast_package, verify_locked  # noqa: F401
from .scoring import Ledger, score_locked_forecast  # noqa: F401
