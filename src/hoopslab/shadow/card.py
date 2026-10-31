"""Model card for L2-A-SHADOW-v0.1: python -m hoopslab.shadow.card [out_path]

Writes shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json: everything an executor needs to run the model without
editing code, with the implementation hash and the card's own SHA-256 identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from ..targets import BINS, PLAYER_TARGETS
from . import model as M
from .interface import ELIGIBLE_COLUMNS, GAME_COLUMNS, HISTORY_POLICY, PLAYER_GAME_COLUMNS, AvailabilityState
from .package import PACKAGE_SCHEMA_VERSION, environment_metadata


def build_card() -> dict:
    card = {
        "model_version": M.MODEL_VERSION,
        "status": "first prospective candidate; not the final basketball model",
        "lineage": "P1 ladder rung L2 = hoopslab.models.baselines.PlayerMean(half_life=5.0), selected on prior seasons in every "
                   "development season (docs/p1/01_ladder_report.md section 3.4); mode A results: core CRPS skill vs L0 +27.15%",
        "forecast_mode": "A - CONDITIONAL ON APPEARANCE: every distribution is P(stat | the player appears); no appearance "
                         "probability is produced or implied",
        "half_life_appearances": M.HALF_LIFE_APPEARANCES,
        "forecast_time": f"scheduled tip - {M.LEAD_MINUTES} min (information cutoff); forecast created at or after the cutoff and before the tip",
        "targets": {t: {"k_max": PLAYER_TARGETS[t][0], "definition": {
            "MIN": "round(seconds / 60), full game incl. overtime", "PTS": "points", "REB": "oreb + dreb", "AST": "assists",
            "FG3M": "three-point field goals made", "TOV": "turnovers", "PRA": "pts + oreb + dreb + ast"}[t],
            "rps_bins": BINS.get(t)} for t in M.TARGETS},
        "pmf_construction": M.PMF_CONSTRUCTION,
        "fitting_at_forecast_time": "refit on every forecast from the supplied history: per-player exponentially weighted means over "
                                    "appearances (seconds > 0), league negative-binomial dispersion per target, league pmf fallback",
        "required_inputs": {"GAME": GAME_COLUMNS, "PLAYER_GAME": PLAYER_GAME_COLUMNS, "ELIGIBLE": ELIGIBLE_COLUMNS,
                            "availability_states": [s.value for s in AvailabilityState], "history_policy": HISTORY_POLICY,
                            "contract": "docs/shadow/01_input_contract.md"},
        "output_schema": {"package": PACKAGE_SCHEMA_VERSION, "fields": [
            "schema", "forecast_id", "forecast_version", "model_version", "forecast_mode", "interpretation", "game{game_id, league, "
            "season_id, phase, scheduled_tip, home_team_id, away_team_id, status_at_cutoff}", "forecast_created_at", "scheduled_tip",
            "information_cutoff", "data_snapshot_id", "input_data_manifest", "model{version, half_life_appearances, mode, targets, "
            "implementation_hash, pmf_construction, dispersion_r_fitted, league_appearances_in_history}", "availability_counts",
            "players[{player_id, team_id, availability_state, status_source, status_time, history_appearances, history_fallback, "
            "distributions{target: {pmf, support, mean, median, p10, p25, p50, p75, p90, threshold_probabilities}}}]",
            "environment", "lock_hash"], "lock": "SHA-256 of the canonical JSON (sorted keys, no whitespace) of every field except lock_hash"},
        "implementation_hash": M.model_implementation_hash(),
        "environment_at_card_build": environment_metadata(),
        "governance": {"research_firewall": "docs/shadow/00_firewall.md", "scoring_policy": "docs/shadow/01_input_contract.md section 4",
                       "executor_instructions": "docs/shadow/02_executor_instructions.md",
                       "no_retuning": "half-life, pmf shapes, dispersion fit and targets are frozen; a change is a new model version"},
    }
    card["card_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in card.items() if k != "card_sha256"}, sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
    return card


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(M.repo_root(), "shadow", f"MODEL_CARD_{M.MODEL_VERSION}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = build_card()
    json.dump(card, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", out, "card_sha256", card["card_sha256"], "implementation", card["implementation_hash"]["combined"])


if __name__ == "__main__":
    main()
