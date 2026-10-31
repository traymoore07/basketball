# 03. Shadow lane build report (2026-10-30)

*Mission: build the infrastructure that lets a separate process test a frozen P1 model on games that
have not happened yet, while the research agent stays blind to 2023+ outcomes. Nothing here is model
research; no 2023+ data was read; no live source was contacted.*

## 0. Fixed answer block

```
SHADOW MODEL:                   L2-A-SHADOW-v0.1 = P1 rung L2, hoopslab.models.baselines.PlayerMean(half_life=5.0),
                                run unchanged through the same AsOfStore; mode A only; targets MIN, PTS, REB, AST,
                                FG3M, TOV, PRA; no new feature, no retuning, no calibration change, no grid change.
                                Card: shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json (card sha256
                                9850040e05a54476fa5196bf2fc6cc68d274c19dc0ddfbf37eeaedc2bcc5aee1)
MODEL HASH:                     caf0cb4407b109128f3d91f8ef631aacc531cafd72365208e74c3894507c65e8 (combined over
                                baselines.py 4fff216d..., features.py 9049e9e2..., targets.py 9e29fe7d...,
                                forecast.py 653d8571..., asof.py d3b10e85..., shadow/model.py f828654f...)
FORECAST MODE:                  A - CONDITIONAL ON APPEARANCE
LIVE INPUT CONTRACT:            PASS (docs/shadow/01: GAME / PLAYER_GAME / ELIGIBLE tables with knowledge_time; availability
                                states CONFIRMED_ACTIVE / EXPECTED_ACTIVE / QUESTIONABLE_UNKNOWN / OUT; provider-agnostic
                                SnapshotProvider / file directory; validated by hoopslab.shadow.interface.validate_tables)
T-60 SNAPSHOT:                  PASS (cutoff = scheduled tip - 60 min; rows with knowledge_time > cutoff dropped and
                                counted; status_time > cutoff refused; forecast refused after tip; replay: leaky
                                provider with post-cutoff games and permuted future box rows -> identical
                                distributions and data_snapshot_id; physically truncated store -> identical lock hash)
IMMUTABLE LOCK:                 PASS (canonical JSON, SHA-256; LockStore writes once with mode "x"; overwrite refused
                                on every replayed game; a correction becomes v2 with a new timestamp and hash while
                                v1's bytes are unchanged; tampering with one pmf value is detected)
SCORING PIPELINE:               PASS (score_locked_forecast() runs only on a verified locked package plus a final box
                                score; frozen metrics CRPS / log / RPS / PIT / MAE / RMSE / twCRPS per unit; populations
                                STATUS_CONFIRMED (primary) / STATUS_EXPECTED / UNKNOWN; NOT_APPEARED and OUT counted,
                                never scored; append-only ledger with (forecast_id, revision) uniqueness; cumulative
                                table by model version, target, population, month)
REPLAY TEST:                    PASS (19 development-period games on 2010-01-15 and 2015-03-10: 535 forecasts
                                issued, max |pmf difference| vs the P1 run-2 L2_recent_hl5 forecasts = 0.0 on every
                                unit; 2,793 ledger rows; provider refuses lockbox and holdout cutoffs;
                                results/shadow/replay_test.{json,md}, lane results/shadow/replay_lane/)
RESEARCH / SHADOW FIREWALL:     PASS (docs/shadow/00: separate lanes and locations; no repository code reads
                                SHADOW_HOME; outcomes reach research only by a logged exposure row; the replay
                                provider cannot serve cutoffs after 2020-10-31; model upgrades are new versions)
CURRENT DATA ACCESSED BY RESEARCH AGENT: NO
READY FOR EXTERNAL LIVE EXECUTION: YES, for the mode-A candidate under the documented policy: the executor supplies
                                the three tables from a source of its choice, runs the CLI, locks before tip, scores
                                after the final box, and shares only the cumulative table. Known limits carried
                                into the policy, not fixed here: no appearance probability (by design), league
                                pmf fallback for players without history, results depend on the history depth
                                supplied (recorded in data_snapshot_id).
EXTERNAL EXECUTOR INSTRUCTIONS: docs/shadow/02_executor_instructions.md (summary in section 5 below)
```

## 1. What was built

| component | file | role |
|---|---|---|
| input contract | `src/hoopslab/shadow/interface.py` | table schemas, availability states, `SnapshotProvider` / `OutcomeProvider` protocols, validation |
| model wrapper | `src/hoopslab/shadow/model.py` | loads the snapshot into `AsOfStore`, takes the view at the cutoff, fits and predicts with the frozen `PlayerMean(5)`, discards p_appear, derived quantities from the pmf, implementation hash |
| packaging | `src/hoopslab/shadow/package.py` | `build_snapshot` (cutoff enforcement, manifest, `data_snapshot_id`), `make_forecast_package`, `canonical_json`, `lock_hash`, `LockStore` (append-only, versioned) |
| scoring | `src/hoopslab/shadow/scoring.py` | `score_locked_forecast`, `Ledger` (append-only JSONL + events + cumulative) |
| replay | `src/hoopslab/shadow/replay.py` | `ResearchCorpusProvider` (development window only), `LeakyProvider` |
| executor CLI | `src/hoopslab/shadow/cli.py` | `forecast`, `score`, `void`, `event`, `report`, `verify`; file-based provider |
| model card | `src/hoopslab/shadow/card.py` -> `shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json` | everything an executor needs, with hashes |
| tests | `tests/test_shadow.py` (5) | identity with the frozen class, post-cutoff drop + invariance, immutable lock, scoring + ledger, CLI end to end |
| replay | `experiments/shadow/replay_test.py` | the full pipeline on development dates against P1 |
| docs | `docs/shadow/00_firewall.md`, `01_input_contract.md`, `02_executor_instructions.md` | firewall, contract + edge cases + scoring policy, executor steps |

## 2. Forecast package (per game)

forecast_id, forecast_version, model_version, forecast_mode, interpretation ("P(stat | the player
appears)"), game metadata, forecast_created_at, scheduled_tip, information_cutoff, data_snapshot_id,
input_data_manifest (row counts, table hashes, latest knowledge_time, rows dropped after the cutoff),
model block (half-life, mode, targets, implementation hash per file and combined, pmf construction,
fitted dispersion), availability_counts, players[] with availability_state, status_source,
status_time, history_appearances, history_fallback and per-target full pmf, mean, median, p10/p25/
p50/p75/p90 and P(X >= edge) at the frozen RPS edges, environment, lock_hash. Canonical JSON (sorted
keys, no whitespace, shortest float repr) hashed with SHA-256.

## 3. Edge-case behaviour

Fixed in docs/shadow/01 section 3: missing history (league pmf fallback, flagged), new players,
traded players (history follows player_id), postponed games and tip changes (lock tied to its tip;
after-lock changes void the package and a new one is issued), questionable players (UNKNOWN
population), late scratches (no mode-A score, counted), incomplete feeds (degraded event, no
back-fill), duplicates (refused), cancellations (voided, kept), overtime (full-game targets),
stat corrections (new score revision, original kept). Future information cannot enter: knowledge_time
filter, status_time check, snapshot identity, replay evidence.

## 4. Replay evidence (development dates only)

| check | result |
|---|---|
| games / forecasts issued / ledger rows | 19 / 535 / 2,793 |
| identity with P1 run-2 L2_recent_hl5 (max abs pmf difference, all units, all targets) | 0.0 |
| leaky provider (post-cutoff games + permuted future box rows): same distributions, same data_snapshot_id | yes, 19/19 |
| truncated store: same lock hash | yes, 19/19 |
| overwrite refused; v1 unchanged after v2; tamper detected | yes, 19/19 |
| forecast created before tip; information cutoff = T-60 | yes, 19/19 |
| provider refuses lockbox (2021-11-01) and holdout (2024-01-01) cutoffs | yes |

The historical replay has no availability feed, so every unit is QUESTIONABLE_UNKNOWN and the primary
population is empty by design; late scratches (NOT_APPEARED) are 3 to 11 per game, which is exactly
the population effect the status-based policy exists to isolate.

## 5. Executor instruction set (full text: docs/shadow/02)

1. Clone the repository at the pinned commit (the one carrying the model card), install, set
   `SHADOW_HOME` outside the repository, run `python -m hoopslab.shadow.card` and check the combined
   hash against the card, run `pytest tests/test_shadow.py`.
2. Each forecast day, from any data source, write a snapshot directory with `game`, `player_game`
   (completed games only, league-wide history, de-duplicated, no NULL stats, knowledge_time on every
   row) and `eligible` (availability state, source and time per player).
3. Between T-60 and tip: `python -m hoopslab.shadow.cli forecast --snapshot DIR --game-id ID`.
   It builds the snapshot at the game's cutoff, runs L2-A-SHADOW-v0.1, locks the package and prints
   the forecast_id and lock hash. Never re-run to update; corrections are new versions.
4. If the tip changes or the game is postponed or cancelled after the lock: `cli void --reason ...`,
   then a fresh package at the new tip.
5. Wait for the final box score.
6. Write the box score as a PLAYER_GAME file and run `cli score --game-id ID --forecast-id FID
   --box FILE --result-time ISO` (revision 2 for an official correction).
7. Keep everything in `SHADOW_HOME`; weekly `cli verify` and `cli report`.
8. Hand only the cumulative table to Tray. Nothing from `SHADOW_HOME` reaches Research Claude unless
   Tray logs an exposure row.

## 6. Governance

C19 (changelog), exposure row 12 (replay on development dates; outcomes already exposed in rows 8-10),
no frozen file changed, no attested file changed. Tests: `pytest -q -m "not slow"` and
`python -m hoopslab.registry verify` after the build (recorded in the commit message).

Stop. No L5, no live outcomes, no 2023+ data.
