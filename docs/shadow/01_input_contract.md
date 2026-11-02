# 01. Live input contract, edge-case behaviour and prospective scoring policy

*For L2-A-SHADOW-v0.1 (`hoopslab.shadow`). Provider-agnostic: the executor supplies tables; the
repository depends on no website or API. Code of record: `src/hoopslab/shadow/interface.py`.*

## 1. What the executor must supply at forecast time

One snapshot per information cutoff (the T-60 cutoff of the game, or of a slate of games with the same
tip). Three tables, UTC timestamps, as CSV or parquet files in a snapshot directory, or as a
`SnapshotProvider` object.

### GAME (one row per game known at the cutoff)

| column | type | meaning |
|---|---|---|
| game_id | string | stable, provider-independent id (recommended `nba:<source>:<source_game_id>`) |
| league | string | `nba` |
| season_id | string | `nba-<end year>` |
| phase | string | regular / postseason / play_in / in_season_tournament / preseason |
| scheduled_tip | timestamp | tip-off as scheduled at the cutoff |
| home_team_id, away_team_id | string | team ids, consistent with PLAYER_GAME / ELIGIBLE |
| status | string | `scheduled` for the forecast games; `final` for history games |
| knowledge_time | timestamp | when the executor's data path first held this row (must be <= cutoff) |

Must contain the forecast games and every game referenced by PLAYER_GAME (the history join needs
each game's tip to order appearances).

### PLAYER_GAME (completed games only; the appearance history)

| column | type | meaning |
|---|---|---|
| game_id, player_id, team_id | string | one row per (game, player); the team the player played for |
| seconds | int | seconds played; 0 = dressed, did not play |
| pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf | int | the full line, never NULL (a game with an unknown line is left out and counted, never imputed) |
| knowledge_time | timestamp | when the final line became available to the executor (must be <= cutoff to enter) |

History policy: league-wide, all completed games from the executor's history start to the cutoff. The
P1 reference history starts 1997-10-01; the league pmf fallback and the negative-binomial dispersion
are fitted on everything supplied, so the history start is part of the snapshot identity
(`data_snapshot_id`). Supply at least the current and the three previous seasons; more is better and
closer to the P1 configuration.

### ELIGIBLE (who gets a conditional forecast)

| column | type | meaning |
|---|---|---|
| game_id, player_id, team_id | string | the player's current team for this game |
| availability_state | enum | CONFIRMED_ACTIVE / EXPECTED_ACTIVE / QUESTIONABLE_UNKNOWN / OUT |
| status_source | string | where the state came from (official report, beat writer, starters feed, none) |
| status_time | timestamp | when the state was published (must be <= cutoff) |

The model does not predict whether a player appears. The executor decides eligibility and records the
state; OUT players may be listed and receive no distribution. No state is ever converted into an
appearance probability.

### Builder guarantees (`hoopslab.shadow.package.build_snapshot`)

- Rows with knowledge_time > cutoff are dropped and counted in the manifest (`rows_dropped_after_cutoff`).
- ELIGIBLE rows with status_time > cutoff abort the snapshot.
- Contract violations (missing columns, duplicates, NULL stats, unknown states, dangling game ids) abort.
- The manifest records per-table row counts, SHA-256 of the sorted rows, the latest knowledge_time and
  the earliest history tip; `data_snapshot_id` = SHA-256 over the three table hashes and the cutoff.

## 2. What comes out

One package per game (`shadow-forecast-package/1`), locked by `LockStore.lock`:

- `forecast_id` = `<model_version>__<game_id>__v<n>`, `forecast_version`, `forecast_created_at`,
  `scheduled_tip`, `information_cutoff` (= tip - 60 min, or the creation time if an early forecast was
  explicitly allowed), `model_version`, `data_snapshot_id`, `input_data_manifest`;
- `game` metadata; `availability_counts`; `model` block with half-life, mode, targets, the
  implementation hash (per file and combined), the pmf construction and the fitted dispersion;
- `players[]`: player_id, team_id, availability_state, status_source, status_time,
  history_appearances, history_fallback (`player` or `league_pmf`), and `distributions` per target:
  full `pmf` over `0..k_max`, `mean`, `median`, `p10`, `p25`, `p50`, `p75`, `p90`, and
  `threshold_probabilities` P(X >= edge) at the frozen RPS bin edges (computed from the pmf, nothing else);
- `environment` (python/numpy/pandas/scipy, platform, hoopslab commit); `lock_hash` = SHA-256 of the
  canonical JSON of everything else (sorted keys, no whitespace, floats as shortest round-trip repr).

Interpretation printed in every package: **P(stat | the player appears)**. Nothing is an
unconditional probability.

Rules: created_at must be before the tip; created_at must be at or after the cutoff unless
`allow_created_before_cutoff` is passed (then the information cutoff is the creation time, never
later). A locked file is never rewritten; `lock()` refuses an existing path; corrections are new
versions with new timestamps and hashes, and the earlier file stays.

## 3. Behaviour in edge cases (fixed before any live game)

| situation | behaviour |
|---|---|
| missing player history (no visible appearance) | the frozen fallback: the league pmf of each target over all visible appearances; `history_fallback = "league_pmf"`, `history_appearances = 0`; scored like any other unit but reportable separately |
| new player (rookie, call-up) | same as above until his first completed game enters the history |
| traded / signed player | history follows `player_id` across teams (the model has no team feature); the team in ELIGIBLE is whatever the executor knows at the cutoff; a wrong team is recorded, not corrected after the fact |
| postponed game | a locked forecast is tied to (game_id, scheduled_tip, information_cutoff). If the postponement is known before the cutoff, the executor supplies the new tip and the forecast is made for it. If it is known after the lock, the locked package is marked VOID_POSTPONED in the ledger events and never scored; a new package is issued at the new tip (new version, new cutoff) |
| changed tip time | same rule as postponement: a lock belongs to the tip it was made for; a tip change after the lock voids it; before the lock, the new tip defines the cutoff |
| questionable player | listed QUESTIONABLE_UNKNOWN; gets a distribution; scored only in the UNKNOWN population |
| late scratch (confirmed/expected active at the cutoff, did not play) | the forecast stands (locked); the unit has no mode-A score; counted as NOT_APPEARED in the ledger event; the scratch rate per state is reported |
| OUT at the cutoff | listed, no distribution; if he plays anyway, counted as OUT_but_appeared, never scored |
| incomplete feed (history or roster partially missing at the cutoff) | the snapshot is built from what is known; the manifest records row counts and the latest knowledge_time; the executor marks the package `degraded` in the ledger events with the reason; nothing is back-filled into that snapshot later |
| duplicate events (two rows for one player-game) | the executor must de-duplicate deterministically before supplying (P1 rule: keep the row with the most seconds); the builder refuses duplicates |
| game cancellation | locked package marked VOID_CANCELLED in the events ledger; never scored; never deleted |
| overtime | targets are full-game totals including overtime (frozen definition); no adjustment |
| corrections to official stats after scoring | the original score rows stay; the corrected box score is scored as `score_revision = 2` with its own result_time; cumulative metrics use the latest revision per forecast |
| information after the cutoff | cannot enter: the builder drops rows by knowledge_time, the provider is asked for the cutoff, and the package carries the data_snapshot_id; the replay test shows a leaky provider yields identical distributions |

Never silently substitute future information: every exception above leaves the locked package as it
was and records an event.

## 4. Prospective scoring policy (fixed before any live game is scored)

A conditional forecast is issued for every CONFIRMED_ACTIVE, EXPECTED_ACTIVE and
QUESTIONABLE_UNKNOWN player; evaluation distinguishes the information state at the cutoff:

| population | definition | role |
|---|---|---|
| STATUS_CONFIRMED | CONFIRMED_ACTIVE at the cutoff and appeared (seconds > 0) | **primary** prospective score |
| STATUS_EXPECTED | EXPECTED_ACTIVE and appeared | secondary |
| UNKNOWN | QUESTIONABLE_UNKNOWN and appeared | stored; excluded from the primary score unless a later frozen protocol admits it |
| NOT_APPEARED | forecast issued, player did not appear | no mode-A score; counted (scratch rate by state) |
| OUT / OUT_but_appeared | no distribution issued | counted only |

Metrics per unit and target (frozen `hoopslab.metrics.scoring`): CRPS, log score (bits), RPS on the
frozen bins, randomised PIT (seed 0 per scoring call), MAE of the median, squared error of the mean,
upper-tail twCRPS (PTS 25, PRA 40, REB 10, AST 8). Cumulative metrics by model version, target,
population and month: n, mean CRPS, log, RPS, MAE, RMSE, twCRPS, 80% coverage, PIT dispersion, PIT
KS, bias. This never recreates the mode-U problem: no appearance probability is inferred, and the
population a unit is scored in is the state that was recorded before the game.

Comparisons between model versions use only games both versions forecast (same game_id, same
population), on the cumulative table; an older version keeps forecasting until retired.

## 5. Ledger (append-only, `SHADOW_HOME/ledger/`)

`scores.jsonl` rows: forecast_id, game_id, model_version, forecast_time, information_cutoff,
lock_hash, result_time, score_revision, player_id, team_id, availability_state, population, target, y,
crps, log, rps, pit, ae_median, se_mean, tw_crps, mean_fc, p10, p90, in_central_80.
`events.jsonl`: scored / voided / degraded events with counts. `cumulative.csv`: derived table.
A (forecast_id, score_revision) pair can be appended once.
