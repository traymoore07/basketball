# 01. Pregame-context input contract (L1), version `pregame-context/1`

*Provider-agnostic. Code of record: `src/hoopslab/pregame/contract.py`. Fixed 2026-11-02 before any live
snapshot exists. Every timestamp UTC. One snapshot directory per information cutoff.*

## 1. Snapshot directory

```
CONTEXT_HOME/live_inputs/<YYYY-MM-DD>T<HHMM>Z/
  manifest.json          contract_version, cutoff, provider, retrieved_at, games[], note
  availability.csv|parquet   required
  roster.csv|parquet         required
  lineup.*                   optional (absence recorded in the validation report)
  recent_role.*              optional
  team_absences.*            optional
```

The information cutoff is the T-60 cutoff of the game (or of a slate with one tip) as in docs/shadow/01.
Every `published_at`, `retrieved_at`, `as_of`, `added_at`, `removed_at` must be <= cutoff; a later row fails
validation (nothing is dropped silently). The validation report records row counts, SHA-256 per table and
`snapshot_id` = SHA-256 over the cutoff and the table hashes.

## 2. Tables

### availability (one row per game x player the executor has looked at)

| column | type | meaning |
|---|---|---|
| game_id, player_id, team_id | string | ids consistent with the shadow GAME / PLAYER_GAME tables |
| availability | enum | ACTIVE / PROBABLE / QUESTIONABLE / DOUBTFUL / OUT / UNKNOWN |
| source | string | free text naming the source (official report, team PR, beat writer name, feed, vendor) |
| source_kind | enum | official / starters_feed / beat_writer / vendor / inferred / none |
| official | bool | true only for a public official confirmation (league / team report, active list, confirmed starters); an `inferred` row can never be official |
| published_at | timestamp | when the source published the state (<= cutoff) |
| retrieved_at | timestamp | when the executor's data path first held it (<= cutoff) |
| confidence | string/float, optional | the source's own qualifier, verbatim where available |
| reason_category | string, optional | injury / illness / rest / personal / suspension / conditioning / g_league / other |
| note | string, optional | body part, restriction, free text |

Rules: unique per (game, player); an official row cannot be UNKNOWN; UNKNOWN is a recorded state, never a
default for ACTIVE; no probability of appearance is ever written.

### roster (current roster at the cutoff)

| column | meaning |
|---|---|
| team_id, player_id | membership at `as_of` |
| roster_status | standard / two_way / exhibit_10 / hardship / unknown |
| assignment | nba / g_league / unknown (where the player is assigned at the cutoff) |
| as_of | the time the roster was read (<= cutoff) |
| added_at, removed_at | optional transaction times for recent additions / removals (<= cutoff) |

### lineup (announced or expected starters)

| column | meaning |
|---|---|
| lineup_status | CONFIRMED_STARTER / EXPECTED_STARTER / BENCH / UNKNOWN |
| source, source_kind, published_at, retrieved_at | as availability |

At most five CONFIRMED_STARTER rows per (game, team). Expected starters are the executor's reading of a
source, never an inference from the model.

### recent_role (the player's role as the executor saw it before the cutoff)

| column | meaning |
|---|---|
| as_of, n_games | the window used (last N completed games) |
| minutes_mean, minutes_last, starts | over the window; empty when n_games = 0 |
| sub_role | starter / first_sub / rotation / deep / unknown |
| rotation_position | integer order in the rotation, or empty |

### team_absences (what has been removed from the rotation)

| column | meaning |
|---|---|
| game_id, team_id, player_id | an OUT or DOUBTFUL player of that team (must have an availability row) |
| availability | OUT / DOUBTFUL |
| role_removed | the R2 role taxonomy (PRIMARY_CREATOR, SECONDARY_CREATOR, HIGH_USAGE_WING, STARTING_CENTER, STARTING_GUARD, STARTING_OTHER, ROTATION_BENCH, DEEP_BENCH) or `unknown` |
| source, published_at | as availability |

## 3. Mapping to the frozen baseline (one way; `to_eligible_v01`)

| L1 availability | official | v0.1 availability_state |
|---|---|---|
| ACTIVE | true | CONFIRMED_ACTIVE |
| ACTIVE | false | EXPECTED_ACTIVE |
| PROBABLE | any | EXPECTED_ACTIVE |
| QUESTIONABLE / DOUBTFUL / UNKNOWN | any | QUESTIONABLE_UNKNOWN |
| OUT | any | OUT |

`status_source` = `source`; `status_time` = `published_at`. The mapping is deterministic and recorded in the
shadow snapshot directory as `pregame_manifest.json` (snapshot id, mapping name). The baseline's own rules
stay: OUT players are listed and receive no distribution; no state becomes an appearance probability.

## 4. Minutes-distribution output contract (R2 -> X1)

`MinutesDistribution(mean, p10, p25, p50, p75, p90, pmf over 0..70)`; quantiles monotone, within 0..70; the
pmf, when present, has 71 non-negative values summing to 1. Challenger forecasts carry this distribution per
player for BASE ROLE and TONIGHT ROLE; the stat distributions are built from it (projected minutes x
per-minute rate), never from a single minutes number.

## 5. Not in this contract

No appearance probability, no outcome, no box score, no score ledger. Those belong to the shadow / context
result stores and never to an input snapshot.
