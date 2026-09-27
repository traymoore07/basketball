# 05. Data contract: the information interface the experiments consume

*Canonical, machine-readable version: `src/hoopslab/contract/data_contract.yaml` (validated by `hoopslab.contract.schema`). This page explains it.*

The external engineering effort that is collecting NBA, WNBA, and NCAA data can store raw data however it likes. The experiments depend only on this contract. A thin **adapter** (`src/hoopslab/adapters/`) maps the external storage onto canonical tables. Nothing downstream knows how the data was collected.

## 1. Two clocks: event time vs knowledge time

Every record carries two different times:

| Field | Meaning | Example |
|---|---|---|
| `event_time` (table-specific name) | When the thing happened, or the moment it applies to | An injury-report entry *for* the 7:00 PM game |
| `knowledge_time` | The earliest UTC instant at which **this version** of the record was publicly available to an outside forecaster | The report was published at 2:00 PM, so knowledge_time = 14:00 local, converted to UTC |

A forecast issued at time *t* may use a record **only if `knowledge_time <= t`**. The as-of engine (`hoopslab.asof.AsOfStore`) enforces this. It is the only way models and features can reach data.

Three further rules:

- **`ingested_time` is not knowledge time.** A collector that backfilled 2015 box scores in 2026 has `ingested_time` = 2026, but the box score was known in 2015. Conversely, a scraped injury report without a publication timestamp does *not* become known at its scrape time. It needs a provenance rule. The validator flags tables where knowledge_time equals ingested_time for more than 99% of rows.
- **Versions, not overwrites.** Stat corrections, injury-status updates, reschedules, bio corrections, and roster-spell endings are *appended* as new versions of the same record key, each with its own knowledge time. For each record key, the as-of view exposes the latest version with knowledge_time ≤ t.
- **Knowledge-time quality is explicit.** Each row has `knowledge_time_quality ∈ {exact, bounded, imputed_conservative, unknown}`:
  - `bounded`: knowledge_time holds the *upper* bound of the interval.
  - `imputed_conservative`: imputed by a documented rule that errs late (below).
  - `unknown`: the pre-registered fallback rules apply.

### Pre-registered fallback rules (conservative: when in doubt, later)

| Situation | Rule |
|---|---|
| Box score / event, time unknown | knowledge_time := scheduled tip + 4 h |
| Injury report, date known but hour unknown | knowledge_time := scheduled tip, so it is **not usable pre-game** |
| Transaction, date known but hour unknown | 23:59:59 local on that date |
| Roster membership known only per season | not usable pre-game until the player's first appearance for that team |
| Market snapshot, time unknown | excluded |

**Sensitivity analysis (mandatory):** every primary result is re-run with *optimistic* imputation (earliest plausible time). If the decision changes, the result is flagged **LEAKAGE-SENSITIVE** in the scorecard.

### Worked example

```
INJURY_REPORT rows for game G (tip 2031-11-03 23:30 UTC), player P:
  report_id  status        knowledge_time          knowledge_time_quality
  R1         questionable  2031-11-03 18:30 UTC    exact
  R2         out           2031-11-03 23:20 UTC    exact      <- late scratch

Forecast at T-60 (22:30 UTC) sees: questionable   (R2 invisible)
Forecast at T-5  (23:25 UTC) sees: out
```

A model calibrated on the *final* status of past games ("out") would learn that its pre-game inputs are more informative than they really are. That subtle leak (L-B6) is prevented by calibrating on the status visible at the same forecast offset. `MinutesRate._status_rates` and `PossessionSimulator._fit_rotation` do this, and the synthetic world contains late scratches specifically to exercise it.

## 2. Canonical tables

For the full field list with `required`, `level`, and `time_role` per field, see the YAML. `time_role ∈ {key, event_time, knowledge, pre, outcome, meta}`:

- **pre:** usable pre-game once visible.
- **outcome:** only after the game's knowledge_time, and never as a feature of its own game.

| Table | Grain | Level | Knowledge time is… |
|---|---|---|---|
| RULESET | league × period of validity | A | rulebook publication |
| SEASON | league × season × phase | A | calendar publication |
| VENUE | venue (versioned) | A | – |
| TEAM | team identity spell | A | – |
| PLAYER | person × version | A | when the bio fact was published (bio corrections are versions) |
| PERSON_LINK | cross-league identity pair | A | when the link *could have been made* (L-F3) |
| COACH_TENURE | team × coach × role spell | C | announcement |
| ROSTER | membership spell × version | A | transaction announcement (not effective date) |
| OFFICIAL_ASSIGNMENT | game × official | C | assignment announcement (typically game morning) |
| GAME | game × version | A | schedule publication (v1); final (vN) |
| TEAM_GAME | game × team × version | A | final + stat-correction versions |
| PLAYER_GAME | game × **rostered** player × version | A | final. **Includes DNPs and inactives** |
| INJURY_REPORT | report snapshot × game × player | A | publication of that snapshot |
| STARTING_LINEUP | game × team × player | C | announcement (≈ T-30) |
| EVENT | play-by-play event | B | game final (pre-game use); event wall-clock (live use) |
| POSSESSION | possession segment | B (lineups: C) | as EVENT. Derived by hoopslab's parser or supplied |
| LINEUP_STINT | game × team × stint | C | as EVENT |
| SUBSTITUTION | substitution event | C | as EVENT |
| SHOT | field-goal attempt (+ tracking context) | B (D fields) | as EVENT |
| MATCHUP | game × offensive player × defender | D | game final |
| TRACKING_FRAME | game × frame × entity | D | game final (or provider delivery time) |
| MARKET_SNAPSHOT | book × market × snapshot | A (optional) | the snapshot time. No `is_closing` flag stored |
| PROPRIETARY_LOAD | player × date × metric | E | when the team would have known it |

`SCHEDULE_CONTEXT` (rest days, back-to-backs, travel distance, time-zone shifts, altitude) is **derived** by hoopslab from GAME + VENUE, as of the forecast time, so that a schedule revised later cannot leak (L-C4).

## 3. Levels: what each experiment needs at minimum

| Level | Adds | Enables |
|---|---|---|
| **A** | RULESET, GAME, TEAM_GAME, PLAYER_GAME (all rostered), ROSTER, PLAYER; *strongly recommended*: INJURY_REPORT with snapshot times | Ladder L0–L8; H3c, H9 (direct), H11a/b, H12, H13, H14 |
| **B** | EVENT (or POSSESSION) with clock, score, actors; SHOT x/y | Event-level science: E-HIST (H3a/b), E-STATE (H2, without lineups), E-SHANNON (H7a/b), E-COMP (H5, partial), L9, H10 (event) |
| **C** | Exact SUBSTITUTION / LINEUP_STINT, STARTING_LINEUP, COACH_TENURE, OFFICIAL_ASSIGNMENT, injury snapshots with exact times | Structural simulators L10–L12; H1, H4, H6, H8, H9 (simulator); lineup-conditional forecasts (mode L) |
| **D** | Tracking frames or derived tracking (MATCHUP, closest defender, shot clock, touches) | H2 with Tier-2/3 context; H7c; H10 tracking variant |
| **E** | Medical/load, practice participation, minute restrictions, intended rotations | Upper bounds on availability and minutes forecasting (never required) |

**Experiment-to-level map**

| Experiment | Minimum | Better with |
|---|---|---|
| E-LADDER | A | A + INJURY_REPORT |
| E-HIST | B | C (lineups in state) |
| E-STATE | B | C, D |
| E-SHANNON | B | C, D |
| E-COMP | B | C |
| E-SIM-VS-DIRECT | C | D |
| E-TAIL | B (team level) | C |
| E-STRATA | A (direct) + C (simulator) | – |
| E-JOINT | C | – |
| E-PARAM | A (direct) / C (simulator) | – |
| E-NEURAL | B | D |
| E-XLEAGUE | A | B |
| E-ORACLE | A | C |
| E-POST | A | – |
| E-MARKET | A + MARKET_SNAPSHOT | – |
| E-REALISM | B (team summaries) | C (substitution timing) |

Work proceeds at whatever level the data reaches. Missing higher levels block only the experiments that need them.

## 4. Conventions that prevent silent errors

- **IDs are opaque.** No ordering, outcome, or future state encoded in IDs (L-F1). Leakage test T3 relabels every ID and requires identical forecasts.
- **PLAYER_GAME includes every rostered player**, so the evaluation population is not selected on the outcome (L-E1). T6 checks that it contains non-appearing players.
- **Timestamps are UTC.** Local times are derived via VENUE.tz. Date-only fields are never compared with timestamps without an explicit rule.
- **Rules are data.** RULESET rows (periods, shot clock, three-point distance, foul limits, bonus rules, overtime length) are looked up by the game's `ruleset_id`, never hard-coded, because NBA/WNBA/NCAAM/NCAAW and eras differ (`10_cross_league.md`).
- **Extra columns are allowed** (the synthetic adapter adds several). Missing required columns and contract violations fail `build_store(strict=True)`.

## 5. Adapter obligations

An adapter for the real dataset must:

1. return canonical tables with honest `knowledge_time` and `knowledge_time_quality`;
2. document the provenance of knowledge times per source (`DataAdapter.provenance()`);
3. keep versions rather than overwrite them;
4. never compute cross-game aggregates (season averages, ratings, clusters). Those are features, computed as-of by hoopslab;
5. pass `build_store(strict=True)` and the T4 knowledge-time audit.

`hoopslab.adapters.files.FileAdapter` is a generic mapping-driven adapter (file + column renames + timestamp columns + constants) for parquet/CSV exports. Real sources plug in through a mapping, with no code changes downstream.
