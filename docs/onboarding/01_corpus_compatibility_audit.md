# 01. Corpus compatibility audit — can the frozen experiments run on BASKETBALL_CORPUS_v1.0.2?

*Onboarding, phase P0. Exposure classes S1 (structural) and S2 (data quality) only: no hypothesis
quantity was computed. Every number below is a count, a null share, a coverage share or a
pass/fail fact. Evidence files: `results/onboarding/`. Mismatches: `registry/DATA_CONTRACT_MISMATCHES.md`
(M1–M23). Exposure ledger: `registry/EXPOSURE_LOG.md` rows 1–6.*

**The frozen pre-registration was not changed.** Nothing in this document moves a hypothesis,
threshold, metric, stratum definition or decision rule. Where the corpus cannot support a frozen
experiment, the experiment is reported as not executable; it is never redefined here.

## 0. Two corrections to the mission text

1. The researcher release is **v1.0.2**, not v1.0. The corpus's data-engineering lane shipped
   v1.0.1 (physical isolation) and v1.0.2 (the `event_family` correction, with a box-score
   release gate) after the onboarding mission was written. v1.0.2 supersedes both; its
   `RELEASE_NOTES.md` documents the correction and the gate (every family within ±1 % of the
   box score on common games). The "known v1.0 defect" paragraph of the mission therefore no
   longer applies: `event_family` is the transcribed source outcome, and the adapter uses it
   only as the shot-outcome field, never as the segment definition.
2. **Disclosure (EXPOSURE_LOG row 1).** The agent performing this onboarding is the same agent
   that engineered v1.0.2 on 2026-10-10 as the corpus's data engineer. In that role it saw the
   unrestricted Silver/Bronze structure, the QC evidence and per-season aggregate counts
   including holdout seasons (shot and possession totals used for release gates). No model
   outcome, entropy, skill or calibration figure was ever computed. The evaluator should weigh
   this when reading any later result; the frozen hypotheses predate it (freeze commit
   `843caaa`, 2026-09-27).

## 1. Frozen research state — verified before any corpus access

| Check | Result |
|---|---|
| Branch / commit | `claude/tender-shannon-5kd1nc` @ `611292a` (fresh clone; no commits after the onboarding-governance commit) |
| Tier-1 hashes (25 files) vs FREEZE entry 0 | **identical** after setting `core.autocrlf=false`: the default Windows checkout rewrote every file with CRLF and failed all three registry tests; the content was never different, the line endings were. Keep `autocrlf=false` on this machine. |
| Tier-2 hashes (43 files) vs FREEZE_SUPPLEMENT | identical |
| Archive `registry/archive/prereg-v1/` vs FREEZE entry 0 | identical |
| `python -m hoopslab.registry verify` | PRE-REGISTRATION INTACT (tier 1, tier 2, archive, logs) |
| `pytest -q -m "not slow"` | 33 passed before onboarding; 41 passed after (8 new adapter tests) |
| AMENDMENTS.md | empty |
| Synthetic record `results/synthetic_suite/seed0.*` | unchanged |

## 2. The researcher release (structure only)

`release/BASKETBALL_CORPUS_v1.0.2/researcher_isolated/researcher_isolated_v1.0.2.duckdb`,
51 objects: 33 materialised tables and 18 views over package-local Parquet. Policy tables inside
the package (`isolated_release_meta`, `isolated_split_rules`, `isolated_policy`) state the seal:
RESEARCH `season <= 2019`, VALIDATION `2020–2022`, SEALED_HOLDOUT `season >= 2023` with **no
in-game row of any kind** and schedule rows projected onto 22 pre-game columns. The corpus's
`verify_isolated_package.py` reports ISOLATION PROVEN. `season` is the END year.

Sources are kept as separate families with **per-source, per-league** canonical ids. One
adapter store = one league × one family; families are never mixed inside a store.

| family | box (player/team) | play-by-play | possessions | substitutions | lineups (id-keyed) | shots x/y | tips |
|---|---|---|---|---|---|---|---|
| nba/espn | 2002–2022 | 2002–2022 | derived parser | exact (ESPN subs) | none (QC: UNRECONSTRUCTABLE) | 100 % | exact UTC |
| nba/nba_stats | 1997–2022 | 1997–2022 | source-provided | exact (regex, 96–98 % both ids) | stints 1997–2022, VALIDATED/USABLE | distance + basket-origin x/y | date only |
| wnba/espn | 2003–2022 (2002 no minutes) | 2002–2022 | derived parser | exact | none | 100 % | exact UTC |
| wnba/wnba_stats | **none in the package** (2026 only, holdout) | 1997–2022 | source-provided | exact | stints 1997–2022, mostly PARTIAL/UNRECONSTRUCTABLE | distance only | date only |
| ncaa_men/espn | 2003–2022 | 2003–2022 | derived parser | **none** | none | 3–20 % | exact UTC |
| ncaa_men/ncaa | 2010–2022 | 2010–2022 | source-provided | out-player id 46–47 % | **name-keyed, no ids** | none in pbp | date only |
| ncaa_women/espn | 2004–2022 (minutes from 2014) | 2004–2022 | derived parser | **none** | none | 3–4 % | exact UTC (97 %) |
| ncaa_women/ncaa | 2010–2022 | 2010–2022 | source-provided | out-player id 46–49 % | **name-keyed, no ids** | none in pbp | date only |

Row counts per (table, league, source, season): `results/onboarding/row_counts_by_league_source_season.csv`.
Null shares per field: `field_coverage_by_league_source_season.csv` (full) and
`field_coverage_league_source_era.csv` (league × source × era).

## 3. Frozen contract → corpus mapping (per table)

| Contract table | Corpus source | Match | Notes (structural) |
|---|---|---|---|
| RULESET | `rule_era_geometry` (one row set per league-season, research-sourced) + `seasons` | COMPATIBLE SUBSTITUTE | periods, period length, OT length, shot clock, OREB reset, 3-pt arc/corner (ft→m), bonus rule text; **foul limit absent for WNBA** (adapter default 6, the only non-corpus value, M13) and sourced from a 1918 record for NCAAW; one ruleset per season (`rs:<league>:<season>`) |
| SEASON | `seasons` | EXACT (regular phase only) | start/end dates per league-season |
| VENUE | `games.venue_id/name/city` (ESPN only) | UNAVAILABLE | no time zone, no coordinates (M12); table omitted |
| TEAM | `games` home/away ids within the family | COMPATIBLE SUBSTITUTE | `franchise_id` = team id (relocations are not resolved in the family's id space) |
| PLAYER | `players` | PARTIAL | height/weight/position single version; no birth date, no draft (M11) |
| PERSON_LINK | — | UNAVAILABLE | ids collide across leagues (M10); adapter namespaces by league |
| COACH_TENURE | — | UNAVAILABLE (M3) | |
| ROSTER | game-day appearances (`player_game_box` incl. DNP rows) | COMPATIBLE SUBSTITUTE via contract fallback | spells known 4 h after the first appearance; closed on a team change or after 20 team games without the player (M2) |
| OFFICIAL_ASSIGNMENT | — | UNAVAILABLE (M4) | |
| GAME | `games` (+ `team_game_box` for home/away in the ncaa family) | COMPATIBLE SUBSTITUTE | version 1 = schedule at tip − 7 d (status scheduled); version 2 = final at tip + 4 h; date-only tips → 12:00 UTC with `tip_quality` (M7, M19); All-Star games dropped; phase from `season_type`/`game_type` |
| TEAM_GAME | `team_game_box` | EXACT (ncaa: stl/pf from player sums, M16) | `team_reb` absent |
| PLAYER_GAME | `player_game_box` | EXACT where minutes exist | includes DNP/inactive rows (T6 passes: 24–53 % of a request set did not appear); seasons with < 95 % minutes coverage among played rows are excluded (M8) |
| INJURY_REPORT | — | UNAVAILABLE (M1) | |
| STARTING_LINEUP | realised starters (`starter` flag; opening lineup stint for stats families) | ADAPTER-FALLBACK (M5) | knowledge time tip + 4 h: never visible pre-game; mode S labelled as realised |
| EVENT | `play_by_play` | COMPATIBLE SUBSTITUTE | `event_type` from the corpus's transcribed `event_family`; `elapsed_s` from the per-season RULESET and the game clock; score on every row; x/y per family (M14); attempt value of ESPN misses inferred from wording (M21) |
| POSSESSION | `possessions` (two definitions) | ADAPTER-FALLBACK (M18) | start/end types mapped approximately; `off_margin_start` from the start event's score; segments = 1 (+2 for second-chance rows); the frozen path (hoopslab's own parser over EVENT) is still to be run in P2 |
| LINEUP_STINT | `lineups` stint grain | EXACT for nba_stats / wnba_stats; UNAVAILABLE for ncaa (M15) | |
| SUBSTITUTION | `substitutions` | EXACT for espn (NBA/WNBA) and nba_stats; PARTIAL for ncaa (out player 46–49 %); none for espn NCAA (M15) | |
| SHOT | view over EVENT shot rows | COMPATIBLE SUBSTITUTE | the corpus `shots` table has no event key; ESPN NBA/WNBA complete, ESPN NCAA 3–20 %, stats families basket-origin coordinates, ncaa none (M14) |
| MATCHUP / TRACKING_FRAME | — | UNAVAILABLE (Level D) | NBA Stats matchups exist in Bronze only |
| MARKET_SNAPSHOT | — | UNAVAILABLE (M17) | |
| PROPRIETARY_LOAD | — | UNAVAILABLE (as expected) | |

## 4. Temporal safety: event time vs knowledge time

The corpus's own `temporal_semantics` (139 items) and `field_known_at` (689 fields) say the same
thing for every family: **effective time only, never a publication time** (`point_in_time_verdict`
= EFFECTIVE_ONLY or NEITHER; `cannot_point_in_time` = True for every family; the only BOTH-axis
items are two 19-day injury/depth-chart snapshot series without an archive). Consequently:

| Information category | Corpus support | Classification | Adapter rule |
|---|---|---|---|
| box scores, events, possessions, shots, substitutions, lineups | effective time (game) | EVENT-TIME ONLY → RECONSTRUCTABLE AS-OF | known at tip + 4 h (contract fallback), `imputed_conservative` |
| schedule (tip, teams, venue, phase) | final schedule only | RECONSTRUCTABLE AS-OF with a documented bound | known at tip − 7 d; postponements invisible (M19) |
| rosters / transactions | season snapshots + game-day rosters; no transactions | RECONSTRUCTABLE AS-OF (appearance spells) | known 4 h after first appearance (M2) |
| injuries, player status | none in the release | TEMPORALLY UNSAFE if taken from Bronze; UNAVAILABLE here | omitted (M1) |
| starters | realised only | EVENT-TIME ONLY | tip + 4 h (M5) |
| coaches, officials | none in the release | UNAVAILABLE | omitted (M3, M4) |
| player bio | single rule-resolved version | TEMPORALLY UNSAFE in principle (L-F5) | treated as known at first-season start, `bounded`; `positions_seen` lists variants |
| rules | per-season research table | SAFE AS-OF (bounded at season start) | known season start − 1 d |
| market | none | UNAVAILABLE | (M17) |
| derived ratings in the corpus (`muse_v0_2`, RAPM-style season aggregates, `player_value`) | computed from full seasons | **TEMPORALLY UNSAFE** | never read by the adapter |

Leakage paths inspected in the schema (not exploited): `games` carries the final score, winner
and line scores for research seasons on the same row as the schedule (routed to TEAM_GAME at
tip + 4 h; GAME v1 carries pre-game columns only); `player_game_box` carries per-row `offensive_rating`,
`usage_rate` and other within-game aggregates (outcome fields; not loaded); `players.first_season /
last_season / n_seasons` encode the future (not loaded); season-aggregate tables (`players_by_season`,
`teams_by_season`, RAPM, value metrics) encode full-season information (not loaded); `coverage_tiers`
and QC tables are computed over full seasons (used for the audit only, never as features).

T4 passes for every family loaded (no outcome known before tip + 90 min; no `ingested_time` column
exposed; no "unknown" quality: every row is `imputed_conservative` or `bounded`, never `exact`).
The mandatory optimistic-imputation sensitivity run (contract §1) will apply to every primary
result because no knowledge time in this corpus is exact.

## 5. League × era coverage (seasons ≤ 2022; shares of rows)

Eras follow source switches and rule breaks: NBA 1997–2001 (nba_stats only), 2002–2010,
2011–2017, 2018–2022; WNBA 1997–2001, 2002–2005 (halves), 2006–2012 (quarters, 24 s from 2006),
2013–2022; NCAA 2002–2009 (ESPN only), 2010–2015, 2016–2022 (30 s clock; NCAAW quarters 2016).

| Field | nba/espn | nba/nba_stats | wnba/espn | ncaa_men/espn | ncaa_men/ncaa | ncaa_women/espn | ncaa_women/ncaa |
|---|---|---|---|---|---|---|---|
| tip (UTC instant) | 100 | 0 | 100 | 100 | 0 | 90–100 | 0 |
| minutes on played rows | 99 (2002–05, 2013–22); 71–80 (2006–12) | 100 | 0 (2002); 78–90 (2003–12); 99–100 (2013–22) | 100 | 100 | 0–34 (2004–13); 98–100 (2014–22) | 100 |
| `did_not_play` / starter flags | 100 | none | 100 | 100 | none | 100 | none |
| event score on row | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| event clock remaining | 100 | 100 | 100 | 100 | 0 (string clock only) | 100 | 0 (string clock only) |
| event actor id | 91–93 | 100 | 90–92 | 90–91 | 73–88 | 90–91 | 73–89 |
| shot x/y | 100 | 0 (basket-origin x_raw 100) | 100 | 0 / 9 / 16 by era | 0 | 0 / 4 / 3 | 0 |
| shot clock | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| possessions with lineup | 0 | 100 | 0 | 0 | 95 (names) | 0 | 96 (names) |
| substitutions both ids | 99–100 | 96–98 | 99–100 | — | 46–47 | — | 46–49 |
| rosters canonical player id | 100 | 100 | 100 | 100 | 38–40 | 100 | 41–44 |

Full matrix: `results/onboarding/field_coverage_league_source_era.csv`.

## 6. Data-quality facts (S2, from the corpus's own QC tables and two recomputations)

- **Play-by-play final score = official score:** nba_stats 98.0 %, ncaa source 99.3–99.5 %,
  ESPN 92–95 % (NBA 93.9, NCAAM 94.6, NCAAW 95.3, WNBA 92.1). The frozen target (≥ 99.5 %) is met
  by the ncaa family only; ESPN play-by-play has missing or duplicated events in 5–8 % of games.
- **Possessions per team-game vs box estimate (FGA − OREB + TOV + 0.44 FTA), derived parser:**
  within ±3 on 67–83 % of team-games, within ±6 on 85–95 % (NBA espn 73/95, nba_stats 67/95,
  NCAAM espn 83/90, NCAAW espn 74/85, WNBA espn 72/95). The frozen target (±3 on 95 %) is not met
  by any family; the **source-provided ncaa possessions** differ from the box estimate by a
  median of −67 per team-game (a different unit: M18) and must not be used as possessions.
- **Lineups (corpus `lineup_reconstruction_qc`):** every ESPN game UNRECONSTRUCTABLE; nba_stats
  7.5–16 % VALIDATED and the rest USABLE_WITH_FLAGS; ncaa source 40–55 % USABLE_WITH_FLAGS,
  31–37 % PARTIAL, 12–26 % UNRECONSTRUCTABLE, but name-keyed (M15); wnba_stats mostly
  UNRECONSTRUCTABLE before 2016 and PARTIAL after.
- **Key uniqueness:** games unique; events unique on (game, sequence); 10,893 duplicate
  possession keys (0.02 %); box duplicates explained by NULL-id team rows and 193 double-recorded
  transfers (M20).
- **Substitution parsing:** ESPN 99.6 % both players identified; nba_stats 96–98 %; ncaa 46–49 %.
- **Player minutes vs expected / player points vs team points:** the corpus `quality_flags`
  rows for these checks did not surface in the package's column layout; left to P1 (they are
  pipeline checks, not outcomes).

## 7. Identity

Canonical ids are per league and per source. Within a family, games join on `source_game_id`
(the adapter's key, M23); cross-source game resolution covers 3–25 % of games (crosswalk) and is
not needed inside a family. Players: nba/espn 100 % canonical ids; nba_stats 100 %; ncaa source
38–44 % in rosters, 89–92 % in boxes; ESPN NCAA 100 %. Cross-league: **none** (M10). The
7,749 "multi-league" canonical ids are collisions of the same `cp:espn:<n>` string across leagues,
not links. NCAA → WNBA/NBA transitions cannot be followed.

## 8. Rule eras

`rule_eras` (4,504 rows; confidence verified / likely / unverified) and `rule_era_geometry` give,
per league-season: 4 × 12 min (NBA); WNBA 2 × 20 min halves to 2005 then 4 × 10 min, shot clock
30 s then 24 s from 2006, 14 s OREB reset from 2016, 3-pt arc 20.52 ft then 22.15 ft; NCAAM
2 × 20 min, 35 s then 30 s from 2016, arc 19.75 → 20.75 → 22.15 ft (2020), 20 s OREB reset 2022;
NCAAW halves to 2015 then quarters, 30 s, same arc history, bonus two shots from the 5th team
foul per quarter from 2016. These are the corpus's values (`results/onboarding/rule_eras_contract_areas.csv`),
not recalled ones; where the corpus has no value (WNBA foul limit) the adapter says so (M13).
Era boundaries used for strata: NBA 2018 (14 s reset), WNBA 2006 and 2016, NCAAM 2016 and 2020,
NCAAW 2016. The first season after each is the pre-registered post-rule-change stratum (07 §4).

## 9. Executability of every frozen experiment

Levels actually reachable per league (one family per store):

| League | Level A | Level B | Level C | Notes |
|---|---|---|---|---|
| NBA | nba_stats 1997–2022 (26 seasons, date-only tips); espn 2002–05 + 2013–22 (exact tips) | both families | **nba_stats** (stints + regex subs + opening-lineup starters) | ESPN at C only by substitutions, lineups UNRECONSTRUCTABLE |
| WNBA | espn 2013–2022 (10 seasons; 2003–12 excluded for minutes) | espn 2002–22; wnba_stats 1997–22 (events only) | **not reachable**: lineups live in wnba_stats, which has no box | |
| NCAAM | espn 2003–2022 (20); ncaa 2010–2022 (13) | both | not reachable (name-keyed lineups; ESPN has no subs) | |
| NCAAW | espn 2014–2022 (9; 2004–13 excluded for minutes); ncaa 2010–2022 (13) | both | not reachable | |

Windows (07 §2, complete Level-A seasons ≤ 2022; the corpus's 2023+ seal is a second, external
lockbox for the evaluator): NBA nba_stats 26 seasons → burn-in 1997–1999, development 2000–2020,
lockbox 2021–2022; WNBA espn 10 → burn-in 2013–2015, development 2016–2020, lockbox 2021–2022;
NCAAM espn 20 → burn-in 2003–2005, development 2006–2020, lockbox 2021–2022; NCAAW espn 9 →
burn-in 2014–2016, development 2017–2020, lockbox 2021–2022. 2020 (bubble / shortened) and 2021
are flagged strata; 2021 is never the only confirmation season because 2022 is present.

| Experiment / hypothesis | Required | NBA | WNBA | NCAAM | NCAAW | Status | Mismatches |
|---|---|---|---|---|---|---|---|
| E-LADDER L0–L2, L4–L8 | A | ready | ready | ready | ready | **READY** | M8, M9 |
| E-LADDER L3 (availability) | A + INJURY | runs, degenerate | same | same | same | **READY WITH DOCUMENTED ADAPTER** (no availability information exists: L3 ≈ L2 + appearance history, reported as such) | M1 |
| H3c (form) | A | ready | ready | ready | ready | **READY** | |
| E-HIST H3a, H3b | B | ready | ready | ready | ready | **READY WITH DOCUMENTED ADAPTER** (segments from supplied possessions until hoopslab's parser runs over EVENT in P2) | M18 |
| E-STATE H2 | B (C) | ready at C (nba_stats) | B only | B only | B only | **PARTIAL** (officials, shot-clock proxies and matchups blocks absent; lineup variant NBA only) | M4, M14, M15 |
| E-SHANNON H7a, H7b | B | ready | ready | ready | ready | **READY**; shot-zone ladder (iii) NBA/WNBA + ESPN-NCAA subset only | M14 |
| H7c | D | — | — | — | — | **BLOCKED** | M14 |
| E-COMP H5 | C (simulator variants) | ready | — | — | — | **PARTIAL** (NBA only) | M15, M22 |
| E-SIM-VS-DIRECT H1a, H1b | C | ready (nba_stats) | — | — | — | **PARTIAL** (NBA only) | M9, M15, M22 |
| E-TAIL H4 | B team / C player | ready | team level | team level | team level | **PARTIAL** (player level NBA only) | M15 |
| E-STRATA H6a teammate_star_out | A + INJURY | — | — | — | — | **BLOCKED** | M1 |
| H6b return_from_absence | A | ready (8.6–9.8 k returns) | ready (526) | ready (15–21 k) | ready (4.8–8.8 k) | **READY** | |
| H6c own_move_21d | ROSTER announcements | partial (1.6 k player-seasons with a mid-season team change; the move is known only after the first game with the new team) | partial (507) | partial (480) | partial (181) | **READY WITH DOCUMENTED ADAPTER** (stratum starts one game late) | M2 |
| H6d coaching_change_20g | COACH_TENURE | — | — | — | — | **BLOCKED** | M3 |
| H6e role_change_alarm | A | ready | ready | ready | ready | **READY** | |
| H6 confirmatory decision (falsified only if H6a AND H6c fail) | | | | | | **INVALID WITH CURRENT DATA** as frozen: H6a cannot exist; any decision on H6b/c/e alone is an amendment (DATA-DRIVEN NECESSITY), to be logged before any outcome is seen | M1 |
| E-JOINT H8 | C | ready (nba_stats) | — | — | — | **PARTIAL** (NBA only) | M15, M22 |
| E-PARAM H9 | A (direct) / C (sim) | ready / ready | direct only | direct only | direct only | **READY** (direct); simulator variant NBA only | |
| E-NEURAL H10 | B | ready | ready | ready | ready | **READY** (compute permitting) | |
| E-XLEAGUE H11a | A, calendar as-of | ready | ready | ready | ready | **READY** (one as-of clock; ids namespaced per league) | |
| H11b rookies / translation priors | A + PERSON_LINK | — | — | — | — | **BLOCKED** | M10, M11 |
| E-ORACLE H12 | A | ready | ready | ready | ready | **READY** | |
| E-POST H13 | A | ready | ready | ready | ready | **READY** | |
| E-MARKET H14 | MARKET | — | — | — | — | **BLOCKED** (optional) | M17 |
| E-REALISM (P3 gate) | B team / C subs | ready | ready (subs exact) | team level | team level | **READY** for every simulator that can be built | |

Counts by hypothesis (20 registered): READY 9 (H3a, H3b, H3c, H7a, H7b, H9, H10, H11a, H12,
H13 → 10 including H13), PARTIAL 6 (H1a, H1b, H2, H4, H5, H8), BLOCKED 4 (H7c, H11b, H14, and
H6 as frozen). Stated plainly: **READY 10 / 20, PARTIAL 6 / 20, BLOCKED 4 / 20.**

## 10. Structural power inventory (counts only; seasons ≤ 2022)

| family | games with box | player-games | players | rookies / season (median) | player-seasons with an in-season team change | returns after ≥ 6 team games absent | possessions | substitutions | events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nba/espn | 26,907 | 708,910 | 2,174 | 79 | 1,649 | 8,565 | 4,960,037 | 1,122,781 | 11,702,617 |
| nba/nba_stats | 32,732 | 902,731 | 2,927 | 83 | 1,596 | 9,815 | 6,372,118 | 1,377,071 | 16,210,329 |
| wnba/espn | 4,459 | 102,459 | 865 | 32 | 507 | 526 | 737,113 | 204,421 | 1,828,428 |
| ncaa_men/espn | 100,359 | 2,876,777 | 75,312 | 3,742 | 480 | 20,581 | 9,641,342 | 0 | 23,044,534 |
| ncaa_men/ncaa | 74,867 | 1,300,086 | 26,820 | 2,015 | 55 | 15,073 | 9,575,947 | 4,262,431 | 34,173,048 |
| ncaa_women/espn | 44,634 | 1,109,369 | 32,761 | 1,710 | 181 | 4,794 | 4,502,222 | 0 | 10,648,403 |
| ncaa_women/ncaa | 70,180 | 1,218,951 | 21,908 | 1,630 | 47 | 8,812 | 9,255,907 | 3,915,375 | 32,599,025 |

Read against the pre-registered power notes: E-HIST wanted ≈ 250 k segments per NBA season for
±0.3 millibit; the corpus gives ≈ 240 k possessions per NBA season, so that holds. The H6c
"trade" stratum has ≈ 60 qualifying player-seasons per NBA season (1,596 / 26), which the
synthetic work already flagged as thin; WNBA (≈ 25 / season) and NCAA (≈ 4 / season in the
ncaa family) are thinner still. H6b has ample returns everywhere. Rookie counts support H11b
in principle; the missing cross-league link is what blocks it.

## 11. The adapter

`src/hoopslab/adapters/corpus_v102.py` — `CorpusV102Adapter(league, family, seasons, level)`.
Read-only (`duckdb.connect(read_only=True)`) over the researcher package only; never Bronze/Silver.
One league × family × season range per store; `level` A/B/C decides how much is loaded. Knowledge
times exactly as in §4; every quality is `imputed_conservative` or `bounded`. Ids are namespaced
(`<league>:<canonical player id>`, `<league>:<family>:<team or game id>`). Seasons whose played rows
lack minutes are excluded and listed in `provenance()`; rows dropped for contract reasons are
counted there too. Split enforcement: holdout seasons appear only as GAME version 1 (scheduled)
rows; the package holds no holdout outcome, so none can be served. `tests/test_corpus_adapter.py`
(8 tests, skipped if the release is absent) covers: strict Level-A build, T4, T6, outcome
invisibility at tip − 60 min and visibility at tip + 5 h, holdout split, determinism, namespacing,
a Level-C window, and the T5 static scan of the feature and baseline modules.

Smoke runs (one or two seasons each): nba/espn C, nba/nba_stats C, wnba/espn C, ncaa_men/espn B,
ncaa_men/ncaa B, ncaa_women/espn A/B, ncaa_women/ncaa B, wnba/wnba_stats B (no Level A). All pass
contract validation at their level and T4; T6 passes wherever Level A exists.

Attestation: `registry/ADAPTER_ATTESTATION.json` records the SHA-256 of the adapter, its tests and
the onboarding scripts at this commit, with the corpus package identity it was built against.

## 12. What this onboarding did NOT do

No ladder rung was run on real data. No forecast was issued. No possession parse by hoopslab's own
parser. No entropy, no skill, no calibration, no home advantage, no league comparison of anything
that a hypothesis is about. The one borderline quantity, T6's share of non-appearing players in a
single request set (24–53 %), is a population fact required by the leakage checklist (L-E1).

---

## REPORT

```
CORPUS ACCESS:                 PASS   (researcher release v1.0.2, read-only, isolation proven)
PRE-REGISTRATION INTEGRITY:    PASS   (tier 1/2 hashes and archive identical; AMENDMENTS empty; 41 tests pass)
ADAPTER STATUS:                PASS   (all 8 league-family combinations load; strict Level A for 6 of them;
                                       Level C only for NBA via nba_stats)
TEMPORAL SAFETY:               PARTIAL (every knowledge time is a conservative imputation: no source
                                       carries a publication time; schedule is final-only; injuries,
                                       transactions, coaches, officials absent; optimistic-imputation
                                       sensitivity run mandatory for every result)
HOLDOUT SEAL RESPECTED:        YES    (package physically excludes holdout in-game rows; adapter serves
                                       holdout schedule as version-1 rows only; no Bronze/Silver read)
NBA RESEARCH READY:            YES    (Level A/B/C via nba_stats 1997-2022; A/B via espn with exact tips)
WNBA RESEARCH READY:           PARTIAL (Level A/B via espn 2013-2022; no Level C; 1997-2012 Level A absent)
NCAA MEN RESEARCH READY:       PARTIAL (Level A/B via espn 2003-2022 and ncaa 2010-2022; no Level C)
NCAA WOMEN RESEARCH READY:     PARTIAL (Level A/B via espn 2014-2022 and ncaa 2010-2022; no Level C)
EXPERIMENTS READY:             10 / 20
EXPERIMENTS PARTIAL:            6 / 20
EXPERIMENTS BLOCKED:            4 / 20   (H6 as frozen, H7c, H11b, H14)
MATERIAL DATA-CONTRACT MISMATCHES:
  M1 no injury reports (H6a blocked; L3 degenerate); M2 no transactions (roster spells from
  appearances; H6c one game late); M3 no coach tenures (H6d blocked); M6 no publication times
  anywhere (all knowledge times imputed conservatively); M8 minutes missing in several ESPN
  seasons (excluded); M9 WNBA Stats has no box scores (WNBA Level C unreachable); M10 no
  cross-league identity and colliding ids (H11b blocked; ids namespaced); M15 NCAA lineups
  name-keyed and ESPN NCAA without substitutions (NCAA Level C unreachable); M18 supplied
  possessions are not the frozen segment definition (hoopslab parser to run in P2); M22 ESPN
  lineups unreconstructable (NBA Level C via nba_stats only). Full list M1-M23.
MATERIAL LEAKAGE RISKS:
  final-schedule-only GAME rows (postponements invisible; bounded by tip - 7 d); single-version
  player bio (retroactive position relabels possible); corpus derived ratings and season
  aggregates computed from full seasons (never loaded); the dual-role disclosure in EXPOSURE_LOG
  row 1 (engineering-level familiarity with holdout-era aggregates).
KNOWN V1.0 DEFECT HANDLING:
  superseded: v1.0.2 corrected event_family and gated it against the box score; the adapter
  reads event_family only as the transcribed shot outcome; the master release was not touched.
READY TO BEGIN BASELINE RESEARCH: YES   (P1: L0-L4 on NBA nba_stats first; no model fitted yet)
```


## Erratum (2026-10-30, before P1 outcomes)

The adapter's roster gap-close rule did not fire at the time of this audit (changelog C9: an integer time-unit mismatch). Roster spells therefore stayed open until a transfer, and the T6 share of non-appearing players quoted in §3 and §12 (24-53 %) was inflated by stale spells. After the fix a team's as-of request population is about 16 players (nba_stats, 2019); T6 still passes. No outcome had been computed when the bug was found; the adapter tests pass and the attestation was refreshed.
