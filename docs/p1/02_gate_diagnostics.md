# P1 gate diagnostics: why the two frozen sanity gates failed

*Diagnostic mission (Tray, 2026-10-30), exposure class O-dev, EXPOSURE_LOG row 11. Reads only the
P1 run-2 artifacts (results/p1_ladder/), the frozen adapter tables and corpus counts. No model was
run, re-fit, tuned or changed; no threshold, grid, stratum or population was altered. Groupings that
are not frozen strata are labelled EXPLORATORY and replace nothing. P1's official results
(docs/p1/01_ladder_report.md, commit bf184f4) are unchanged. Scripts: experiments/p1_diag/
population_accounting.py, gate_decomposition.py; outputs: results/p1_diag/*.json, *.md.*

## 0. Fixed answer block

```
P1 ORIGINAL RESULTS:            UNCHANGED (no corrected run is justified: the implementation conforms to
                                the frozen specification and no leak-free alternative population exists in
                                the corpus; the one structural issue found, DI-2, is recorded as mismatch
                                M25 and cannot be repaired by an adapter change without hindsight)
DI-2 CLASSIFICATION:            DATA/ADAPTER ARTIFACT, not real historical information availability:
                                the nba_stats box score lists ~18 players per team-game through 2002-03
                                (~8 of them did-not-play rows: the whole roster incl. the injured list) and
                                ~12 from 2003-04 on (~2 DNP rows: the active list only); played rows are
                                ~10.0-10.6 per team-game in every season. The adapter's roster spells are
                                built from box rows incl. DNP (M2 fallback), so the as-of request
                                population drops from ~20 to ~14 players per team in 2004 and its
                                appearance share rises from 47-49% to 63-71%. Nothing is lost between
                                rostered -> eligible -> requested -> scored (0 rows at every step on 6
                                reconciled game days); the population itself changed with the source.
L1 SANITY-GATE CLASSIFICATION:  PRE-DATA EXPECTATION FALSIFIED. In mode A (no appearance component)
                                L1 vs L0 is 12-29% per target in both eras (TOV 12%, AST 26-29%); the
                                35-55% expectation is not met on any target in any era or mode, and L1
                                conforms to its specification. The extra drop in mode U after 2004
                                (33-35% -> 13-21%) is DI-2 plus the appearance-rate population artifact
                                (section 5): L1's p_appear is worse than climatology on non-appearing
                                units after 2004 (-25% to -103% per target), better before 2004.
L3 MIN SANITY-GATE CLASSIFICATION: PRE-DATA EXPECTATION FALSIFIED (under mismatch M1). L3's p_appear is
                                bit-identical to L2's (max |diff| = 0 in every season: no availability
                                information exists, M1), so the MIN loss is entirely the conditional
                                minutes pmf: the frozen-default estimator (EW half-life 15 appearances,
                                sd = EW sd + 2) is slower and wider than L2's (half-life 5, sd 6): MAE of
                                the median 5.77 vs 5.45 min (mode A), dispersion ratio 0.77 (too wide) vs
                                1.11, bias +0.14 vs +0.04; the gap is largest under role_change_alarm
                                (-7.5% vs -5.4% elsewhere, mode A) and in days 14-59 of a season (-7.5%);
                                L3 wins only in low_sample (+3.2% A) and on non-appearing units (+2.5% U).
                                Code matches the frozen specification; the result stands.
+2 MIN U BIAS EXPLANATION:      POPULATION MISMATCH between the appearance model's training population
                                and the forecast population, not a property of the conditioning definition
                                and not a leak. p_appear (identical in L1-L4) is the player's played
                                box rows over his box rows; a rostered player with no box row for a game
                                is never counted as a non-appearance (12-26% of requests per season). Mean
                                forecast p_appear exceeds the realised appearance share by +0.08 to +0.19
                                in every season (2020: 0.812 vs 0.626) while the conditional minutes
                                forecast is unbiased (mode A bias -0.08 to +0.15 min). The product gives
                                the +1.1 to +3.1 min mode-U bias at every rung above L0.
IMPLEMENTATION CONFORMANCE:     PASS (section 6: every rung matches registry/baseline_ladder.yaml,
                                docs/prereg/07 and 08 on features, window, decay, shrinkage, modes,
                                populations, selection, refit and forecast timing; four spec-silent
                                implementation choices are listed, none violates a frozen rule)
TEMPORAL SAFETY:                PASS (leakage T1/T2/T3/T6/T7 pass 23/23 seasons, T5 passes; knowledge
                                times: schedule 7 d before tip, box rows and roster spells 4 h after the
                                game; forecast at tip - 60 min; request population never uses outcomes;
                                lockbox guard asserted at every forecast time)
HOLDOUT:                        UNTOUCHED (no table with season >= 2023 was read by any script of this
                                mission; the package carries no in-game rows >= 2023)
BEST CURRENT P1 BASELINE:       L2 (half-life 5) in mode U (core +29.80% vs L0, best on MIN in both
                                modes); L3 for the counting targets in mode A (+29.57%), with L4 adding
                                nothing outside the low-sample stratum
SAFE FOR PROSPECTIVE SHADOW MODE: NO for the frozen mode-U forecasts (biased appearance component,
                                season-start roster staleness). The conditional (mode A) L2 forecasts are
                                reproducible, temporally safe and unbiased and are the candidate for a
                                shadow lane once the blockers below are cleared.
BLOCKERS TO LIVE SHADOW MODE:   (1) no live data path: the researcher package ends at 2022 and 2023+ is the
                                sealed holdout (pre-game artifacts only), so history for a future game is
                                outside the package: needs a governance decision on a separate live feed;
                                (2) appearance-rate population mismatch (section 5): mode-U forecasts carry
                                a +2 min bias and p_appear is miscalibrated by +0.1 to +0.2;
                                (3) season-start staleness of appearance spells (opening night 2003-10-28:
                                81 of 115 requests had no box row, 38 of 72 box players were not requested);
                                (4) the MIN pmf of L1/L2 is a fixed-sd normal (sd 6): 80% intervals cover
                                73-75% (mode U/A); (5) DI-2 shows the source's listing policy can change
                                silently: a live feed needs a listing-policy monitor.
```

## 1. Scope and governance

Allowed: counts on the frozen adapter tables and the corpus; per-unit score, PIT and forecast
artifacts already exposed in P1 (EXPOSURE_LOG rows 8-10); the frozen strata (season, phase,
low_sample, role_change_alarm, own_move_21d, roster_change_21d). Post-outcome groupings used here and
labelled EXPLORATORY: appeared-vs-not (outcome split), realised-minutes bins (outcome split), L2's
forecast-mean bins (as-of), days into season (as-of). Not done: any model run, re-fit, grid change,
feature, injury proxy, calibration, threshold or population change; no L5+, no entropy, no possession,
no simulator, no H6, no holdout.

## 2. DI-2: population accounting

### 2.1 What changed in the source (counts, results/p1_diag/population_accounting.md section A)

nba_stats `player_game_box`, NBA, per team-game:

| season | rows / team-game | rows with NULL minutes and zero stats (DNP) / team-game | played rows / team-game |
|---|---|---|---|
| 1997 | 17.1 | 7.1 | 10.0 |
| 1998 | 16.1 | 6.0 | 10.1 |
| 2000 | 18.0 | 7.8 | 10.2 |
| 2003 | 18.2 | 8.1 | 10.1 |
| **2004** | **12.3** | **2.2** | 10.1 |
| 2010 | 12.0 | 1.9 | 10.1 |
| 2020 | 12.6 | 2.0 | 10.5 |

The `did_not_play` flag is never set for nba_stats (0 rows in every season); DNP rows are recognised
by the adapter as NULL minutes with zero stats (status `dnp_coach`). Through 2002-03 the box lists
about 18 players per team: the dressed players plus the injured-list players of the era. From
2003-04 it lists about 12: the active list. Played rows do not change. The corpus documentation
does not mention the transition; it is a source-format change, visible only in these counts.

### 2.2 How it propagates through the adapter (sections B, C)

Roster spells are built from box rows including DNP rows (mismatch M2: the corpus has no transaction
records, so membership = appearance spells, known 4 h after the first box row, closed on a team change
or after 20 team games without a box row). As-of roster size on 15 January:

| season | mean players per team | last box status of rostered players |
|---|---|---|
| 1998-2003 | 18.6-20.5 | about half `dnp_coach` |
| 2004-2011 | 13.8-14.3 | about one quarter `dnp_coach` |
| 2012 (lockout) | 16.7 | |
| 2013-2020 | 14.3-15.9 | |

### 2.3 Reconciliation on representative game days (section F)

Scheduled games -> teams -> as-of rostered players -> eligible -> requests -> scored:

| game day | season | games | rostered as-of | requests | rostered not requested | requested not rostered | requested & played | requested & DNP | requested & no box row | box players not requested |
|---|---|---|---|---|---|---|---|---|---|---|
| 2003-01-15 | 2003 | 11 | 441 | 441 | 0 | 0 | 223 | 182 | 36 | 1 |
| 2003-03-15 | 2003 | 5 | 195 | 195 | 0 | 0 | 96 | 83 | 16 | 0 |
| 2003-10-28 (opening night) | 2004 | 3 | 115 | 115 | 0 | 0 | 30 | 4 | 81 | 38 |
| 2003-11-01 | 2004 | 13 | 561 | 561 | 0 | 0 | 275 | 35 | 251 | 2 |
| 2004-01-15 | 2004 | 6 | 168 | 168 | 0 | 0 | 124 | 20 | 24 | 0 |
| 2004-03-15 | 2004 | 3 | 83 | 83 | 0 | 0 | 59 | 13 | 11 | 0 |

No row is lost at any step: every as-of rostered player becomes a request and every request is
scored (mode U: all; mode A: the played ones). What changes is who is rostered: in January 2003 a
team's request list is 20 players of whom 10 play and 8 are listed DNP; in January 2004 it is 14 of
whom 10 play, 2 are DNP and 2 have no box row. The season-start rows show a second, era-independent
population defect: on opening night the open spells still carry last season's departed players (81 of
115 requests without a box row) and cannot yet contain newcomers (38 of 72 box players not requested),
because a spell opens only after the first appearance and closes only after 20 team games.

### 2.4 Request population and appearance share per season (section D)

| season | requests / game | share played | share DNP | share no box row | mean p_appear L1-L4 | L0 |
|---|---|---|---|---|---|---|
| 2000 | 41.3 | 0.489 | 0.374 | 0.137 | 0.576 | 0.592 |
| 2003 | 41.0 | 0.488 | 0.392 | 0.120 | 0.566 | 0.573 |
| 2004 | 31.6 | 0.630 | 0.137 | 0.233 | 0.735 | 0.584 |
| 2010 | 28.4 | 0.706 | 0.132 | 0.162 | 0.846 | 0.681 |
| 2020 | 33.3 | 0.626 | 0.117 | 0.257 | 0.812 | 0.740 |

### 2.5 Could the adapter use another roster source?

The corpus `rosters` view for nba/nba_stats holds `team_season` snapshots (about 15 players per
team-season, 1997-2026) and sparse `game` rows (55-127 rows per season through 2005, 6,000-11,000 from
2006). The season snapshots are classified SEASON_GRAIN_FROZEN / EFFECTIVE_ONLY in
`temporal_semantics` (no known-at time; a season snapshot contains players acquired later in the
season), so they cannot be used as-of without hindsight; the game-grain rows are post-game artifacts
and incomplete before 2006. There is no leak-free roster source in the package that would make the
population consistent across 2004. Consequence: DI-2 is recorded as mismatch **M25** (consequence
NONE for P1: the frozen season stratum already separates the eras); any population redefinition is an
amendment-level decision, listed under future experiments.

### 2.6 Classification

**DATA/ADAPTER ARTIFACT**: a source-format transition (which non-playing players the box lists)
propagated into the forecast population by the M2 appearance-spell fallback. It is not a change in
real information availability about who was on a roster, and not an implementation bug (the adapter
does what M2 says). All mode-U results pool two populations: 2000-2003 (half the units are listed
non-appearers) and 2004-2020 (mostly dressed players). Mode-A results are affected only through the
composition of appearing players, which the played-row counts show to be stable.

## 3. The L1 >> L0 gate

### 3.1 Per target, per era, per mode (regular season, frozen season stratum)

CRPS skill of L1 vs L0:

| target | A 2000-2003 | A 2004-2020 | U 2000-2003 | U 2004-2020 |
|---|---|---|---|---|
| MIN | 22.1% | 18.5% | 40.8% | 14.8% |
| PTS | 22.5% | 19.1% | 35.2% | 16.8% |
| REB | 21.6% | 21.1% | 34.0% | 16.9% |
| AST | 28.9% | 26.4% | 34.9% | 21.3% |
| FG3M | 24.6% | 23.2% | 25.1% | 19.7% |
| TOV | 12.3% | 11.7% | 26.6% | 11.3% |
| PRA | 23.7% | 20.7% | 38.1% | 17.7% |

Units: A 95,245 / 419,371; U 198,472 / 618,303.

- **Clearly improves:** AST (26-29% in A), FG3M (23-25%), PTS/REB/PRA/MIN (18-24% in A).
- **Small improvement:** TOV (12% in A, 11% in U after 2004).
- **No improvement:** none in mode A; in mode U after 2004, on non-appearing units L1 is *worse*
  than L0 on every target (see 3.2).
- **Before vs after DI-2:** in mode A the drop is 2-4 points (composition of appearing players barely
  changes); in mode U the drop is 6-26 points.

### 3.2 Why mode U drops in 2004 (EXPLORATORY outcome split, results/p1_diag/gate_decomposition.md section 2)

| era | units that did not appear | L1 vs L0 on them (MIN / PTS / AST) | units that appeared | L1 vs L0 on them (MIN / PTS / AST) |
|---|---|---|---|---|
| 2000-2003 | 129,495 (52%) | +34.9% / +34.4% / +28.3% | 118,967 | +44.0% / +35.2% / +36.2% |
| 2004-2020 | 211,282 (32%) | **-25.5% / -30.2% / -40.6%** | 444,059 | +33.3% / +27.6% / +30.2% |

Before 2004 half the request units are listed DNP players with individual histories of not playing;
L1's player-specific appearance rate beats the league rate on them. After 2004 the non-appearing
units are mostly players with no box row at all, whose own histories contain no non-appearances, so
L1 assigns them p_appear 0.74-0.85 against a league rate of 0.58-0.74 and a realised rate of
0.63-0.71: on those units L1 is worse than climatology, and that cancels a quarter to a half of its
gain on the appearing units.

### 3.3 Is L1 implemented as specified?

Specification (baseline_ladder.yaml): "All-history per-appearance mean with league-fitted
negative-binomial dispersion; player appearance rate." Implementation (hoopslab.models.baselines.
PlayerMean(half_life=None)): per-player unweighted mean over all visible appearances (`ew_weights`
returns ones), league NB dispersion fitted on as-of residuals (`fit_nb_dispersion`), appearance rate
`(played + 5 * league_rate) / (box_rows + 5)`. MIN uses a normal pmf with fixed sd 6 (spec-silent:
MIN is not a count). Conforms. Nothing in L1 explains the gate: the expectation was wrong.

### 3.4 Classification

**PRE-DATA EXPECTATION FALSIFIED.** The bar (35-55%, strong) is missed on every target in mode A in
both eras, where no population artifact operates. Secondary, documented: DI-2 and the appearance-rate
mismatch (section 5) remove a further 6-26 points in mode U after 2004. The implemented gate check in
summarize.py (> 20% on every target) also fails (TOV).

## 4. The L3 >= L2 on MIN (mode U) gate

### 4.1 What L3 adds on MIN, as implemented

L2 (h = 5): MIN pmf = normal(mu = EW mean of minutes over appearances with half-life 5, sd = 6).
L3: MIN pmf = normal(mu = EW mean with half-life 15, sd = EW sd + 2, floor 1); p_appear from the
injury-report status, calibrated as-of, falling back to the player's own appearance history when no
report exists. The corpus has no INJURY_REPORT (M1), so the fallback is used everywhere and L3's
p_appear equals L2's bit for bit (max |difference| = 0.0 in every development season, section 4 of
gate_decomposition.md). The L3-vs-L2 difference on MIN is therefore entirely the conditional
minutes pmf (window and spread).

### 4.2 Decomposition (regular + postseason, 2000-2020; frozen strata first)

Mode A (appearing players; 601,319 units):

| group | share of units | L3 vs L2 skill | share of the L3-L2 loss | bias L2 / L3 (min) | dispersion L2 / L3 | MAE L2 / L3 |
|---|---|---|---|---|---|---|
| all | 1.00 | -5.6% | 1.00 | +0.04 / +0.14 | 1.116 / 0.771 | 5.45 / 5.77 |
| regular / postseason (frozen) | 0.94 / 0.06 | -5.6% / -5.3% | 0.94 / 0.06 | | | |
| low_sample (frozen) | 0.026 | **+3.2%** | -0.02 | +0.16 / -0.35 | 1.15 / 1.17 | 5.97 / 5.66 |
| not low_sample | 0.974 | -5.8% | 1.02 | | 1.115 / 0.761 | |
| role_change_alarm (frozen) | 0.065 | **-7.5%** | 0.09 | -0.31 / +0.21 | 1.16 / 0.77 | 5.64 / 6.14 |
| no alarm | 0.935 | -5.4% | 0.91 | | | |
| own_move_21d (frozen) | 0.078 | -4.8% | 0.08 | +0.27 / +0.84 | 1.27 / 0.96 | 6.30 / 6.70 |
| roster_change_21d (frozen) | 0.51 | -5.5% | 0.51 | | | |
| days into season < 14 (EXPLORATORY) | 0.076 | -0.8% | 0.01 | +0.59 / +0.67 | 1.22 / 0.87 | 6.06 / 6.15 |
| days 14-59 | 0.263 | **-7.5%** | 0.35 | +0.05 / +0.18 | 1.11 / 0.77 | 5.39 / 5.81 |
| days 60+ | 0.662 | -5.4% | 0.64 | -0.03 / +0.06 | 1.11 / 0.76 | 5.40 / 5.71 |
| L2 forecast mean < 8 min (EXPLORATORY, as-of) | 0.048 | -6.0% | 0.04 | -0.82 / +0.45 | 0.89 / 0.84 | 4.65 / 5.01 |
| 8-16 | 0.203 | -5.6% | 0.22 | -0.28 / **+1.06** | 1.25 / 0.83 | 5.99 / 6.44 |
| 16-26 | 0.319 | -5.7% | 0.35 | +0.09 / +0.18 | 1.21 / 0.80 | 5.89 / 6.27 |
| 26+ | 0.430 | -5.4% | 0.38 | +0.25 / **-0.36** | 1.00 / 0.71 | 4.95 / 5.17 |
| realised 1-9 min (EXPLORATORY, outcome) | 0.139 | -4.9% | 0.16 | +6.9 / +8.2 | | 7.01 / 7.95 |
| realised 30+ min | 0.345 | -5.7% | 0.35 | -4.1 / -4.7 | | 5.36 / 5.80 |
| 2000-2003 / 2004-2020 (frozen season) | 0.21 / 0.79 | -5.3% / -5.7% | | | | |

Mode U (963,113 units): overall -3.2%; on non-appearing units (y = 0, 38% of units) L3 is *better*
(+2.5%), on appearing units -8.4%; low_sample +6.6%; role_change_alarm -6.2%; days 14-59 -3.3%, 60+
-4.0%, < 14 +1.2%; every forecast-mean bin negative (-7.6% below 8 min to -1.4% above 26).

Per season (regular, mode A): L3 vs L2 is between -4.6% and -6.6% in all 21 seasons; L3's
dispersion ratio 0.74-0.80 (too wide) and L2's 1.00-1.19 (too narrow) in every season; L3's bias is
above L2's in 18 of 21 seasons.

Reading:

- **Bias.** Small overall (+0.14 vs +0.04 min in A) but structured: L3 forecasts bench players too
  high (+1.06 min in the 8-16 bin) and starters too low (-0.36 min above 26), the signature of a
  slower window pulling toward older minutes levels.
- **Dispersion.** L3's spread (EW sd + 2) is too wide (0.77: 80% intervals cover 89%); L2's fixed
  sd 6 is too narrow (1.11: cover 75%). CRPS penalises L3's width more than L2's narrowness because
  L3's centre is also worse.
- **Calibration.** In mode U L3 is the better-calibrated of the two (cov80 0.794 vs 0.728) and still
  loses CRPS; calibration alone does not rescue a less accurate centre.
- **Role changes and stable roles.** The loss is largest under the role_change_alarm (-7.5% vs
  -5.4%) and in weeks 3-8 of a season when roles settle (-7.5%); for stable players (days 60+) it is
  -5.4%: the slower estimator costs everywhere, most where roles move.
- **Low-minute and high-minute players, traded players, season transitions, roster changes:** all
  negative except the first two weeks of a season (-0.8%, where both estimators are stale) and the
  low_sample stratum (+3.2%, where the sd floor and the prior help).
- **Missing injury information (M1).** Removes the only input that could have made L3's p_appear
  better than L2's; with it absent the two are identical, so M1 explains why L3 has no availability
  advantage, and the minutes estimator explains why it is worse.
- **Does L3's slower estimator react too slowly?** Yes, by the pattern above (role-change alarm,
  weeks 3-8, bench-high / starter-low bias). The half-life 15 and the sd + 2 are the frozen
  implementation defaults recorded in config.json; the specification says only "minutes pmf".

### 4.3 Classification

**PRE-DATA EXPECTATION FALSIFIED** (under M1). Code conforms to the specification; the
availability information the expectation assumed does not exist in the corpus; the minutes estimator
that remains is slower and wider than L2's and loses on every frozen stratum except low_sample. No
corrected run; the result stands. Changing the half-life, the spread rule or adding an availability
proxy would be model changes and are listed under future experiments only.

## 5. The persistent +2 min mode-U bias

From gate_decomposition.md section 6 (L2; L1, L3 and L4 share the same p_appear):

| season | realised appearance share | mean forecast p_appear | difference | mode-U MIN bias (min) | mode-A MIN bias (min) |
|---|---|---|---|---|---|
| 2000 | 0.489 | 0.576 | +0.087 | +1.23 | +0.07 |
| 2003 | 0.488 | 0.566 | +0.078 | +1.13 | +0.12 |
| 2004 | 0.630 | 0.735 | +0.105 | +1.44 | +0.04 |
| 2010 | 0.706 | 0.846 | +0.141 | +2.09 | +0.13 |
| 2015 | 0.687 | 0.822 | +0.135 | +2.10 | -0.02 |
| 2020 | 0.626 | 0.812 | +0.186 | +2.92 | -0.03 |

Mechanism (code, not conjecture): `_HistoryModel.fit` builds the appearance history from
PLAYER_GAME rows (box rows); `appear_counts` = played rows / box rows per player; `league_appear` =
the same over all box rows. The request population is the as-of ROSTER spell set. A rostered player
who is not in the box at all for a game (12-26% of requests, section 2.4) contributes no
non-appearance to any history, so every rung's p_appear is calibrated on a population that appears
more often than the one it is scored on. The conditional forecast given appearance is unbiased
(mode A: -0.08 to +0.15 min). The mode-U bias is the product of the p_appear excess and the
conditional minutes level, and it grows after 2004 because the share of "no box row" requests grows.

Is it expected from the conditioning definition? No. Mode U is defined as p_appear x cond +
(1 - p_appear) x delta_0 on the as-of roster; a p_appear calibrated on that roster would give zero
bias. It is a **POPULATION MISMATCH** between the training population of the appearance component
(box-listed player-games) and the forecast population (roster spells), inherent in the frozen
reference implementation under the M2 fallback. It is deterministic, visible in every season, and
not a leak (it uses less information, not more). Not corrected here; a fix (counting rostered team
games without a box row as non-appearances in the history, or redefining the request population)
changes the models' training population and is an amendment-level decision.

## 6. Line-by-line conformance audit of L0-L4

Sources: registry/baseline_ladder.yaml (frozen), docs/prereg/07 (forecast time, refit, population),
docs/prereg/08 (modes, metrics), config.json, hoopslab.models.baselines, experiments/p1_ladder/
models.py (shared history cache; bit-identical per check_shared_equals_vanilla.py), run_season.py.

| item | specification | implementation | verdict |
|---|---|---|---|
| L0 features | league pmf per target conditional on appearing; league appearance rate | empirical pmf of each target over all visible appearances (+0.01 smoothing); `league_appear` = share of box rows with seconds > 0 | conforms |
| L1 features / window / decay | all-history per-appearance mean; NB with league-fitted dispersion; player appearance rate | unweighted mean over all visible appearances; `fit_nb_dispersion` on as-of residuals (clipped 1.5-500); p_appear = (played + 5 league) / (box rows + 5); MIN: normal, sd 6 | conforms; MIN pmf and the +5 pseudo-count are spec-silent choices |
| L2 decay / selection | EW mean, half-life in appearances fixed on the development window | `ew_weights` 0.5^(age/h) over appearances; grid 5/10/20/40 run in parallel; per-season choice = lowest core CRPS ratio vs L1 on strictly prior scored seasons (burn-in 1998-1999 for 2000); rule written before any outcome | conforms (selection is stricter than "fixed on the development window": it never uses the season being scored) |
| L3 features | P(appear) from injury-report status calibrated as-of; minutes pmf; per-minute EW rates; compound NB over the minutes grid | `_status_rates` over INJURY_REPORT (absent: {} -> falls back to the player's own rate, identical to L1/L2); minutes normal(EW mean h=15, EW sd + 2); EW rate sums h=15; compound moment-matched NB mixed over the minutes grid | conforms under M1; half-life 15, sd + 2 and prior strength 150 are implementation defaults (spec-silent) |
| L4 shrinkage | gamma-Poisson partial pooling of every rate toward position-group priors (method of moments, between-player variance); posterior predictive; plug-in variant | per (listed_position, target) gamma prior from players with >= 60 EW minutes, tau^2 = between-player variance minus sampling variance (floor (0.05 mean)^2); `a0 + y, b0 + m`; posterior predictive adds `a/b^2 * minutes^2`; plug-in sets it to 0 | conforms; minutes component shared with L3 (so L4 = L3 on MIN by construction) |
| conditioning modes | U and A scored separately, never averaged | `score_forecasts` writes one row per (model, target, mode, unit); mode A mask = seconds > 0; summaries keep modes apart | conforms |
| training population | everything visible at the forecast time (as-of view) | `player_history(view)` = PLAYER_GAME rows with knowledge_time <= t joined to GAME | conforms; see section 5 for the appearance-rate population mismatch (spec-silent) |
| target / request population | as-of ROSTER of both teams at the game's tip; never "players who played" | `make_requests` -> `roster_at(view, teams, tip)` on spells visible at t | conforms (0 rows lost, section 2.3) |
| hyperparameter selection | L2 half-life only; everything else fixed | only the L2 half-life is selected, on prior seasons | conforms |
| refit timing | direct models refit every forecast day | `m.fit(view)` at every forecast time (every tip, i.e. every game day) | conforms (at least daily) |
| forecast timing | tip - 60 min, info set "pre" | `forecast_points` yields t = tip - lead(60 min); schedule read as known at chunk start - 1 day with lag-derived 6-day chunks (C11) | conforms |
| scoring metrics | CRPS primary; log, RPS secondary; PIT, MAE, RMSE, twCRPS diagnostics | `hoopslab.metrics.scoring` (frozen file, hash intact) | conforms |
| decision rule | 08 section 5 incl. no-harm and calibration floor | compare.decide (frozen) + constraints.py (C15) | conforms after C15 |
| gates | "L1 >> L0 on every core target (expected 35-55%)"; "L3 >= L2 on MIN mode U"; plug-in coverage | summarize.py tests > 0.20 per target (an implementation threshold written before run 1's outcomes; the 35% expectation would fail as well), skill >= 0, cov80 comparison | conforms; the 0.20 is spec-silent |

Spec-silent implementation choices (none violates a frozen rule; all are candidates for future
experiments, none changed here): MIN pmf shape and sd for L1/L2; the p_appear denominator
(box rows rather than rostered team games); L3/L4 minutes half-life 15 and sd + 2; the gate's 20%
threshold.

## 7. Population consistency through time (train, forecast, score, calibrate)

| stage | population | consistent through time? |
|---|---|---|
| train (history) | box-listed player-games visible at t (played + DNP rows) | changes in 2004 (DNP listing), see 2.1 |
| forecast (requests) | as-of appearance spells of both teams | changes in 2004 (roster size 20 -> 14; appearance share 0.48 -> 0.63-0.71); stale at season starts |
| score, mode U | every request, non-appearance = 0 | same as requests |
| score, mode A | requests with seconds > 0 | stable: played rows 10.0-10.6 per team-game in every season |
| calibration (PIT) | same units as the scores, per mode | same as the scores |

Train and forecast populations are mutually inconsistent in one respect in every season (section 5)
and both shift in 2004 (section 2). Scoring and calibration are consistent with the forecasts they
evaluate. Mode A is the time-consistent view; mode U pools two populations and carries the
appearance-rate mismatch.

## 8. Temporal safety

- Knowledge times: GAME v1 exactly 7 d before tip (single lag, asserted); PLAYER_GAME and ROSTER
  spells tip + 4 h; GAME v2 (final) tip + 4 h; forecast at tip - 60 min; the as-of view exposes
  record versions with knowledge_time <= t only.
- Leakage detectors (docs/prereg/06): T1 truncation, T2 future perturbation, T3 id relabel, T6
  population (share of non-appearing requests > 0 at every tip), T7 feature recompute: 23/23 seasons
  pass on 20 tips each; T5 static scan passes.
- The request population never conditions on an outcome (0 "requested not rostered" rows).
- The lockbox guard is asserted at every forecast time; no season touched 2021-2022; the package has
  no in-game rows >= 2023.
- Verdict: PASS. The population defects above are information deficits, not leaks.

## 9. Corrected-run decision

A corrected run is justified only for a specification violation or a structural corpus/adapter
defect. Findings: no specification violation (section 6). DI-2 is a structural source/adapter
interaction, but the only roster sources in the corpus that could replace appearance spells have no
known-at time (hindsight) or are incomplete before 2006 (section 2.5), so no leak-free corrected
population can be built from the package; the appearance-rate mismatch is a spec-silent
implementation choice whose change alters the models' training population (amendment territory).
**P1 stands unchanged.** DI-2 is recorded as M25; the three spec-silent population items are listed
as future experiments for Tray's decision.

## 10. Prospective shadow forecasting

Can a frozen rung produce reproducible, temporally safe forecasts for future games?

- **Reproducible:** yes. L2 (h = 5) is deterministic given the as-of store (no random component;
  PIT randomisation is scoring-side), bit-identical across the shared-history wrapper and the vanilla
  class, and its artifacts are hashed.
- **Temporally safe:** yes on the corpus (section 8). Prospectively it needs the schedule 7 days
  ahead and box rows within 4 hours of each game, both of which the frozen adapter semantics assume.
- **Population defects:** unresolved for mode U (sections 2, 5): p_appear is +0.1 to +0.2 too high,
  mode-U forecasts +2 min biased, opening-night rosters are stale, and the live source's listing
  policy (DI-2) must be monitored. Mode-A (conditional on appearing) L2 forecasts are unbiased
  (bias -0.08 to +0.15 min across seasons), do not use p_appear, and are the safe candidate.
- **Data path:** the researcher package ends at 2022 and 2023+ is the sealed holdout with pre-game
  artifacts only. A forecast for a future game needs 2023-2026 box history, which is outside the
  package by design: a shadow lane needs a governance decision on a separate live feed (the corpus's
  daily ingest exists on the data-engineer side) that never writes into the researcher package.

Answer: **NO** for the frozen mode-U forecasts as they stand; **the conditional L2 forecasts are
the candidate** once (1) a live data path is approved, (2) the appearance component is either
excluded (mode A only) or re-based on the roster population by an approved amendment, and (3) a
season-start and listing-policy monitor is in place.

## 11. FUTURE EXPERIMENTS (recorded, not run)

1. Re-base p_appear histories on rostered team games (count a rostered game without a box row as a
   non-appearance) and re-measure the mode-U bias and L1 vs L0 on non-appearing units. Amendment.
2. Define the P1 request population as "dressed in the team's previous game" (as-of, box-based) or
   restrict mode U to post-2004 seasons, and re-run the ladder as a separately identified run.
   Amendment.
3. Close appearance spells at season boundaries (or on the first game day of a season) and re-measure
   opening-week populations. Adapter engineering + attestation.
4. L3 minutes estimator half-life and spread rule (sensitivity over 5/10/15 and sd + 0/1/2), and a
   heteroscedastic MIN pmf for L1/L2. Model changes: later authorised phase only.
5. L2 half-life below 5 appearances (boundary result of P1). Model change.
6. A listing-policy monitor for the nba_stats box (rows per team-game, DNP share) as a data-quality
   gate for any live feed.

Stop. No L5, no entropy/possession/simulator work, no H6, no holdout.
