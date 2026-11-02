# R1 specification: season carryover in the historical weighting of L2

*Drafted 2026-11-02 and FROZEN 2026-11-02 by POSTFREEZE_CHANGELOG row C25 before any R1 model was fitted or any R1
number existed. Everything below is fixed; deviations go to the changelog. No number in this document comes
from a run; the only outcome knowledge used is P1's published L2 result and the P1 gate diagnostics
(season-start roster staleness, docs/p1/02). The grid, thresholds, strata, selection rule and decision rule
are not changed after a result is seen.*

*DRAFT, 2026-11-02. Not frozen: becomes `01_carryover_spec.md` by a POSTFREEZE_CHANGELOG row before any
evaluation; everything below is fixed at that moment and deviations are logged. No number in this document
comes from a run; the only outcome knowledge used is P1's published L2 result and the P1 gate diagnostics
(season-start roster staleness, docs/p1/02).*

## 1. Question

Let a player's appearances before a forecast be ordered by tip, with age `a = 0, 1, 2, ...` (most recent
first). L2 weights appearance `i` by `recency(a_i) = 0.5^(a_i / 5)`. R1 asks whether a weight of the form

```
w_i = recency(a_i) x transition(i) x similarity(i)
```

with `transition(i) < 1` for appearances from an earlier season and `similarity(i)` depending on the
player's context then versus now, forecasts the first games of a season better, and how fast the gain
vanishes as current-season appearances accumulate. The targets, the pmf construction (normal for MIN,
negative binomial for counts, league fallback), the dispersion fit and the T-60 cutoff stay exactly L2's.

## 2. Candidate families (the governed grid)

`n_cur` = the player's current-season appearances before the forecast; `prev` = the appearance is from a
season before the current one (season_id of its game differs from the forecast game's).

| family | transition(i) for prev appearances | grid |
|---|---|---|
| A EXISTING BASELINE | 1 | the single reference point (= L2-A-SHADOW-v0.1) |
| B HARD RESET | 0 | one point; with `n_cur = 0` the frozen league-pmf fallback applies (reported separately as the `no_current_history` share) |
| C SOFT SEASON DISCOUNT | c | c in {0.10, 0.25, 0.50, 0.75} |
| D DECAYING CARRYOVER | max(floor, rho^n_cur) | rho in {0.50, 0.70, 0.85}; floor in {0, 0.10} |
| E CONTEXT-SENSITIVE CARRYOVER | D's value x kappa_team^[team changed] x kappa_role^[role changed] x kappa_turnover^[high turnover] | rho, floor as D (restricted to rho in {0.50, 0.85}); kappa_team in {0.25, 0.50}; kappa_role in {0.25, 0.50}; kappa_turnover in {0.50, 1.00} |

Context indicators for E, all as-of the forecast cutoff, built only from allowed tables:

- `team changed`: the player's current ROSTER spell team differs from the team of his last appearance of the
  previous season (or no previous-season appearance);
- `role changed`: his starter share over the last 10 realised-starter records of the previous season versus
  his current-season starter share (current-season part only once `n_cur >= 3`; before that, the previous
  season's last-10 minutes mean versus the current roster's depth: a player is "role changed" when his
  previous-season minutes-per-appearance was below 12 and the current team lost >= 40% of its
  previous-season minutes, or when the frozen P1 `role_change_alarm` stratum fires);
- `high turnover`: the current team retained < 60% of its previous-season minutes among players on its
  current roster spell (share of last season's team minutes played by players still rostered).

The thresholds 10, 3, 12, 40%, 60% are fixed here, before any run, and are not tuned.

Multiple seasons back: `transition` applies once per season boundary crossed (two seasons back =
transition squared for C; for D the exponent uses the current season's n_cur only, applied once per
boundary).

Total grid: A 1 + B 1 + C 4 + D 6 + E (2 rho x 2 floor x 2 x 2 x 2 = 32) = 44 points per target.

## 3. Selection and evaluation (prequential, as in P1)

- Unit: (game, player) forecast at the T-60 cutoff, mode A (P(stat | appears)), the P1 request population
  and the P1 scoring populations.
- Development seasons 2000-2020; for each season S, the grid point per target is **selected on seasons
  <= S-1** (mean CRPS over the season's units, as the P1 half-life was selected) and **evaluated on season
  S**. Nothing is chosen after seeing season S.
- Targets, each evaluated separately: MIN, PTS, REB, AST, FG3M, TOV, PRA.
- Phases of the season (as-of game index of the player's team): game 1; games 2-5; 6-10; 11-20; rest of
  season. The primary contrast is games 1-20; the secondary is the rest of the season (where carryover must
  not hurt).
- Strata (as-of, labelled under governance): same team vs new team; stable role vs changed role; starter vs
  bench (previous-season starter share >= 0.5); high vs low roster turnover; the frozen P1 strata
  `own_move_21d`, `roster_change_21d`, `low_sample`, `role_change_alarm`. Any stratum introduced later is
  labelled EXPLORATORY.
- Metrics: the frozen `hoopslab.metrics.scoring` set (CRPS primary; log score, RPS, PIT, coverage, MAE).
- Inference: paired contrasts against family A on identical units with the frozen
  `hoopslab.metrics.compare` block bootstrap over game days and the P1 decision rule; the minimum practical
  effect is the P1 CRPS MPE (`registry/expectations.yaml`), not a new number.

## 4. Decision rule (fixed at freeze)

- `SEASON-AWARE-L2-v0.2` is produced only if the selected non-A family beats A on CRPS in games 1-20,
  pooled over the development seasons, under the P1 decision rule, and does not lose on the rest of the
  season beyond the MPE, for at least the core targets PTS, REB, AST and MIN. Otherwise the report says
  "carryover: NO ARTEFACT" and the baseline stands.
- "How fast does previous-season information stop being useful": the first phase in which the gain of the
  selected family over A falls below the MPE, per target, with its CI.
- The artefact's weighting rule is the grid point selected on all development seasons (the prequential
  selection's final state), written into the model card. It contains no injury information and no
  current-season data.

## 5. Report fields (fixed)

```
R1 STATUS; BASELINE (A) CRPS games 1-20 per target; BEST FAMILY per target + grid point; GAIN vs A games 1-20
[CI]; GAIN BY PHASE (1 / 2-5 / 6-10 / 11-20 / rest) [CI]; PHASE WHERE GAIN < MPE; STRATA (same vs new team,
stable vs changed role, starter vs bench, high vs low turnover) [CI]; HARD RESET vs SOFT DISCOUNT vs DECAYING
vs CONTEXT; NO-CURRENT-HISTORY SHARE; REST-OF-SEASON LOSS CHECK; ARTEFACT: SEASON-AWARE-L2-v0.2 YES/NO + hash;
HOLDOUT UNTOUCHED; SHADOW/CONTEXT FIREWALL INTACT.
```

## 6. Not done here

No run, no number, no grid change after a number is seen, no injury proxy, no current-season data, no
change to L2-A-SHADOW-v0.1.
