# Lane R2: ROTATION-OPPORTUNITY-RESEARCH

*Lane card. Created 2026-11-02 (docs/lanes/00_lane_registry.md). Historical research, operated by
Research Claude. Status: CREATED, no experiment run.*

**Purpose.** Estimate what happens to the remaining players when one or more rotation players are
unavailable: how minutes, starts, field-goal attempts, usage, possession involvement, assist opportunities
and rebound opportunities are redistributed, and how reliably historical absence situations let us estimate
that. The problem is not "player X is out" but "what happens to the opportunity X leaves".

**Core outputs the lane must be able to produce for a player and a game.**

- BASE ROLE: expectation given the player's usual roster.
- TONIGHT ROLE GIVEN AVAILABLE ROSTER: expectation given who is available.
- A minutes DISTRIBUTION, never a single estimate: at least mean, median, p10, p25, p75, p90 (contract in
  `src/hoopslab/pregame/contract.py: MinutesDistribution`), propagated into the stat forecasts as
  projected minutes x per-minute rate with hierarchical shrinkage where justified, so that production
  uncertainty reflects role uncertainty (a 6-10-minute bench player facing a 25-minute role is not his old
  total-game average).

**Location.** `research_rotation/` (this card, the draft spec, later the frozen spec and report); results
under `results/research_rotation/`; code under `src/hoopslab/` and `experiments/research_rotation/` when the
experiment starts.

**Allowed data.** The corpus researcher package v1.0.2, development seasons 1997-2020, through the as-of
store: PLAYER_GAME box rows with status `played` / `dnp_coach` and the nba_stats did-not-play comment
(historical absence labels, section 2 of the draft spec), rostered players without a box row, ROSTER spells,
PLAYER listed position, GAME, TEAM_GAME; at level C the LINEUP_STINT, SUBSTITUTION and realised
STARTING_LINEUP tables (post-game knowledge; used to measure what happened, never as a pre-game input).

**Forbidden data.** As R1. In addition: injury labels may never be manufactured from outcomes (a poor line
is not an injury); an absence label is the recorded fact that the player did not dress or did not play and
the stated reason category. No current-season data, no live injury report, no `SHADOW_HOME` /
`CONTEXT_HOME` content.

**Inheritance.** Frozen rungs L3 (availability-aware minutes x per-minute rates; `MinutesRate`), L4
(hierarchical shrinkage to position-group priors) and the L5 definition ("vacated usage" share-weighted
redistribution, GLM, no trees); frozen hypothesis H12 (how much forecast error is minutes and
availability). The lane adds the role taxonomy, the absence-situation labels and the redistribution model;
it does not retune L2 or touch the shadow baseline.

**Governance before any run.** `01_opportunity_spec_draft.md` fixes the absence labels and their
limitations, the role taxonomy, the questions, the model family order and the decision rule. Frozen by a
changelog row before evaluation; exposure rows as for every O-dev experiment.

**Output.** Frozen components (role classifier, P(minutes | available roster), opportunity redistribution,
per-minute rates with shrinkage) packaged into `CONTEXT-SHADOW-v0.4` (and the availability-only step
`AVAILABILITY-SHADOW-v0.3`, which needs no R2 model) for the X1 executor, each with a model card and hash.
