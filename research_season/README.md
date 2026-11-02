# Lane R1: SEASON-CARRYOVER-RESEARCH

*Lane card. Created 2026-11-02 (docs/lanes/00_lane_registry.md). Historical research only, operated by
Research Claude. Status: CREATED, no experiment run.*

**Purpose.** Determine how much information from a player's previous season(s) should influence his
forecast early in the next season, and how fast that information stops being useful. The frozen L2 uses a
5-appearance exponential half-life and carries history across seasons without a reset or a context
adjustment; early in a season a large share of its weight can still rest on last season's roster, role,
rotation, teammates and coaching context.

**Primary question.** Should the weight of a historical appearance depend on more than its age in
appearances? Candidate form: `weight = recency(age) x season_transition(previous season?) x
context_similarity(then vs now)`.

**Location.** `research_season/` (this card, the draft spec, later the frozen spec and report);
results under `results/research_season/`; code under `src/hoopslab/` (shared library) and
`experiments/research_season/` when the experiment starts.

**Allowed data.** The corpus researcher package v1.0.2, development seasons 1997-2020 (burn-in 1997-1999),
prequential by season, through the as-of store: GAME, PLAYER_GAME (box rows including did-not-play rows),
ROSTER spells (transaction announcement times), PLAYER listed position, realised starters / LINEUP_STINT at
post-game knowledge time (as history for "role last season"; never pre-game). Exposure ledger rows as for
every O-dev experiment.

**Forbidden data.** Any 2023+ outcome (holdout); lockbox 2021-2022 except an authorised confirmation run;
any prospective outcome, forecast error, ranking or score from `SHADOW_HOME` or `CONTEXT_HOME`; live injury
reports; anything about the current season. No pooling of leagues to raise sample size.

**Inheritance.** P1 rung L2 (`hoopslab.models.baselines.PlayerMean(half_life=5.0)`, half-life fixed on the
development window) and its shadow package `L2-A-SHADOW-v0.1`. The lane changes only the weighting of
historical observations; targets, pmf construction, dispersion fit, league fallback and the T-60 cutoff are
inherited unchanged. Tier-1 registry files are not amended by this lane.

**Governance before any run.** `01_carryover_spec_draft.md` fixes the candidate families, the grid, the
evaluation strata and the decision rule. It is a DRAFT until a POSTFREEZE_CHANGELOG row freezes it (then it
is renamed `01_carryover_spec.md`); only after that row may an evaluation start, with an EXPOSURE_LOG start
row. Weights are never chosen after looking at results: the grid point is selected on prior seasons inside
the prequential loop, as the P1 half-life was.

**Output.** A report with the frozen return fields and, only if the evidence supports it, a frozen artefact
`SEASON-AWARE-L2-v0.2` (model card in `shadow_context/cards/`, implementation hash, pinned commit) that the
X1 executor runs as `SEASON-AWARE-SHADOW-v0.2`. The artefact contains no injury information and no
current-season prospective outcome, so its prospective effect is attributable to season carryover alone.
