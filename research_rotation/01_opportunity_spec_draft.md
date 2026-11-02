# R2 draft specification: rotation and opportunity redistribution under absences

*DRAFT, 2026-11-02. Frozen by a POSTFREEZE_CHANGELOG row before any evaluation (then renamed
`01_opportunity_spec.md`). No number here comes from a run. Structural facts used: the nba_stats box rows
carry a did-not-play comment (distinct values seen 2026-11-02, EXPOSURE_LOG row 16, class S1): "DNP -
Coach's Decision", "DND - Injury/Illness", "DNP - Injury/Illness", "NWT - Personal Reasons", body-part
strings ("DND - Sprained Left Ankle", ...), "no memo for staff", blank.*

## 1. Questions (historical, development seasons)

1. When a rotation player is absent, how are his minutes, starts, FGA, usage, possession involvement,
   assist opportunities and rebound opportunities redistributed across the remaining roster, and with what
   spread? (Not required to sum to the absent player's line; roles change nonlinearly.)
2. Is the redistribution estimable reliably enough to improve P(minutes) and the stat distributions of the
   remaining players at the T-60 cutoff, historically, when the absence is known at the cutoff?
3. Does a role-based description (which ROLE is absent) generalise better than player-pair interactions,
   in particular for rookies and bench players?
4. Specific situations to answer: Player A when Player B is absent; one of the team's top-two usage players
   absent; the starting point guard absent; the starting centre absent; two starters absent.

## 2. Absence labels (historical, post-game facts; never inferred from a line)

| label | definition from allowed tables | known when |
|---|---|---|
| ABSENT_INJURY | box row with seconds 0 and comment starting "DND" or "DNP - Injury/Illness" (any body-part string counts) | post-game (box score) |
| ABSENT_PERSONAL | comment starting "NWT" (not with team) | post-game |
| DNP_COACH | comment "DNP - Coach's Decision" or did_not_play without an injury/NWT comment | post-game |
| NOT_ON_BOX | rostered (current ROSTER spell) with no box row for the game | post-game |
| UNKNOWN_REASON | seconds 0 with blank / "no memo for staff" comment | post-game |

Limitations, stated once and carried into every result: (i) these are realised absences; the corpus has no
INJURY_REPORT table, so the pre-game knowledge of an absence cannot be reconstructed historically; (ii)
therefore every historical estimate is of the form "given that X did not play, what happened to the
others" and is an upper bound on what a pre-game model can use (the live lane L1 supplies the pre-game
state; the prospective comparison v0.3 vs v0.2 measures what that is worth); (iii) ABSENT_INJURY before
the comment field is populated is indistinguishable from DNP_COACH and is labelled UNKNOWN_REASON; the
share per season is reported; (iv) nothing is ever labelled from the stat line itself.

## 3. Role taxonomy (as-of; frozen here)

Computed from the player's last 10 appearances for the team before the cutoff (fewer if unavailable;
`low_sample` stratum below 5):

| role | rule |
|---|---|
| PRIMARY_CREATOR | highest as-of usage proxy on the team (FGA + 0.44 FTA + TOV per minute x minutes) and assists per minute in the team's top two |
| SECONDARY_CREATOR | second in the usage proxy, or first in assists per minute and not PRIMARY_CREATOR |
| HIGH_USAGE_WING | usage proxy in the team's top three, listed position G/F or F, not a creator |
| STARTING_CENTER | realised starter share >= 0.5 and listed position C (or F-C) |
| STARTING_GUARD | realised starter share >= 0.5 and listed position G (or G-F), not a creator |
| STARTING_OTHER | starter share >= 0.5, none of the above |
| ROTATION_BENCH | starter share < 0.5 and minutes per appearance >= 12 |
| DEEP_BENCH | minutes per appearance < 12 |

The usage proxy uses only box-score columns (no play-by-play needed), so it is available in every season
and every league; possession-level involvement (touches) is a level-B/C refinement, reported where present.

## 4. Outputs and model families (in order; a later family only if the earlier one leaves a material gap)

Output contract per (game, player): `MinutesDistribution` (mean, median, p10, p25, p75, p90, and the pmf on
0..70 minutes) and `OpportunityExpectation` (FGA, usage proxy, assist opportunities, rebound opportunities:
mean and p10/p90), for BASE ROLE and for TONIGHT ROLE GIVEN AVAILABLE ROSTER.

1. F0 reference: frozen L3/L4 minutes x rates (no redistribution).
2. F1 share redistribution: the absent players' minutes and usage proxy go to teammates in proportion to
   as-of shares by role (the L5 "vacated usage" definition), per absent role.
3. F2 role-pair model: redistribution multipliers per (absent role, receiving role), fitted historically
   with hierarchical shrinkage across teams and seasons (gamma-Poisson / normal, no trees).
4. F3 player-pair adjustment: F2 plus a shrunken player-pair term for pairs with >= 10 historical
   co-absence situations; rookies and bench players fall back to F2.

A neural or tree model is not in the lane's first pass; it enters only if F3 leaves a residual the frozen
decision rule calls material, with its own changelog row.

## 5. Evaluation

Prequential by season (2000-2020), T-60 cutoff, mode A; the absence set at the cutoff is taken from the
realised labels of section 2 (the historical upper bound, stated as such). Targets: MIN first (CRPS, PIT,
coverage of the 80% interval), then PTS, REB, AST, FG3M, TOV, PRA through projected minutes x rates.
Populations: all remaining players; by receiving role; by number of absent starters (0, 1, 2+); by absent
role; the four named situations of section 1. Paired contrasts F1-F3 vs F0 on identical units with the
frozen comparison block and the P1 MPE. Strata introduced later are labelled EXPLORATORY.

## 6. Decision rule (fixed at freeze)

A component enters CONTEXT-SHADOW-v0.4 only if it beats F0 on MIN CRPS in the absence populations under
the frozen decision rule and does not lose in the no-absence population beyond the MPE. The report states,
per question of section 1, whether the historical situations support a reliable estimate (CI width relative
to the effect) or not.

## 7. Report fields (fixed)

```
R2 STATUS; ABSENCE LABEL COVERAGE per season (shares of ABSENT_INJURY / DNP_COACH / NOT_ON_BOX / UNKNOWN);
ROLE TAXONOMY COVERAGE; F0 MIN CRPS; F1/F2/F3 GAIN vs F0 by population [CI]; ROLE-BASED vs PLAYER-PAIR
GENERALISATION (rookies / bench); THE FOUR NAMED SITUATIONS [CI]; MINUTES DISTRIBUTION CALIBRATION (80%
coverage, PIT); PROPAGATION TO PTS/REB/AST/FG3M/TOV/PRA [CI]; COMPONENTS FROZEN FOR v0.4 YES/NO + hashes;
HOLDOUT UNTOUCHED; FIREWALL INTACT.
```

## 8. Not done here

No run, no label from outcomes, no injury proxy beyond section 2, no live data, no change to the frozen
baseline.
