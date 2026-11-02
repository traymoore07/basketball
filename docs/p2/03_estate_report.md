# P2 report: E-STATE (H2), does the history signal vanish when the state is represented correctly? NBA `nba_stats` possession segments, development seasons 2000-2020

*Exposure class O-dev, event level (EXPOSURE_LOG rows 15 and 18). Written 2026-11-02 from the complete run
(results/p2_state/). Specification frozen before any fit: docs/p2/02_estate_spec.md (C23). The only post-freeze
changes are C24's summary-script fixes (a dtype error in a diagnostic grouping; a league switch for the WNBA
run) and the iteration cap raised from 1000 to 3000 after the timing month showed unconverged fits
(results/p2_state/launch_log.txt, before any pooled number existed). No block, threshold, contrast or rule was
changed after a result was seen. E-HIST's artefacts and result are untouched. P1 L2-A unchanged.*

## 0. Fixed return block

```
E-STATE STATUS:                     COMPLETE. 62 pre-specified months (every third test month of 2000-2020),
                                    1,891,555 held-out segments in 8,997 games; 23 logit fits + 2 controls and
                                    24 LightGBM fits per month, every fit converged (maxiter 3000); game-cluster
                                    bootstrap B = 2000 on every contrast; WNBA portability run (15 months,
                                    86,821 segments, logit) completed; NCAA not attempted (different event
                                    format, ~15% of fouls without player ids; spec section 7).
REFERENCE STATE S0:                 2.4529 bits/event held out (frozen compact state = E-HIST set A, 1,225
                                    columns median). History on S0: +3.473 mb [+3.327, +3.619] (E-HIST: +3.47).
BEST ENRICHED STATE:                S7 = S0 + team fouls/bonus + gap + late clock x margin + finer score and
                                    early clock x margin + player fouls + rotation + previous segment class
                                    (1,403 columns). Parsimonious state by the frozen "not useful" rule: S4 + PC
                                    (player fouls +0.15 mb and rotation +0.21 mb have CI_high < 0.5 mb and are
                                    left out). The regime is read at S6 as pre-registered.
BEST ENRICHED STATE LOG LOSS:       logit S7 2.4159 bits/event (S6 2.4169; S4 2.4173); LightGBM S7 2.3839
                                    (S6 2.3860) on the same events. Total S0 -> S7 + history: +37.6 mb (logit),
                                    +60.6 mb (GBM).
GAIN FROM FOUL/BONUS STATE:         +8.32 mb [+8.08, +8.56] (logit, 22% of the total); GBM +8.88 [+8.62, +9.14].
GAIN FROM GAP:                      +14.24 mb [+13.96, +14.53] (logit, 38%); GBM +31.65 [+31.21, +32.09]. The
                                    seconds since the offence's previous segment is the largest single block
                                    in both families.
GAIN FROM CLOCK x SCORE REFINEMENT: +13.12 mb (late clock x margin +6.87 [+6.67, +7.06] + finer score / early
                                    clock x margin +6.25 [+5.88, +6.61]) in the logit; GBM +17.82 (+7.14, +10.68).
GAIN FROM PLAYER FOUL STATE:        +0.15 mb [+0.10, +0.20] (logit); GBM +0.06 [-0.05, +0.17]. NOT USEFUL at the
                                    0.5 mb MPE (on-court personal fouls, foul trouble, 5-foul players).
GAIN FROM PREVIOUS CLASS:           +1.05 mb [+0.98, +1.12] on S6 (logit); +1.12 on S0; GBM +2.17 [+2.02, +2.33].
                                    The previous class is NOT a proxy for omitted state: its gain on the
                                    enriched state has CI_low >= 0.5 mb (frozen rule). S0 + PC is 34.5 mb WORSE
                                    than S4 (fouls + gap + clock/score), so the class carries little of the
                                    state information but keeps its own.
RESIDUAL HISTORY AFTER BEST STATE:  r6 (S6 vs S6 + GL + L1 + L3 + L10 + EW) = +1.085 mb [+0.999, +1.172] logit;
+ 95% CI                            GBM +1.226 [+1.052, +1.398] on identical events. r7 (after the previous
                                    class) = +0.531 [+0.465, +0.600] logit; GBM +0.057 [-0.097, +0.214]. Beyond
                                    the game latent: S6 +1.026 [+0.943, +1.107], S7 +0.493 [+0.429, +0.554].
                                    Positive in every season (r6 +0.72 .. +1.61). Controls: state blocks
                                    permuted within cells +0.07 [-0.04, +0.18]; history permuted -0.01
                                    [-0.04, +0.02].
LOGIT VS GBM ON IDENTICAL EVENTS:   1,891,555 events, 62 months, paired per event. GBM better by +8.6 mb
                                    [+8.2, +8.9] on S0, +9.1 on S1, +26.5 on S2, +31.3 on S4, +30.9 on S6, +32.0
                                    on S7 (and the same with history). The extra value sits almost entirely
                                    in the gap block (GBM +31.6 vs logit +14.2) and the clock x score blocks
                                    (+17.8 vs +13.1): interactions the additive logit cannot form. Residual
                                    history is similar (r6 logit +1.09 / GBM +1.23; r7 +0.53 / +0.06).
STATE-SUFFICIENCY RESULT:           REGIME B (small residual memory) by the frozen rule: r6 CI_high >= 0.5 mb in
                                    both families; CI_low >= 1.0 mb in GBM (+1.05) but not in the logit
                                    (+0.999), so the families disagree at the C boundary and B is reported as
                                    pre-registered. Reading: the enriched observable state absorbs 69% of the
                                    history signal (3.47 -> 1.09 mb); with the previous class 85% (-> 0.53 mb
                                    logit) and, in GBM, 99% (-> 0.06 mb). r7 is REGIME B (logit) / A (GBM).
IS LONG SEQUENCE MODELING JUSTIFIED: NO. Not REGIME C; after state + previous class the residual history is
                                    0.53 mb (logit) and 0.06 mb (GBM); E-HIST's GRU was 13.8 mb worse than the
                                    logit. Nothing here reverses that.
IS SHORT MEMORY USEFUL:             YES. The previous segment class adds +1.05 mb [+0.98, +1.12] on the richest
                                    cross-sectional state (CI_low >= 0.5 mb), +2.17 in GBM; it is not explained
                                    by fouls, gap or clock/score. A first-order Markov term belongs in the
                                    state; the residual after it (r7) sits at the MPE in the logit and at
                                    zero in GBM.
PRE-GAME PLAYER-STAT IMPLICATION:   NONE. Every gain here is a within-game, live-state quantity (fouls, gap,
                                    clock, score, previous possession) unavailable at the T-60 cutoff; the
                                    E-HIST downstream result stands (no T-60 player-stat gain from
                                    within-game information). No T-60 claim is made from an event-level
                                    result (spec section 6). L2-A-SHADOW-v0.1 unchanged.
DOES THIS JUSTIFY A FULL SIMULATOR: NO, on information grounds. A direct model of P(next segment | enriched
                                    state + previous class) leaves at most ~0.5 mb/event (logit) and ~0.06
                                    mb (GBM) of within-game history unexplained; a state-transition
                                    simulator cannot recover information the direct state model already
                                    has. What a simulator would still offer is coherence (joint, game-level
                                    distributions from the same transition kernel), which is not an
                                    information question and was not tested here. If a simulator is ever
                                    built, this result fixes its state: start type, home, margin (finer
                                    than +-3), period, clock x margin cells, team fouls / bonus incl. the
                                    last-two-minute flag, gap since the previous segment, teams, lineups,
                                    previous segment class; player fouls and rotation add nothing at the MPE.
P1 L2-A:                            UNCHANGED.
HOLDOUT:                            UNTOUCHED (>= 2023 absent from the package; lockbox 2021-2022 never parsed:
                                    results/p2_state/preflight.json).
SHADOW FIREWALL:                    INTACT (no shadow-executor outcome, forecast or score read; operational
                                    status only).
READY FOR NEXT PHASE:               YES, for whatever Tray designates. Not a simulator and not H6 on this
                                    evidence; the open items are (i) why GBM gains 17-22 mb over the logit on
                                    the gap and clock x score blocks (interaction form of the state, a
                                    modelling question, not a memory question) and (ii) where the 1.1 mb of
                                    residual memory sits: periods 1-3 (+1.24) more than 4Q/OT (+0.64), close
                                    games, and the free-throw and three-point outcome classes (diagnostic,
                                    section 5).
```

## 1. Design (frozen before any fit; docs/p2/02_estate_spec.md)

Unit, target, data, protocol as E-HIST (possession segments, 8-class terminal outcome, prequential by month with
a 730-day training window, held-out bits per event, game-cluster bootstrap B = 2000, MPE 0.5 mb). Nested
observable states S0..S7 (section 3 of the spec), each with and without the frozen within-game history
GL + L1 + L3 + L10 + EW; the frozen L2 multinomial logit (L2 = 1e-4) as primary instrument and LightGBM with
the E-HIST fixed parameters as replication on identical events; permutation controls; 62 months (the E-HIST
GBM months, chosen by the timing projection before any test-month outcome). State variables built from
play-by-play foul events with the running-count cross-check (>= 99.4% agreement in every NBA season; >= 99.7%
WNBA). The regime rule (section 6 of the spec) uses the primary residual r6 in both families.

## 2. Results (logit; tables in results/p2_state/summary_tables.md)

| state | held-out bits | add-one gain (mb) [CI] | residual history (mb) [CI] |
|---|---|---|---|
| S0 compact | 2.4529 | - | +3.47 [+3.33, +3.62] |
| S1 + team fouls / bonus | 2.4446 | +8.32 [+8.08, +8.56] | +0.99 [+0.90, +1.07] |
| S2 + gap | 2.4304 | +14.24 [+13.96, +14.53] | +1.16 [+1.07, +1.25] |
| S3 + late clock x margin | 2.4235 | +6.87 [+6.67, +7.06] | +1.13 [+1.04, +1.22] |
| S4 + finer score, early clock x margin | 2.4173 | +6.25 [+5.88, +6.61] | +1.06 [+0.97, +1.14] |
| S5 + player fouls | 2.4171 | +0.15 [+0.10, +0.20] | +1.11 [+1.02, +1.20] |
| S6 + rotation | 2.4169 | +0.21 [+0.16, +0.25] | +1.09 [+0.99, +1.17] |
| S7 + previous class | 2.4159 | +1.05 [+0.98, +1.12] | +0.53 [+0.47, +0.60] |

Decomposition of S0 -> S7 + history (+37.6 mb): gap 38%, fouls / bonus 22%, late clock x margin 18%, finer
score 17%, previous class 3%, remaining history 1%, player fouls and rotation < 1% together. The residual
history drops from +3.47 mb to +0.99 mb with the first block (team fouls / bonus) and then stays near 1.1 mb
through S6: the gap, clock and score blocks improve the state a lot but explain no further history. The
previous class takes it to +0.53 mb.

Frozen H2 (registry decision_override, S* = S0 vs R = S7): loss +37.1 mb against a threshold of 7.6 mb (10%
of S7's +75.5 mb over the class-frequency model) -> FALSIFIED, as E-HIST already implied; the "supports further
research" clause of H2 is what this experiment executed.

## 3. LightGBM replication (identical events)

Same months, same events, paired per event. The chain reproduces: fouls +8.9, gap +31.6, late clock x margin
+7.1, finer score +10.7, player fouls +0.06 [-0.05, +0.17], rotation -0.07 [-0.20, +0.05], previous class
+2.2; residual r6 +1.23 [+1.05, +1.40], r7 +0.06 [-0.10, +0.21]. Leave-one-block-out from S6 agrees (gap
-26.9 mb when removed; player fouls and rotation 0). GBM beats the logit by 8.6 mb at S0 and by ~31 mb from
S4 on: the additive logit cannot form the gap x clock x margin interactions GBM finds. This is the largest
family difference in P2 and is a modelling result about the state's functional form, not about memory.

## 4. Previous class, short memory, controls

PC on S0 +1.12, on S6 +1.05 (CI_low +0.98 >= 0.5 mb): not a proxy for omitted state. S0 + PC is 34.5 mb worse
than S4. History on S0 + PC +2.84, on S6 +1.09, on S7 +0.53. Both controls at zero (state permuted within
cells +0.07 [-0.04, +0.18]; history permuted -0.01 [-0.04, +0.02]).

## 5. Where the residual memory sits (diagnostic, pooled means; no decision rests on it)

r6 is larger in periods 1-3 (+1.24 mb) than in the fourth quarter and overtime (+0.64), in close games
(|margin| <= 5: +1.29; 16+: +0.22), after makes (+1.26) and defensive rebounds (+1.05), and it concentrates in the
rare outcome classes: free-throw trips (classes 1-3: +11.8, +5.6, +3.9 mb per event of those classes) and
three-point attempts (classes 6-7: +2.4, +1.7), while two-point outcomes carry +0.6 (miss) and -0.2 (make).
The remaining memory is about whether the next segment ends in a foul / free throws or a three, not about
two-point scoring. Per season r6 ranges +0.72 (2000) to +1.61 (2008) with no trend.

## 6. Portability: WNBA (spec section 7; results/p2_state/wnba/summary_tables.md)

wnba_stats 2016-2020 through the same parser (regulation clock rescaled 600 -> 720 s, overtime unchanged;
points reconcile 0.994-0.997, FGA 1.0; foul-tag agreement >= 0.997), 15 prequential months 2018-05..2020-10,
86,821 segments, logit only. The chain gives fouls / bonus +8.0 mb, gap +16.9, late clock x margin +5.5;
finer score -4.0, player fouls -1.6, rotation -0.3; previous class +0.15 [-0.26, +0.55]; r6 -0.42
[-0.95, +0.10] -> REGIME A by the rule, logit only. Caveat that limits this reading: with ~50,000 training
segments the frozen S0 design (1,225 columns incl. ten player bags) is 24 mb WORSE than the class-frequency
model, so the WNBA chain measures overfitting of the frozen design at that sample size as much as state; the
blocks that add columns hurt and the compact blocks (fouls, gap, late clock) still gain +30 mb together.
The NBA result is the result; the WNBA run shows the first three blocks port and says nothing reliable
about the residual. No pooling was done.

## 7. What was not done

No possession simulator, no entropy-exploitation analysis, no H6, no neural sequence model (not justified:
regime B, residual after state + previous class <= 0.5 mb), no change to the E-HIST artefacts or to the P1 /
shadow baselines, no reading of shadow-executor outcomes, no holdout or lockbox access, no NCAA run.

## 8. Artefacts

results/p2_state/: summary.json, summary_tables.md, preflight.json, launch_log.txt, convergence_check.*,
state_build_log.txt, parser_regression_2019.txt, pytest_p2_state.log; wnba/: summary.json, summary_tables.md,
prepare_log.txt; bits/ (per-event bits, 62 months x 2 families, git-ignored, on the i9; the 9 server-run logit
months copied back). Code: experiments/p2_state/{state_blocks, state_vars, run_state, launch_state,
summarize_state, preflight_state, wnba_prepare}.py, tests/test_p2_state.py. Compute: 441 fit-hours (logit) +
89 (GBM) across the i9 and TrayServer, 2026-11-01 01:30Z to 2026-11-02 07:30Z wall.
