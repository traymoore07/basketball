# 03. Baseline ladder

*Canonical: `registry/baseline_ladder.yaml`. Reference implementations: `src/hoopslab/models/`.*

The ladder exists to make one thing impossible: **a complex model receiving credit for an improvement that a much simpler model already provides.**

## The rungs

| Rung | Model | Info set | Data level | What it adds | Implemented here |
|---|---|---|---|---|---|
| L0 | Climatology | pre | A | league pmf per target; league appearance rate | yes |
| L1 | Player historical mean (NB) | pre | A | player identity | yes |
| L2 | Recency-weighted mean (NB) | pre | A | recency | yes |
| L3 | Availability-aware minutes × per-minute rates | pre | A + INJURY_REPORT | availability (calibrated at the same forecast offset); minutes × rate compound | yes |
| L4 | Hierarchical empirical-Bayes shrinkage (posterior predictive; L4-plugin variant) | pre | A | partial pooling by position; parameter uncertainty | yes |
| L5 | Context-adjusted analytic model | pre | A | opponent, pace, home, rest, vacated-usage redistribution | interface only |
| L6 | **Strong direct distributional ML** | pre | A (+C features) | everything tabular; the "strong direct" reference | interface only |
| L7 | L6 + Gaussian copula | pre | A | joint benchmark | interface only |
| L8 | State-space latent-skill filter | pre | A/B | Kalman skill tracking | interface only |
| L9 | Simple team-level possession Markov sim | pre | B | the classic Markov simulation | interface only |
| L10 | Structural possession simulator, plug-in, no latents | pre | C | lineups, usage-as-attention, rotations, game script | yes |
| L11 | L10 + game-level latents | pre | C | correct aggregate dispersion | yes |
| L12 | L11 + posterior parameter sampling (**full structural simulator**) | pre | C | parameter uncertainty | yes |
| L13 | Neural sequence components | pre | B/D | learned history and set encoders | interface only |
| L14 | Calibrated blend | pre | C | stacking, PIT recalibration, reconciliation | interface only |
| O1–O4 | Oracles (minutes, availability, lineups, synthetic game latent) | – | – | diagnostic ceilings; never in the credit chain | O4 synthetic only |

"Interface only" rungs are real-data work for after the freeze. Their specifications here are binding: feature families, the distributional head, and the information set.

## Why this order (and not the obvious one)

- **Availability before sophistication.** L3 sits immediately after the recency rung because, in mode U, knowing who plays is plausibly the largest single gain (`04`: 5–15% on MIN). Any later rung that silently lacks availability information would lose to L3 for a trivial reason.
- **Shrinkage before ML.** L4 establishes what partial pooling alone achieves. Gradient boosting (L6) must beat it, not L1.
- **A strong direct model before any simulator.** H1 compares the simulator (L12) with L6, not with L1–L4. A simulator that only beats weak baselines has not shown anything the research question cares about.
- **A simple Markov simulator before the structural one.** L9 isolates what "simulating possessions" buys without player-level structure, lineups, or rotations. If L9 ≈ L10, the structure is not doing the work.
- **Latents and parameter uncertainty as separate rungs.** L10 → L11 → L12 lets H4 and H9 attribute calibration gains to a specific mechanism.
- **Neural last.** L13 must beat the same components with identical inputs (H10). It gets no credit for information a hand-built feature already carries.

## Credit rule (binding)

```
reference(Lk) = the best rung Lj, j < k, with the same info set, selected on the DEVELOPMENT window
credit(Lk)    = relative core-CRPS skill of Lk vs reference(Lk), on the evaluation window
if credit(Lk) is FALSIFIED (CI_high < MPE):  Lk is NON-ADDITIVE
   and every rung built on Lk is compared with reference(Lk), not with Lk
```

The default MPE for a rung is 1% core CRPS skill. Hypothesis-specific MPEs (`02`) take precedence where a rung pair is the subject of a hypothesis.

## Sanity gates (P1)

These are **pipeline checks**, not scientific claims. Failing them means a bug:

- L1 ≫ L0 on every core target (expected 35–55%).
- L3 ≥ L2 on MIN in mode U.
- L4-plugin coverage in the low-sample stratum ≤ L4 coverage.

On a synthetic world with known structure (first smoke run, seed 1, 20 evaluation days), the implemented rungs ordered as expected on MIN (mode U CRPS):

| Rung | MIN CRPS |
|---|---|
| L0 | 6.58 |
| L1 | 1.94 |
| L2 | 1.93 |
| L3 | 1.54 |
| L4 | 1.54 |

The structural simulator (L10) reached ≈1.00 in that world. This is expected, because the synthetic rotation policy is simple and known to it, which the real world's is not (`09`).
