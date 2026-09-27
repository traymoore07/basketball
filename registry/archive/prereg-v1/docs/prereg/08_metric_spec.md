# 08. Metric specification

*Implementation: `src/hoopslab/metrics/scoring.py` (scores), `compare.py` (paired inference and the decision rule), `information.py` (bits, usable information, arithmetic coding), `diagnostics/joint.py` and `diagnostics/realism.py`.*

## 1. What we predict

### Player targets (full game including overtime; integer supports)

| Target | Definition | Support |
|---|---|---|
| MIN | round(seconds / 60) | 0–70 |
| PTS | points | 0–90 |
| REB | OREB + DREB (player rebounds; team rebounds excluded) | 0–40 |
| AST | assists | 0–30 |
| FG3M | made threes | 0–16 |
| TOV | turnovers | 0–15 |
| STL, BLK | steals, blocks | 0–12, 0–14 |
| PRA, PR, PA | sums | 0–140 / 0–120 / 0–110 |

Joint quantities are computed from each model's own joint samples:

- double-double;
- P(PTS ≥ a ∧ REB ≥ b);
- teammate combinations (P(PTS_i ≥ a ∧ PTS_j ≥ b));
- player share of team points.

Models without joint samples must supply a copula (L7) or they are not scored on joint targets.

### Game targets

- home and away team points,
- total,
- margin (support −80…80),
- P(home win), incl. overtime,
- possessions,
- P(OT),
- team REB / AST / 3PM / TOV / FTA.

### Conditioning modes (never silently mixed; every score row carries its mode)

| Mode | Population scored | Forecast |
|---|---|---|
| **U** unconditional (primary) | every player on the as-of roster at forecast time | p(appear)·cond + (1 − p(appear))·δ₀ |
| **A** conditional on appearing | player-games with seconds > 0 | cond |
| **S** conditional on starting | starters (announced or realised; labelled) | model's starter-conditional pmf |
| **L** conditional on known lineup | forecasts issued at lineup lock (info set "lock") | as issued |

Scores in different modes are never averaged together.

## 2. Scoring rules: which metric belongs to which target, and why

All scores are losses (lower is better). Model comparisons use **relative skill** 1 − loss_cand / loss_ref on paired units.

| Target type | PRIMARY | SECONDARY | DIAGNOSTIC | Why |
|---|---|---|---|---|
| Player count stats (PTS, REB, AST, 3PM, TOV, MIN, PRA…) | **CRPS** (exact integer form) | log score (bits), RPS on pre-registered bins | randomised PIT (KS, dispersion ratio, coverage 50/80/95), MAE of median, RMSE of mean, twCRPS (upper tail) | CRPS is strictly proper, sensitive to distance (29 is closer to 30 than 10 is), robust to tail smoothing, and in the units of the stat. The log score is local and dominated by tail smoothing of simulated pmfs, so it vetoes rather than decides. |
| Binned player forecasts (e.g. P(<20), 20–24, …) | **RPS** | log score over bins | reliability per bin | ordinal |
| Binary events (win, over a threshold, OT, double-double) | **log score** | Brier | reliability diagram; Murphy decomposition | the log score is the information-theoretic score for probabilities; Brier decomposes into reliability and resolution |
| Team totals, margin | **CRPS** | log score (integer pmf) | PIT, coverage, realism gates | as for player counts |
| Event level (possession segment) | **log score in bits** = code length | Brier (multiclass) | calibration by class, placebo nulls | prediction = compression; usable information is a difference of log scores |
| Joint within-team vectors | **energy score** on standardised components | variogram score (p = 0.5) | pair-τ calibration (on jittered-PIT scale), derived-quantity PIT (PRA, team totals), named checks (`01` E-JOINT) | Strictly proper. In the synthetic calibration (`09`), with identical marginals, it detected true minutes dependence (+10.7%, CI [8.1%, 13.7%]) where the variogram score (p = 0.5) did not. The variogram score is kept as secondary because the literature finds it more sensitive in other settings |
| Tails | **threshold-weighted CRPS** (weight 1{k ≥ r}) | coverage error at 90%, PIT dispersion | exceedance reliability | H4 |

**Point forecasts** are diagnostic only. MAE is consistent for the median and RMSE for the mean (Gneiting 2011). Neither can decide a hypothesis.

**Log-score smoothing (pre-registered):** every pmf is mixed with the uniform distribution on its support using ε = 1e-4 before log scoring, identically for all models. This keeps scores finite without favouring any model. Simulation-based models are responsible for their own Monte Carlo smoothing (the reference simulator uses a reflected discrete Gaussian kernel, `smoothed_pmf_from_samples`). A model whose log score is materially worse than its CRPS suggests is flagged for Monte Carlo under-resolution, and the comparison is re-run with 4× simulations before interpretation.

**Standardisation for multivariate scores:** component scales are standard deviations per target over the development window (not the evaluation window), fixed at P6.

## 3. Aggregation

- **Core aggregate:** equal-weight mean of per-target CRPS skills over {MIN, PTS, REB, AST, FG3M, TOV, PRA}, mode U.
- Per-target skills are always shown next to the aggregate.
- Leagues are never pooled.
- Skill is computed as a ratio of sums over paired units (not a mean of per-unit ratios).

## 4. Inference

- **Paired differences** on identical units (same game, same player, same forecast time).
- **Clusters:** game-days (player-game level) or games (event level).
- **Moving-block bootstrap** over clusters in time order (block = 7 game-days, B = 2000) for CIs of relative skill. One-sided bootstrap p-values for H₀: effect ≤ 0.
- **Multiplicity:** Holm across the confirmatory `primary` family, and Holm within each secondary family.
- **Stratum interactions** (H6): the bootstrap distribution of skill_in − skill_out on the same resamples.
- **Rank correlations across model variants** (H5): joint resampling of event-level and player-level clusters by game-day.

## 5. Decision rule (the "scorecard rule")

Each hypothesis has one primary metric, one MPE, and one reference. With effect E, 95% CI [lo, hi], and Holm-adjusted p:

```
SUPPORTED     p_adj < 0.05  and  lo > 0  and  E >= MPE
HARMFUL       hi < 0
FALSIFIED     hi < MPE                      (confidently below practical relevance)
PROMISING     otherwise, E >= MPE           (supports further research, proves nothing)
INCONCLUSIVE  otherwise
```

**Additional binding constraints**

1. **No-harm:** a SUPPORTED claim is downgraded to PROMISING if any secondary metric for the same comparison is HARMFUL at the aggregate level.
2. **Calibration floor:** a SUPPORTED claim is downgraded if the candidate's PIT-KS distance or |coverage80 − 0.8| is worse than the reference's by more than 0.02 in the same comparison.
3. **One metric cannot carry a claim.** A secondary metric can never upgrade a decision.
4. **Equivalence-style hypotheses** (H2, H3b, H9-noninferiority parts) state their decisions explicitly in the registry, with a `decision_override` field.

## 6. Calibration diagnostics (always reported, never decisive on their own)

- **Randomised PIT histogram** per model × target × mode × stratum (10 bins).
- **KS distance** of the PIT to uniform.
- **Dispersion ratio** 12·Var(PIT): > 1 means too narrow; < 1 means too wide.
- **Central coverage** at 50/80/95%, computed from PITs so that discrete intervals do not over-cover by construction.
- **Reliability diagrams** for binary events.
- **Distribution calibration per bin** for the pre-registered bins.
