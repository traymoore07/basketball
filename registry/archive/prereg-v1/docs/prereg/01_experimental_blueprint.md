# 01. Experimental blueprint

*Pre-registered before any real dataset has been seen. Frozen by `registry/FREEZE.json`. Changes after the freeze go through `registry/AMENDMENTS.md` and are labelled in every report.*

This document is the master plan of the laboratory that will decide which parts of the forecasting architecture proposed in `docs/research/nba_sequence_modeling.md` actually work. It does **not** build that architecture. It defines:

- the experiments,
- the order they run in,
- the gates between them,
- how each result will be read.

The goal is unchanged: **calibrated probability distributions for player statistics and game outcomes**. The claim under test is that the proposed structure helps reach that goal. It is not assumed.

Companion documents:

| # | Document | Canonical source |
|---|---|---|
| 02 | Hypothesis registry | `registry/hypotheses.yaml` |
| 03 | Baseline ladder | `registry/baseline_ladder.yaml` |
| 04 | Pre-registered expectations | `registry/expectations.yaml` |
| 05 | Data contract | `src/hoopslab/contract/data_contract.yaml` |
| 06 | Leakage checklist | this folder, plus `src/hoopslab/backtest/leakage.py` |
| 07 | Temporal backtesting specification | this folder, plus `src/hoopslab/backtest/runner.py` |
| 08 | Metric specification | this folder, plus `src/hoopslab/metrics/` |
| 09 | Synthetic test suite | this folder, plus `src/hoopslab/experiments/synthetic_suite.py` |
| 10 | NBA / WNBA / NCAA design | this folder |
| 11 | What I need from the real dataset | this folder |
| 12 | Repository guide | this folder |

---

## 1. Principles (binding)

1. **Same information, or it's not a comparison.** Two models compared on a hypothesis receive the same as-of information set, unless the hypothesis is *about* information (H11, H12, H14). A simulator that knows the injury report cannot be compared with a direct model that does not.
2. **Everything is as-of.** Every model and feature sees data only through an `AsOfView` at the forecast timestamp. Visibility is decided by `knowledge_time`, never by event time or ingestion time.
3. **Credit flows up the ladder, never around it.** A model's improvement is measured against the best *simpler* model at the same information level (`03_baseline_ladder.md`). A complex model cannot take credit for what a lower rung already delivers.
4. **Proper scoring rules, pre-assigned.** Each target has exactly one primary metric, fixed in `08_metric_spec.md`. Secondary metrics can veto a claim (the "no-harm" rule) but cannot rescue one.
5. **Minimum practical effects are fixed now.** Each hypothesis has an MPE. Statistically significant but practically negligible effects are reported as FALSIFIED under the equivalence-style rule, not as "support".
6. **Development and confirmation are separated in time.** All building, tuning, and exploration happen on the development window. Confirmatory tests run **once** on the sealed lockbox window with frozen configurations (`07_temporal_backtest_spec.md`). The runner refuses to score the lockbox without the frozen manifest hash.
7. **Leagues are scored separately.** NBA, WNBA, NCAAM, and NCAAW results are never averaged into one number (`10_cross_league.md`).
8. **Instruments are verified before they are trusted.** Every diagnostic in this document has a synthetic world where the right answer is known. It must give that answer there first (`09_synthetic_test_suite.md`).
9. **Negative results are results.** Every pre-registered test is reported, including FALSIFIED and HARMFUL outcomes, in the same scorecard format.
10. **Amendments are visible.** Any change to a frozen file after the freeze is logged with its date, its reason, and whether real results had been seen at the time.

---

## 2. Phases and gates

Each phase ends with a gate. A failed gate stops the phases that depend on it. It never triggers a retroactive change of the plan.

| Phase | Content | Data window | Gate to pass |
|---|---|---|---|
| **P0 Intake** | Adapter; contract validation; knowledge-time audit (T4); coverage report; `11_real_data_checklist.md` items | all | Contract valid at the claimed level. Knowledge-time provenance documented per source. Leakage tests T1–T7 pass for L0–L4. |
| **P1 Ladder (simple)** | L0–L4 rolling backtest | development | Ladder is sane: L1 ≫ L0, and L3 ≥ L2 on MIN (mode U). A failure here means a pipeline bug, not a finding. |
| **P2 Event science** | E-SHANNON, E-HIST, E-STATE, E-COMP (event level) | development | Placebo controls behave (H7a on development data). Positive control (future canary) detected. |
| **P3 Simulators** | L9–L12 built; realism gates (E-REALISM) | development | L10+ pass realism gates. **A simulator that fails its gates may not enter H1/H4/H6/H8/H9.** |
| **P4 Strong direct** | L5–L8 built and tuned | development | L6 beats L4 (otherwise L4 becomes the "strong direct" reference, as the credit rule requires) |
| **P5 Dev comparisons** | H1, H4, H5, H6, H8, H9, H10, H13 exploratory runs | development | none (exploratory) |
| **P6 Config freeze** | All model configurations hashed; amendment log closed | – | `python -m hoopslab.registry verify` passes; configuration hashes recorded |
| **P7 Confirmation** | Confirmatory runs of the primary family, once | lockbox | – (results are final) |
| **P8 Cross-league** | E-XLEAGUE (H11) with its own lockboxes per league | per league | – |

Development-window results can motivate an amendment *before P6*. They can never replace a confirmatory result.

---

## 3. Experiment catalogue

Each experiment names:

- the hypotheses it serves (see `02_hypothesis_registry.md`),
- the minimum data level (see `05_data_contract.md`),
- the protocol.

### E-LADDER: baseline ladder (all hypotheses depend on it)

- **Level:** A (L3+ needs INJURY_REPORT).
- **Protocol:** rolling-origin backtest of every implemented rung. Report core CRPS per rung, the credit chain, and the NON-ADDITIVE flags.
- **Output:** the table that fixes which model is the reference for each hypothesis.

### E-HIST: how much does history matter? (H3a, H3b, H3c)

This is the cleanest test of the Markov question.

- **Level:** B.
- **Unit:** possession *segment* (offensive-rebound continuations are separate segments).
- **Target:** 8-class terminal outcome (TO, FT 0/1/2 of 2, 2PA miss/make, 3PA miss/make).
- **Model family:** one fixed family for all feature sets, trained on identical data:
  - L2-regularised multinomial logistic regression with a pre-registered penalty,
  - a gradient-boosted replication,
  - a neural sequence model for set E (H10).

**Nested feature sets:**

| Set | Features | Question |
|---|---|---|
| A | compact state S (start type, home, margin bucket, period, clutch, offense/defense, on-court players) | baseline |
| B | S + previous possession outcome (same offense) | 1-step memory |
| C | S + previous 3 | short memory |
| D | S + previous 10 | medium memory |
| E | S + whole-game history (EW summaries; RNN/transformer in H10) | long memory |
| F | S + game-level latent estimate (running shooting residual, shrunk) | slow variable only |
| G | F + everything in E | history *beyond* the latent |

**Pre-registered contrasts:**

- history_signal = bits(A) − bits(E)
- latent_signal = bits(A) − bits(F)
- history_residual = bits(F) − bits(G)

**Diagnosis (`hoopslab.experiments.history.diagnose_history`, MPE = 0.5 millibit/event):**

- NO_HISTORY_SIGNAL: CI_high(history_signal) < MPE
- GENUINE_HISTORY: CI_low(history_residual) > 0 and estimate ≥ MPE
- APPARENT_HISTORY_EXPLAINED_BY_LATENT: history_signal ≥ MPE, latent_signal ≥ MPE, CI_high(history_residual) < MPE
- INCONCLUSIVE: otherwise

**Omitted-slow-variable attribution.** If GENUINE_HISTORY appears, add the slow variables one at a time and re-measure history_residual: stint length / fatigue proxy, fouls, bonus, timeouts remaining, lineup-change indicator. The history signal is "explained" by the first slow variable that brings history_residual's CI below the MPE.

**Controls:**

- *Negative control:* permute history features across games within the same state cell. This must give ≤ 0.
- *Positive control (canary):* a feature computed from the *next* segment. This must give a large gain. It proves the pipeline can see information, and it is never used by any model.
- *Hot-hand contrast (diagnostic):* pooled vs within-game P(make | make) − P(make | miss). This reproduces the Miller–Sanjurjo bias and between-game confounding on the real data.

**Between games (H3c):** add last-3-game residuals to L8. The metric is player-game core CRPS.

**Statistics:** paired per-event bit differences, bootstrap over games (B = 2000).

**Power note:** with ~37k test events (one synthetic season), CIs are about ±0.7 millibit. One NBA season has roughly 250k segments, so expect about ±0.3.

### E-STATE: is a compact state enough? (H2)

- **Level:** B (lineups at C; tracking context at D).
- **Protocol:** compare S* (Tier 1) with R = S* + Tier-2/3 blocks, in two model families (GLM, GBM) plus a neural variant. Report usable information per block, both leave-one-block-out and add-one-block.
- **Decision:** equivalence-style. H2 is SUPPORTED if the information lost by S* has CI_high below max(0.5 millibit, 10% of I(R)).
- **Follow-through:** any block with significant usable information is also tested downstream (the H7b logic).

### E-SHANNON: entropy and usable information (H7a, H7b; feeds H2 and H5)

**Measure held-out cross-entropy under a nested conditioning ladder.** Each row is an out-of-sample estimate of an upper bound on the corresponding conditional entropy:

```
H(next event)                          intercept only
H(next event | player)                 + identity of the possession user (known after the fact: diagnostic)
H(next event | lineup)                 + 10 on-court players
H(next event | lineup, game state)     + start type, margin x clock, period, bonus
H(next event | ..., extended context)  + history, latent, fatigue, officials (as available)
```

The same ladder is run for three event definitions:

- (i) segment outcome (8 classes);
- (ii) *who uses the possession* among the 5 on court (usage entropy; the attention-shaped component);
- (iii) shot zone given shooter (level B shot data).

**Estimator rules (pre-registered):**

- Report held-out cross-entropy from a fixed regularised model family. Plug-in entropies are shown only for contrast.
- Every "information gain" is reported in bits per event with a game-cluster bootstrap CI.
- Every context block is accompanied by a **placebo** (the same block permuted across games). The fraction of significant placebos is H7a's statistic.
- Downstream relevance is a separate test. Each block with usable information is added to the simulator, and the change in core CRPS is measured (H7b, H5). Bits do not count as success on their own.

**Exploitability (H7c, exploratory, level D):** an indifference test of the expected value of used actions in high-information contexts. "Low entropy" is never reported as "exploitable" without it.

### E-COMP: prediction = compression, tested directly (H5)

1. Fix one symbol stream for all competing event models: the 8-class segment outcome. An optional extended stream adds the actor token (user among the 5 on court) and the rebound and assist actor tokens.
2. Each model is trained prequentially (refit monthly, as-of) and emits a predictive distribution for each symbol, in time order.
3. Compute cumulative code length **twice**:
   - the ideal Σ −log₂ q(symbol);
   - the actual length from the arithmetic coder in `hoopslab.metrics.information`, verified lossless by decoding.

   The two must agree within 2 bits plus quantisation overhead. This check is part of the synthetic suite.
4. A model that does not predict a sub-token (for example, the rebounder) uses a shared, fixed reference coder for it, so totals stay comparable.
5. **Link to player-stat distributions (H5):** run the pre-registered family of ≥ 8 simulator variants (listed in H5) and compute Spearman ρ between Δ(bits per event) and Δ(core CRPS). Name every discordant variant, where compression improves but distributions get worse or vice versa. The expected discordant ones are written down in `04`.

### E-SIM-VS-DIRECT: structural vs direct (H1a, H1b)

- **Level:** C.
- **Comparisons:** L12 vs L6 on core targets, mode U, primary metric CRPS. Also L14 (blend) vs the best single model.
- **Reporting:** stacking weights with CIs (a simulator weight whose CI excludes 0 counts as "supports further research"). Per-target and per-stratum breakdowns are secondary.

### E-TAIL: game-level latents and tails (H4)

- **Comparison:** L11 vs L10 on upper-tail threshold-weighted CRPS and 90% coverage error, for player and team totals.
- **Diagnostic:** the per-event bit difference between L11 and L10 components. The expectation is < 0.5 millibit, which is the "event-calibrated but aggregate-wrong" signature.
- **Realism gates** on team points and total points.

### E-STRATA: generalisation after change (H6a–e)

Interaction tests with **as-of** strata, as defined in `hoopslab.backtest.strata`:

- teammate star out,
- return from absence,
- the player's own move (trade or signing) within 21 days (team-level roster change is secondary),
- coaching change within 20 games,
- CUSUM role-change alarm.

Holm-adjusted within the H6 family. Strata defined from outcomes or from hindsight are forbidden (L-E3).

### E-JOINT: joint forecasts (H8)

- **Models:** L12 joint samples vs L7 (Gaussian copula) vs L6-independent.
- **Primary:** energy score on within-team vectors (MIN, PTS).
- **Secondary:** variogram score. The choice was made before the freeze from synthetic calibration (`09`).

Named dependence checks, each with a statistic and a pass criterion (a model "gets it right" if the realised statistic lies inside the model's 90% predictive interval of the statistic):

| Check | Statistic |
|---|---|
| Teammates' minutes | Kendall τ of randomised-PIT pairs, top-2 minutes players; predicted τ computed on the same jittered-PIT scale |
| Teammates' usage | τ of FGA+0.44FTA+TOV shares, predicted vs realised |
| Teammate scoring | τ of PTS PITs, top-2 scorers |
| Rebounds vs made shots | corr(team DREB, opponent missed FGA); corr(player REB, opponent FG%) |
| Assists vs teammate FGM | corr(player AST, teammates' FGM) |
| Blowout starter minutes | corr(\|final margin\|, starters' total minutes), predicted vs realised |
| Overtime | reliability of P(OT); minutes distribution conditional on OT |
| PRA | PIT dispersion of PTS+REB+AST from each model's own joint samples |
| Game environment | corr(team total points, each starter's PTS), predicted vs realised |
| Joint thresholds | reliability of P(PTS_i ≥ a ∧ AST_j ≥ b) for pre-registered (a, b) grids |

### E-PARAM: parameter uncertainty (H9)

- **Comparisons:** L12 vs L11, and L4 vs L4-plugin.
- **Primary:** coverage-error reduction at 80% in the low-sample stratum (< 10 as-of appearances).
- **Constraint:** overall CRPS must be non-inferior.

### E-NEURAL: neural sequence models (H10)

- **Level:** B (D variant).
- **Comparison:** L13 vs the GLM/GBM components with identical inputs. The GBM also gets hand-made history summaries, so the comparison isolates *architecture* rather than *information*.
- **Learning curves** at 1/2/4/8 seasons and with pooled leagues.
- **Pre-registered reading:** a gain that grows with data size counts as "supports further research" even if it is below the MPE at the current size.

### E-XLEAGUE: NBA/WNBA/NCAA (H11a, H11b)

See `10_cross_league.md`.

### E-ORACLE: where is the error? (H12, descriptive)

Oracle minutes (O1) and oracle availability (O2) substituted into L6 and L12. Share of CRPS removed, per target.

### E-POST: post-processing (H13)

Rolling as-of PIT recalibration (isotonic on the CDF) of L6 and L12.

### E-MARKET: encompassing test (H14, sealed optional)

Runs only if market snapshots with honest times exist. Uses only the T-60 snapshot, never the closing line for earlier forecasts.

### E-REALISM: do simulated games look like real games? (gate for P3)

For each held-out game, the simulator's predictive distribution of each game-level summary produces a randomised PIT of the realised value. Tolerances are pre-registered (`hoopslab.diagnostics.realism.TOL`):

| Criterion | Practical tolerance | Widened by (n = games) |
|---|---|---|
| 80% central coverage | 0.75–0.85 | ± 2·√(0.16/n) |
| PIT dispersion ratio Var(PIT)·12 | 0.80–1.25 | ± 2·0.894/√n |
| Mean PIT | 0.45–0.55 | ± 2·√(1/(12n)) |

At one NBA season (n ≈ 1230), the widening is +0.023 / +0.051 / +0.016.

Summaries checked:

- possessions,
- team points, total points, margin,
- lead changes,
- P(OT) reliability,
- team FTA, OREB, AST, TOV, PF,
- starters' total minutes, maximum player minutes,
- number of players used,
- substitution count and timing distribution (level C),
- top-scorer points,
- usage-share distribution (top player's share of team usage),
- blowout frequency (|margin| ≥ 20) and starters' minutes conditional on blowouts.

Every summary is also compared as a distribution across games: simulated vs real histograms with a two-sample KS or energy distance, flagged if p < 0.01 after Holm correction across summaries.

A simulator must pass all gates on the development window before its forecasts enter any hypothesis test. A failure is diagnostic information, and it is reported.

---

## 4. Reading results: the scorecard

For each hypothesis the scorecard reports, in this order:

1. the pre-registered expectation (`04`);
2. the effect estimate, its CI, the Holm-adjusted p-value, and the MPE;
3. the decision (SUPPORTED / PROMISING / INCONCLUSIVE / FALSIFIED / HARMFUL);
4. the no-harm check across secondary metrics;
5. calibration diagnostics for both models;
6. the leakage-sensitivity flag (the decision re-run with optimistic knowledge-time imputation);
7. whether any amendment touched this hypothesis, and whether it was post-hoc.

**Anti-metric-shopping rules:**

- No hypothesis is declared supported on a metric other than its primary metric.
- A secondary metric can only produce "HARMFUL on X" warnings.
- Stratified or per-target results are labelled exploratory unless the stratum is pre-registered (H6, H9).

---

## 5. What may change before the freeze vs after

- **Before P6 (development only):** tuning, feature engineering, and adding exploratory experiments, all logged. Adjusting synthetic scenarios is also allowed. This affects instrument calibration, not claims.
- **After the freeze:** MPEs, primary metrics, reference models, strata definitions, lockbox windows, and decision rules are fixed. An amendment is possible but is labelled POST-HOC wherever the affected hypothesis is reported.
