# 09. Synthetic test suite: verifying the instruments before trusting them

*Code: `src/hoopslab/synthetic/world.py` (worlds), `src/hoopslab/synthetic/scenarios.py` (A–H and expected diagnoses), `src/hoopslab/experiments/synthetic_suite.py` (checks). Results: `results/synthetic_suite/seed0.json`. Run: `python -m hoopslab.experiments.synthetic_suite`.*

**This is not evidence about real basketball.** Each synthetic world has a known mechanism. The question is whether the *instruments* (history ablation, usable information, compression, stratified interaction tests, change-point strata, calibration diagnostics, joint scores, realism gates) diagnose that mechanism correctly, before the same instruments are pointed at real data.

## 1. The synthetic world

A vectorised possession engine (`sim/engine.py`) generates complete seasons:

- 8 teams, 12-man rosters with depth-chart slots, 3 seasons × 42 game-days (168 games per season), about 37k possession segments per season;
- player skills: usage, shot mix, make probabilities, FT%, turnover and FT-trip rates, rebound, assist, steal, and block propensities; skill drift across seasons;
- team defence and pace; home advantage; transition effects by possession start type;
- slot rotation with a blowout rule and overtime; injuries with late scratches; questionable-but-available noise;
- optional mechanisms: true momentum, game-level shooting latent, per-game usage form, role changes, star absences, trades, rookies.

Every world emits **canonical data-contract tables** with event and knowledge times (RULESET, TEAM, PLAYER, ROSTER with versioned spells, GAME with schedule/final versions, TEAM_GAME, PLAYER_GAME for all rostered players, INJURY_REPORT snapshots, STARTING_LINEUP, POSSESSION with lineups). The real pipeline (adapter → as-of store → models → backtest → metrics) therefore runs unchanged. Hidden truth tables never reach models (T5 scans for them).

**Known simplifications**, which are gaps in realism and not in the instruments:

- no foul-outs, timeouts, or end-game intentional fouling;
- a fixed rotation rule known to the reference simulator (so synthetic simulator-vs-direct gaps are *larger* than real ones will be);
- no scorekeeper bias;
- the synthetic truth and the reference simulator share structure by design. Misspecification is introduced deliberately (history, latents, pace, blowout rule).

## 2. Scenarios and pre-registered expected diagnoses

| Scenario | Mechanism (truth) | Instrument under test | Expected diagnosis |
|---|---|---|---|
| **A** no history | Markov in the compact state; no latents | E-HIST | `NO_HISTORY_SIGNAL` |
| **B** true history | +0.45 logit on shots after the team scored on its previous possession | E-HIST | `GENUINE_HISTORY` (history residual beyond the latent > 0) |
| **C** latent momentum | per-game team shooting latent, SD 0.55 logit; no causal dependence | E-HIST | `APPARENT_HISTORY_EXPLAINED_BY_LATENT` |
| **A** (again) | a 2,000-level context with no information; start type (real information) | E-SHANNON usable information vs plug-in; placebo | plug-in shows a fake gain; held-out shows none for noise, a positive gain for start type, none for the permuted placebo |
| **A** (again) | – | E-COMP arithmetic coder | lossless; code length ≈ Σ −log₂q; model order preserved |
| **D** role change | 6 bench players' usage × 2.5 on day 8 | as-of CUSUM strata + stratified interaction (recent vs all-history mean) | advantage localised in the alarm stratum; changes detected with a low false-alarm rate |
| **E** star absence | 4 stars out for 14 games | stratified interaction (simulator vs direct), `teammate_star_out` | simulator advantage larger inside the stratum |
| **F** trade | 4 trades: starter ↔ bench player of another team | stratified interaction, `own_move_21d` (2 pooled worlds) | simulator advantage larger for moved players |
| **G** parameter uncertainty | half of every roster replaced by new players each season | L4 posterior-predictive vs L4-plugin, `low_sample` stratum | plug-in 80% coverage significantly below nominal; posterior-predictive closer |
| **H** event OK, tails wrong | per-game shooting latent SD 0.40 | E-TAIL + realism gates | L10 (no latent) fails team-points dispersion gates; L11 passes; the per-event latent signal is small in bits |
| **A** (joint) | teammates' minutes coupled by rotation, blowouts, OT | E-JOINT: joint samples vs identical-marginal independent copula; E-REALISM gates | identical marginal CRPS; energy score prefers joint; pair-τ and blowout coupling match reality; gates pass the correct simulator and fail a misspecified one (pace × 0.9, no blowout rule) |

## 3. Results (seed 0, frozen with this blueprint)

See `results/synthetic_suite/seed0.json` for all numbers. Summary:

All 11 checks: **PASS** (`results/synthetic_suite/seed0.log`, about 10 minutes on 4 CPUs).

| Check | Diagnosis | Key numbers |
|---|---|---|
| History, A | NO_HISTORY_SIGNAL | history signal −0.06 millibit/event, CI [−0.45, +0.38]; 37k test events |
| History, B | GENUINE_HISTORY | history signal +6.5 [5.1, 7.9]; residual beyond latent +5.7 [4.3, 7.0]; latent signal +0.1 (n.s.) |
| History, C | APPARENT_HISTORY_EXPLAINED_BY_LATENT | history signal +6.5 [4.1, 9.1]; latent signal +21.1 [16.1, 25.8]; residual −0.8 [−1.1, −0.4] |
| Usable information vs plug-in | as expected | plug-in fake gain **0.28 bits** from a pure-noise 2,000-level context; held-out gain −0.40 millibit [−0.81, +0.00]; start type +1.99 [1.29, 2.66]; permuted placebo −0.60 [−0.84, −0.38] |
| Compression | lossless; code ≈ log loss | 6,000 events: state model 15,411 coded bits vs 15,410.2 ideal; intercept model 15,423 vs 15,422.0; order preserved |
| Role change, D | localised; detected | skill of recent-weighted over all-history mean: +25.6% inside the alarm stratum vs −0.7% outside (interaction +26%, CI [19%, 34%]); detection 6/6, false alarms 1.1% |
| Star absence, E | localised | simulator vs L4 (PTS, mode U): +18.5% inside `teammate_star_out` vs +6.2% outside (interaction +12.3%, CI [10.3%, 14.8%]) |
| Trade, F | localised | +17.9% for moved players vs +8.7% others (interaction +9.2%, CI [2.2%, 14.9%]); 2 pooled worlds, 336 stratum units |
| Parameter uncertainty, G | plug-in overconfident | low-sample 80% coverage: plug-in 76.5% (CI upper 78.1%) vs posterior-predictive 78.2%; rest 81.7% vs 82.4% |
| Tails, H | event-OK, aggregate underdispersed | the latent is worth only 9.6 millibits per event (0.4% of 2.55 bits), yet without it team-points 80% coverage is **55.6%** (dispersion 1.55); with it, 81.3% (0.97). Player PTS 90% coverage 86.7% → 89.4% |
| Joint + realism | joint beats independent at identical marginals; gates discriminate | marginal CRPS difference exactly 0; energy skill on team MIN vectors +11.2% [8.7%, 14.2%]; teammates' MIN τ predicted 0.51 vs realised 0.48 (independent: 0.00 vs 0.47); blowout corr(\|margin\|, starters' min) predicted −0.75 vs realised −0.70; correct simulator passes all gates, misspecified simulator fails home points and total points (80% coverage 0.53 / 0.44) |

Two results worth carrying into the real-data phase:

- **(G)** In this world the parameter-uncertainty effect is 1.7 percentage points of coverage, *below* the H9 MPE of 3 pp. If reality is like this world, H9 will be FALSIFIED or INCONCLUSIVE. The instrument detects the direction but not a practically relevant size.
- **(H)** The "event-calibrated but aggregate-wrong" signature is large. A mechanism worth < 1% of per-event bits drove team-total coverage from 80% to 56%. This is exactly why H5 (compression ↔ distribution quality) must be tested rather than assumed.

## 4. Instrument calibration notes (changes made BEFORE the freeze, on synthetic data only)

Everything below was decided on synthetic worlds before any real data existed. Each change is recorded because it shaped the frozen design.

1. **Scenario C was initially underpowered.** At latent SD 0.35 the diagnosis was `INCONCLUSIVE` (history signal 0.47 millibit, below the 0.5 MPE; latent signal 3.9 millibits). That is a correct non-claim, but it has no power to demonstrate the "apparent momentum" diagnosis. SD was raised to 0.55.

   *Implication for real data:* with ~37k test events, CIs are ±0.7 millibit. A real NBA season has roughly 250k segments (±0.3). Diagnoses near the MPE may be INCONCLUSIVE, and that is an acceptable outcome.
2. **CUSUM role-change detector.** The default (k = 0.5, h = 5) detected all changes but had a ~30–40% false-alarm rate. Frozen at k = 0.75, h = 6, warm-up 10: 100% detection, 1–4% false alarms on two worlds.
3. **H6c stratum.** The team-level `roster_change_21d` stratum diluted a player-level effect: interaction +2%, CI [−7%, +7%], with 6 of 8 teams in the stratum. `own_move_21d` is now the primary H6c stratum.

   The first trade scenario (star-for-starter swaps) produced a true localised effect of only ~2%, even with three pooled worlds. The instrument was fine; the mechanism was weak. The scenario now uses starter ↔ bench trades, a role change through a trade.

   *Implication for real data:* H6c may be underpowered, because trades are rare. The registry keeps its MPE, and INCONCLUSIVE is a legitimate outcome.
4. **Pair-dependence instrument.** Randomised-PIT jitter destroys realised dependence when marginals are concentrated on a few values (minutes). The first version compared realised τ on jittered PITs with predicted τ on raw samples (0.45 vs 0.83 for a *correct* model). Fixed: predicted τ is now computed on the same jittered-PIT scale (0.51 vs 0.48).
5. **H8 primary metric.** With identical marginals and true within-team minutes dependence, the **energy score** detected the joint structure (+10.7%, CI [8.1%, 13.7%]). The **variogram score** (p = 0.5) did not (−2.8%, CI [−13%, +7%]). The energy score is therefore the H8 primary and the variogram score secondary. PRA dispersion was uninformative in this world (P, R, and A are nearly independent given minutes), so it remains a diagnostic.
6. **Realism gates caught three genuine bugs in the reference simulator.** None were fixed by loosening tolerances:
   - (a) pace estimated from segments truncated at period ends: +1.7 possessions per game;
   - (b) home advantage halved;
   - (c) home advantage double-counted in player make probabilities: +2.5 points per team.

   All were fixed at the source. The gates then passed the correct simulator and failed the misspecified one.
7. **Plug-in margins are too narrow.** Even without latents, the plug-in simulator's margin forecasts were under-dispersed, because estimation error in team strength is ignored. That is the H9 mechanism. The "correct" simulator in the realism check therefore carries parameter uncertainty (player make rates, usage, team defence). Margin dispersion (≈ 1.15) is still slightly narrow and passes only within the sample-size widening. Worth watching on real data.
8. **Sample-size-aware gates.** With n = 144 games, the fixed tolerance (±0.05 coverage) was tighter than sampling noise (SE ≈ 0.033). The gates now widen each tolerance by 2 sampling SEs, which is +0.02–0.05 at a real season's n.
9. **Leakage-test design.** T1/T2 cannot see side channels (a model holding the full dataset outside the factory). This is demonstrated in a test. It led to the `factory(store)` rule and the static scan T5.

## 5. Pre-registered instrument expectation (to be checked after the freeze)

- Every check passes at seed 0 (frozen record above). I expect ≥ 4 of 5 alternate seeds (1–5) to pass each check (strong prior; `registry/expectations.yaml`).
- Borderline checks by design: C (effect near MPE), G (small coverage gap), F (few traded players). Those are where a seed failure would be least surprising.

Alternate-seed results will be appended as a post-freeze report, without changing the checks.
