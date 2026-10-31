# NBA Games as Probabilistic Sequences: What Holds Up, What Doesn't, and What I Would Build

*A first-principles look at whether ideas from information theory, probability, statistics and machine learning can be combined into a better system for forecasting NBA player statistics and game outcomes.*

Supporting numerical toy demonstrations: [`experiments/toy_demos.py`](../../experiments/toy_demos.py) (output in [`experiments/toy_demos_output.txt`](../../experiments/toy_demos_output.txt)). Those demos use made-up but realistic parameters, not real NBA data. They show structural points and are not empirical estimates.

---

## 0. Verdict up front

1. **Treating a game as a sequence isn't a hypothesis. It's a factorization.** By the chain rule of probability, *any* joint distribution over a game can be written as a product of next-event conditionals. So "a game is a probabilistic sequence" can't be true or false. What *can* be true or false are the specific claims: (a) a small state is enough to predict the next event, (b) conditional models at the event level are easier to learn than a direct model of the box score, and (c) errors don't compound when you simulate forward. Those three claims are what need testing.

2. **The idea isn't new, and the existing work is instructive.** Markov possession models of basketball go back at least to Shirley (2007) and Štrumbelj & Vračar (2012). The most sophisticated version, Cervone, D'Amour, Bornn & Goldsberry's *Expected Possession Value* (JASA 2016), is a multiresolution semi-Markov model on optical tracking data with competing-risk hazards for passes, shots and turnovers. It is very close to the "possession as a sequence" idea at its best. Franks et al. (2015) used a hidden Markov model to infer defensive matchups from tracking data. You would be building on this literature, not opening a new field.

3. **For *pre-game* player box-score distributions, I expect possession-level simulation to give only small average gains over a strong direct distributional model.** Most of the predictable variance sits in slow variables: availability, minutes, role, and estimates of true talent. Most of the rest is irreducible shot-making noise. Neither of those lives in within-game sequence structure. This is my prediction, not a known fact, and Section 17 gives experiments that could prove it wrong.

4. **The simulation approach does have real, specific advantages:**
   - **Coherent joint distributions.** Players on a team share 240 minutes, one ball and one scoreboard. Blowouts cut starters' minutes together.
   - **Compositional generalization.** It handles lineup configurations that never appeared in the data, such as a star being out or a player arriving by trade.
   - **Live in-game forecasting.** The model's state simply carries forward during a game.
   - **Tails.** It gets tail probabilities right, but *only* if game-level latent effects are modeled explicitly (see Demo B).

   These advantages are narrower than "model the whole game like a language model."

5. **Where the connections are real and where they're analogy:**
   - **Mathematically exact:** log loss = code length; next-token prediction = the chain rule plus cross-entropy; Elo = stochastic gradient descent on logistic log loss; Kalman filter = Elo with an uncertainty-adaptive step size; absorbing Markov chains = exact possession values.
   - **Mostly analogy or misleading:** "basketball is a language," "more parameters will help," "low entropy means exploitable," and the perceptron as anything more than an ancestor.

6. **The most important concepts missing from the original framing:**
   - hierarchical Bayesian shrinkage (partial pooling)
   - proper scoring rules and calibration
   - latent game-level effects and exchangeability, which cause overdispersion and fake "hot hands"
   - compositional constraints (minutes and shot usage must add up)
   - competing-risk hazard models
   - causal inference for counterfactual lineups and confounded matchup data
   - statistical post-processing of simulator output, as weather forecasters do (Model Output Statistics)
   - game theory: predictability is not the same as exploitability

7. **The system I would build** is a *multi-timescale hierarchical generative model with statistical post-processing*, closer in spirit to numerical weather prediction than to a language model:
   - a slow latent-skill layer, filtered over days and seasons
   - an availability-and-rotation layer
   - a possession-level competing-risks engine
   - game-level random effects
   - Monte Carlo over *parameters and games*
   - a calibration and blending layer that combines the simulator with direct models and is scored by proper scoring rules in a rolling, as-of-date backtest

---

## 1–2. Which historical ideas are genuinely useful, and which connections are real vs. superficial

### Historical corrections first

- **Shannon and Markov are already linked.** Markov's 1913 paper analyzed the vowel/consonant sequence in about 20,000 letters of Pushkin's *Eugene Onegin*. That was the first application of Markov chains, and it was to text. Shannon's 1948 paper (*A Mathematical Theory of Communication*) built English approximations from *n*-gram Markov models of letters and words. The 1951 paper (*Prediction and Entropy of Printed English*) is the one with the human guessing game. It measured **letters** over a 27-symbol alphabet, not words, and bracketed English at roughly 0.6–1.3 bits per letter with long context. So "information theory + Markov chains" isn't a novel combination; it is the founding combination.
- **The perceptron (Rosenblatt 1957/58) isn't a probabilistic model.** It outputs a hard 0/1 decision. The probabilistic relative is logistic regression, which is itself a generalized linear model (Nelder & Wedderburn 1972). The softmax output layer of a modern language model is multinomial logistic regression. The lineage that matters here runs from logistic regression and GLMs, through backpropagation (Werbos 1974; Rumelhart, Hinton & Williams 1986), to deep networks. The perceptron is a historical ancestor, not a working component.
- **PageRank (1998) is a *stationary distribution*, not a next-state predictor.** It answers "where does a random walk spend its time?" That is a useful idea for ranking. Kvam & Sokol's logistic-regression Markov chain (LRMC) ranks college teams this way. But it is a different use of Markov chains from sequence prediction.
- **Language models work because of scale that basketball doesn't have.** Frontier language models train on roughly 10¹³ tokens. All publicly available NBA play-by-play since 1996–97 comes to roughly 10⁷ events: about 500 events per game × 1,230 games × ~30 seasons. That's a gap of about a million. Rough compute-optimal heuristics from language modeling (tens of tokens per parameter) would cap a play-by-play model at around 10⁵–10⁶ parameters. It's even less in practice, because the data isn't stationary. "Large numbers of learned parameters" is the part of the LLM story that transfers *least* well.

### Scorecard

| Idea | Verdict | Why |
|---|---|---|
| **Log loss / cross-entropy** | **Exact and central** | The strictly proper scoring rule for probability forecasts. Averaged over data, it estimates cross-entropy H(p, q) = H(p) + KL(p‖q). Since H(p) is fixed, ranking models by log loss ranks them by KL divergence from the truth. |
| **Compression ↔ prediction** | **Exact** (Kraft inequality; arithmetic coding) | A model assigning probability q(x) can encode x in −log₂ q(x) + 2 bits. Better log loss equals shorter code. This is identical to likelihood-based evaluation, not an extra piece of evidence (Section 9). |
| **Chain rule / next-token factorization** | **Exact** | p(x₁…x_T) = ∏ p(x_t ∣ x_<t). Every generative game simulator is an instance. |
| **Entropy, conditional entropy, mutual information** | **Real, but hard to estimate** | I(A; C) = H(A) − H(A ∣ C) is exactly the log-loss gain from knowing C. Plug-in estimates are badly biased with sparse contexts (Demo D). |
| **Markov chains** | **Real at the possession level, with a well-chosen state** | Absorbing-chain algebra gives exact expected possession values: N = (I − Q)⁻¹. Whether the Markov assumption holds depends entirely on the state (Section 7). |
| **Hidden Markov models** | **Real for specific latents** | Defensive matchups (Franks et al. 2015), coverage schemes, and possibly within-game "form." Often over-applied elsewhere. |
| **Elo** | **Exact link to SGD** | The Elo update R ← R + K(S − E), with E a logistic function of the rating difference, *is* one stochastic-gradient step on logistic log loss. (With E = 1/(1+10^(−Δ/400)), K is the learning rate up to a factor of ln 10/400.) |
| **Kalman filter** | **Exact and better than Elo** | Same error-correcting form θ ← θ + K_t(y − ŷ), but the gain K_t comes from current uncertainty. New players, returns from injury and post-trade periods get big updates; established veterans get small ones. Glicko (Glickman 1999) and TrueSkill (Herbrich et al. 2006) are approximate Bayesian filters of this kind. DARKO, a public NBA player projection system, uses Kalman-style filtering, as its name says. |
| **Perceptron** | **Mostly superficial** | Its useful lessons are the error-correction rule (shared with LMS, logistic SGD, Elo and Kalman) and the Minsky–Papert result that single-layer linear models can't represent XOR-type interactions. That second lesson is directly relevant: linear lineup models (RAPM) can't capture effects like "two non-shooters together are worse than additive." |
| **Attention / transformers** | **Real in two specific places** | (1) Who uses a possession is a softmax over the five players on the floor with context-dependent logits. That is structurally an attention distribution, and the attention weights literally *are* usage shares. (2) Multi-agent tracking data, where the data volume is real. Alcorn & Nguyen's *baller2vec* (2021) trains a transformer on NBA player trajectories. For play-by-play-level features I expect little gain over gradient-boosted trees and generalized linear models. That is testable (E2). |
| **Recurrent networks** | **Diagnostic more than production** | Useful for measuring how much history matters beyond a hand-built state (E2). |
| **Reinforcement learning** | **Mostly unnecessary for prediction** | Prediction needs the *behavior* policy (what coaches and players *will* do), not the optimal one. RL's value function is essentially possession value (EPV), which you get without RL. Inverse RL (inferring a coach's objective) is interesting but second-order. |
| **Monte Carlo** | **Exact as a tool** | It computes marginals of the joint. It is not a source of accuracy in itself (Section 11). |
| **"Game = language"** | **Superficial** | Language has a huge, fairly stationary corpus and a mostly non-adversarial generator. Basketball has a tiny corpus, drift in rules and rosters, and strategic opponents who respond. |
| **Shannon's guessing game** | **Surprisingly apt, reframed** | Shannon used *human predictors* to bound entropy. The modern equivalent is the betting market: an aggregated human-plus-model predictor whose closing prices bound how predictable outcomes are from public information. That makes it the right scientific benchmark, not a target. |

---

## 3. Concepts missing from the original framing

In rough order of how much they would matter:

1. **Hierarchical Bayes / partial pooling / empirical Bayes.** The central statistical fact in sports is small samples of noisy rates. James–Stein (1961) and Efron & Morris (1975), whose worked example was baseball batting averages, show that shrinking individual estimates toward group means beats raw averages. Actuaries have used the same idea since Bühlmann (1967) under the name *credibility theory*. Every rate in the system (3P%, usage, rebound rate, foul rate) should be a partially pooled estimate with its own reliability.
2. **Proper scoring rules and calibration** (Gneiting & Raftery 2007). You can't evaluate distributional forecasts without them.
3. **Latent game-level effects and exchangeability.** By de Finetti's theorem, exchangeable shots look like independent shots *given* a game-level latent. Omitting that latent gives individually calibrated events but *under-dispersed* totals (Demo B), and it creates fake "hot hands" when shots are pooled (Demo E).
4. **Compositional constraints.** Team minutes sum to 240 (plus overtime). Shot usage shares on each possession sum to 1. Possessions are nearly equal for the two teams. Per-player direct models ignore this, so their joint forecasts are incoherent.
5. **Competing risks and hazard models.** A possession ends in exactly one way: shot, turnover, foul or end of period, with a shot-clock-dependent hazard. Substitutions, foul-outs and injury exits are time-to-event processes. This is the standard statistical form for the "possession engine," and it is exactly the form Cervone et al. use.
6. **Causal inference.** Two places matter most:
   - *Counterfactual lineups.* "What happens to Player X's usage when the star sits?" is a causal question with confounding: stars sit in back-to-backs and in blowouts.
   - *Matchup data is confounded.* Coaches assign defenders strategically. Raw "Player X vs Defender Y" splits are biased.
7. **Measurement error.** Play-by-play is a noisy channel. Assists and blocks are scorekeeper judgments, reportedly with arena-to-arena variation. Substitution timestamps and team rebounds have errors.
8. **Nonstationarity and the Lucas critique.** Rules change (for example, the 2018–19 reset of the shot clock to 14 seconds after offensive rebounds, and the 2022–23 transition take-foul rule). Coaches change, and opponents adapt. A model fitted to behavior can break when agents respond to it (Lucas 1976).
9. **Sufficient statistics / information bottleneck / "causal states."** These give a principled definition of what a "state" is (Section 6).
10. **Forecast combination and post-processing.** Stacking and weather-style Model Output Statistics usually beat any single model.
11. **Posterior predictive distributions.** Forecasts must integrate over parameter uncertainty, not only game randomness (Demo C).
12. **Game theory.** In a mixed-strategy equilibrium, being predictable is not exploitable. Section 8 makes the entropy question much sharper because of this.
13. **Variance decomposition and irreducible noise.** Before building anything, ask how much of the target's variance is predictable at all (Section 3a).

### 3a. The information budget: why expectations must be modest

**Possession level.** A possession's point outcome (0, 1, 2, 3, 4+) has entropy of about **1.65 bits** (Demo A). How much of that does knowing the teams remove? The spread of team offensive quality is roughly ±3–4 points per 100 possessions. Its mutual information with a single possession's outcome is only about **0.5–1 millibit**. Possession start type (after a made basket, after a defensive rebound, or after a live-ball turnover) carries about **3 millibits**.

Model comparisons therefore happen in the third or fourth decimal place of bits per possession. You need hundreds of thousands of held-out possessions and *paired* tests (for example, Diebold–Mariano) to tell models apart.

**Player-game level.** For a 25-points-per-game scorer, game-to-game standard deviation is typically 7–9 points. By the law of total variance:

  Var(PTS) = E[Var(PTS ∣ minutes, attempts)] + Var(E[PTS ∣ minutes, attempts])

The first term is dominated by binomial make/miss noise. About 20 field-goal attempts at ~50% gives a standard deviation of ~2.2 makes, or roughly 5 points. No architecture removes that term. The second term (minutes, usage, availability, game script) is where models can differ, and most of it is determined by *slow, pre-game* information.

**Game outcomes.** Final margins around a good point-spread forecast have a standard deviation of roughly 11–13 points. Even excellent forecasts pick winners only about two-thirds to 70% of the time.

---

## 4. Architecture (summary; full design in Section 18)

```
 RAW DATA ──► as-of data warehouse (bitemporal: event time + knowledge time)
                │
  SLOW LAYER    ▼   (days → seasons)
  Latent player/team skill state  ── Kalman / EnKF-style filtering, hierarchical priors,
                                     aging curves, translation priors for rookies/imports
                │
  MEDIUM LAYER  ▼   (per game)
  Availability + rotation model    ── P(active), P(start), substitution hazard policy,
                                     240-minute constraint, injury nowcasting
  Game-level latents               ── pace shock, team shooting night, player form, refs
                │
  FAST LAYER    ▼   (per possession)
  Possession engine (semi-Markov, competing risks):
    start type → duration/terminal hazard → user (softmax over 5) → shot zone
    → make prob (player skill × defense × context) → FT / OREB / assist attribution
    → game state update (score, clock, fouls, fatigue) → substitutions → next possession
                │
  SIMULATION    ▼   Monte Carlo over (posterior parameter draw × game-level latent × game)
                │
  POST-PROCESS  ▼   Blend with direct distributional models; PIT recalibration;
                    reconcile player ↔ team totals
                │
  OUTPUT        ▼   Joint samples → marginals, bins, PRA, team totals, win prob
                │
  EVALUATION    ▼   Prequential rolling backtest; log score, RPS, CRPS, PIT, energy score
```

---

## 5. Predict final stats directly, or let them emerge from simulation?

**Math.** If the event-level conditionals were exactly right, the simulated marginals would be exactly right. A direct model p(y ∣ pre-game information) is *also* a consistent estimator of the same marginal. So in the limit of infinite data the two agree. The question is purely one of **bias, variance and extrapolation with finite data**:

| | Direct distributional model | Emergent from simulation |
|---|---|---|
| Learns the marginal variance… | …directly from data. Can't get dispersion wrong through a misspecified dependence structure. | …only as well as the dependence structure is specified. Omitted latents give tails that are too thin (Demo B). |
| Coherence across players and stats | None by default (minutes can sum to 265). Needs copulas or reconciliation. | Automatic. |
| Novel configurations (star out, trade, new coach) | Weak. Few training rows look like the new situation. | Strong. Recombines components that *are* in the data. |
| Error compounding | None | Small biases in usage or pace compound through the rollout. Also *exposure bias*: training on true histories but simulating from generated ones. |
| Live in-game updates | Needs separate models | Native. Condition on the current state and roll forward. |
| Engineering cost | Low | High |

**My answer: neither alone.** Build the generative model at **possession granularity**, not per-touch unless you have tracking data. Put game-level latent effects in it. Then **post-process** its output with direct models, the way meteorologists correct numerical weather prediction with Model Output Statistics (Glahn & Lowry 1972) and EMOS (Gneiting et al. 2005). The physics model supplies structure and coherence; the statistical layer fixes its systematic bias and dispersion errors.

**A correction to the proposed causal chain.** The proposal wrote *minutes → possessions → lineups → touches → …* as though minutes come first. They don't. Minutes are an **outcome** of substitution decisions made in response to in-game events: fouls, score margin, injuries, fatigue. Minutes belong *inside* the loop as an emergent quantity, driven by a rotation policy with a pre-game prior.

Also, public play-by-play never records touches, formations, coverages or defender identity. It records terminal events: shots, turnovers, fouls, rebounds, substitutions, and assists as an attribution. Most of the proposed chain is **unobservable without tracking data**.

---

## 6. What should constitute a "state"?

### The principled definition

A state S_t is good to the extent that it is a **sufficient statistic of the past for the future**:

  p(future ∣ entire history) ≈ p(future ∣ S_t),  equivalently  I(future ; history ∣ S_t) ≈ 0.

This is the "causal state" of computational mechanics (Crutchfield & Young 1989) and the target of the **information bottleneck** (Tishby, Pereira & Bialek 1999): compress the history as far as possible while keeping predictive information. It turns "which variables matter?" into a **measurable** question. Add a variable to the state and measure the held-out log-loss reduction on the targets you care about. If a flexible sequence model over the full history (RNN or transformer) can't beat your hand-built state on held-out log loss, the state is approximately sufficient for what that model family can extract (Xu et al.'s "usable information," 2020).

### My prior ranking (to be confirmed by exactly that test)

**Tier 1: clearly matters, and is observable in public data**

- **The 10 players on the floor**, represented through player skill vectors or embeddings, not player IDs. This drives who shoots, efficiency, rebounding and pace.
- **Possession start type**: after a made field goal or free throw, a defensive rebound, a live-ball turnover, an offensive rebound, or a dead ball / after timeout. Transition possessions after steals are far more efficient than half-court possessions.
- **Game clock and period, jointly with score margin.** This covers end-of-quarter heaves, end-game intentional fouling, garbage time and the pace behavior of a leading team.
- **Team fouls / bonus status**, and **individual personal fouls** (which drive substitution hazards; "two fouls in the first quarter" rotation behavior).
- **Home/away** plus game-level context: rest days, back-to-backs, travel.

**Tier 2: matters, but partly or noisily observable**

- **Fatigue proxies**: time on court in the current stint, cumulative minutes. These mostly work *through substitution decisions*. The direct efficiency effect is probably small.
- **Shot clock**, within possessions. Needed for the duration and terminal-event hazards. Play-by-play only gives it approximately.
- **Matchup assignments** (who guards whom). Tracking-derived. Public NBA.com matchup data exists, but it is aggregated and confounded.

**Tier 3: my prior is that the gain is small once Tiers 1–2 are in (test before investing)**

- **Previous possession outcomes / "momentum."** The evidence says small (Section 7).
- **Offensive formation, defensive coverage.** These require tracking or video tagging. They are valuable for tactical analysis. For *box-score* forecasts they matter mostly through things already captured: who shoots, from where, and how efficiently. Where they help is in estimating *skill* more precisely (shot quality), which feeds the slow layer.
- **Ball-handler identity per action.** Tracking only. Largely summarized by the "who uses the possession" softmax.

**A key design choice:** player "tendencies" are **not state**. They are **parameters**, or slow latent states on the day-to-season timescale. Mixing timescales is a common modeling error. Keep a fast within-game state (lineup, score, clock, fouls, stint) separate from a slow between-game state (skill, role, health).

---

## 7. How much history matters. Is basketball Markov?

"Is basketball Markov?" is ill-posed: every process is Markov in its full history. The real question is whether it is Markov **in a small state**, and the answer differs by timescale:

| Timescale | Structure | Is Markov adequate? |
|---|---|---|
| Frames (25 Hz tracking) | Physics: position, velocity, acceleration | Yes, at second order (positions plus velocities). This is the trajectory-model regime. |
| Within a possession (0–24 s) | Semi-Markov: shot clock, where the ball has been | Yes, with elapsed time and a few summaries. Cervone et al. model it as semi-Markov with hazards. |
| Across possessions in a game | Weak direct dependence. Strong dependence *through slow state variables*: lineup, fouls, score/time, fatigue | **Yes, in an augmented state**, plus a **game-level latent**, which is *not* Markov in observables. |
| Across games | Latent skill, role, health | Hidden Markov / state-space model: Markov in the latent, not in observed box scores. |
| Across seasons | Aging, development, role changes | State space with drift (aging curves), plus change-points. |

**The evidence on memory in observed scoring:**

- **Scoring events.** Gabel & Redner (2012) and Merritt & Clauset (2014) found NBA scoring events close to a Poisson process. They reported weak *anti*-persistence (a basket slightly lowers the chance the same team scores next, largely from possession alternation) and a small restoring force toward a tied score. Stern (1994) modeled the score margin as Brownian motion with drift, and it works well.
- **Hot hand.** Gilovich, Vallone & Tversky (1985) found none. Miller & Sanjurjo (2018) showed their within-sequence estimator is biased *against* finding one (reproduced in Demo E). Corrected analyses find a real but **small** effect. Bocskocsky, Ezekowitz & Stein (2014) found that after adjusting for shot difficulty, "hot" players take harder shots and make them at a slightly higher rate than expected. This is a percentage-point-level effect.
- **Game-level heterogeneity mimics momentum.** Demo E: a pure game-level latent with *no* causal dependence between shots produces P(make ∣ previous make) − P(make ∣ previous miss) ≈ +0.02 when games are pooled. A good model must separate (i) a static per-game latent, (ii) a slowly varying within-game latent (HMM-like), and (iii) true state-dependence on the last outcome. Those three hypotheses make different predictions and can be compared by held-out likelihood.

**What I'd expect:** a Markov model on (lineup, start type, score margin × time, fouls, bonus, stint length) plus a game-level random effect captures nearly all *usable* predictive information across possessions. Longer history mostly acts as a noisy proxy for things better modeled explicitly: fatigue, substitution patterns, and the game-level latent. Experiment E2 tests this directly.

---

## 8. Can entropy reveal exploitable predictability?

**What is correct.** H(A ∣ player) − H(A ∣ player, context) = I(A; context ∣ player) is exactly the reduction in expected log loss from knowing the context. It is a principled way to rank which context variables carry information about a player's next action (play type, shot zone, drive direction, pass target).

**Four serious problems:**

1. **Estimation bias.** Plug-in conditional entropy with many cells manufactures predictability from noise. In Demo D, with 2,000 observations of a *completely random* 4-way action across 500 contexts, the plug-in estimate shows a 0.65-bit "reduction." Held-out cross-entropy shows none; it gets *worse*. **Always estimate entropy as held-out cross-entropy of a regularized predictive model**, or use bias-corrected estimators (Miller–Madow; Nemenman–Shafee–Bialek). Treat any reported reduction as a lower bound on usable information for that model class.
2. **Constraint is not exploitability.** Entropy collapses at 2 seconds on the shot clock because the offense *has* to shoot, not because the defense has an edge.
3. **Payoff matters, entropy doesn't.** Entropy equals the value of information *only when the payoff is log loss*. For a defense, the value of knowing the context is:

   VOI = E_c[ max_d E(−points ∣ d, c) ] − max_d E(−points ∣ d)

   Here d is a defensive response. VOI depends on the payoff differences between responses, not on entropy. A perfectly predictable action that the defense can't stop has zero exploitable value.
4. **Game theory.** If the offense plays a mixed-strategy equilibrium, the defense *can't* profit even from knowing the offense's mixing probabilities exactly. Only **deviations from equilibrium** are exploitable. The right test isn't "low entropy." It is the **indifference condition**: in equilibrium, every action used with positive probability has equal expected value. Walker & Wooders (2001, tennis serves) and Palacios-Huerta (2003) and Chiappori, Levitt & Groseclose (2002) (soccer penalties) used this test. In the NBA, it is the question of whether marginal points per shot are equalized across shot types and players, i.e. allocative efficiency (e.g., Goldman & Rao's work).

**A better diagnostic stack:**

- (a) *Usable information*: held-out log-loss gain from context.
- (b) *Payoff inequality*: in contexts with high usable information, do the actions the offense uses have unequal expected values? That signals a departure from equilibrium.
- (c) *Response evidence*: do defenses that plausibly responded (different coverage, tracking-derived) actually lower efficiency in those contexts? That is a causal question.
- (d) *Out-of-sample persistence*: does the pattern survive into the second half of the season, after opponents have had film to adapt?

Entropy is a good **screening tool** for step (a). Treating it as a conclusion would be a mistake.

---

## 9. Compression as model evaluation (MDL)

**What holds exactly.** A model assigning probability q to a sequence can arithmetic-code it in −log₂ q + O(1) bits. So "model A compresses possession sequences better" is *identical* to "model A has lower total held-out log loss."

**What MDL adds:**

- **Prequential evaluation** (Dawid 1984): sum −log q(x_t ∣ x_<t, model refitted only on data before t) over the season in time order. That total is a valid code length. It automatically charges for model complexity without counting parameters, and it is exactly the right backtesting protocol for a nonstationary sport. The Bayesian marginal likelihood is the prequential code of the Bayesian mixture. Rissanen's two-part MDL (1978) is an alternative when you want an explicit complexity cost.
- **A single currency.** Every component can be scored in bits: shot-make model, rebound model, substitution model. That makes it clear *where* a model improves.

**Where it misleads:**

1. **Different random variables can't be compared.** You can't compare "bits to code the play-by-play" for a simulator against "bits to code the box score" for a direct model. They code different objects. To compare them *on the target*, score both on the **same** variable: −log q(box score), with the simulator's q obtained by marginalizing through simulation (smoothing for tail bins).
2. **Bits go where the entropy is, not where you care.** Most of the ~1.65 bits per possession are make/miss coin flips and rebound attributions that are almost irreducible. A model can gain meaningfully on "who gets the defensive rebound" and learn nothing useful about scoring distributions. Experiment E3 tests whether event-level compression gains translate into box-score gains at all.
3. **Calibrated parts don't make a calibrated whole** (Demo B). A model missing a game-level latent loses only a few millibits per shot in log loss, since the upper bound on the achievable gain is about 6 millibits. Yet it underestimates P(35+ points) by roughly 30% in relative terms. Per-event compression is **insensitive to dependence structure**, and dependence structure is exactly what determines aggregate distributions.
4. **Deterministic bookkeeping is free.** After a made basket the ball changes hands. Tokenization choices can inflate or deflate apparent compression without changing any real forecast.

**Bottom line:** use prequential log loss (= code length) as the backbone metric for *each component*. Decide *architecture* questions on proper scores of the final targets.

**A related Shannon-lineage result worth knowing** (Kelly 1956; Cover & Thomas, ch. 6): for a log-optimal bettor, the expected growth rate of wealth against a market equals the reduction in log loss relative to the market's implied probabilities. This is why the market is a natural *information-theoretic benchmark*: beating its log loss is the same thing as having information it doesn't. You don't have to bet for this to be a useful measuring stick.

---

## 10. How to represent uncertainty

**Principles:**

- **Report posterior predictive distributions.** These integrate over (i) game randomness, (ii) game-level latents, and (iii) *parameter* uncertainty. Leaving out (iii) makes forecasts overconfident. That matters most for rookies, newly traded players, players returning from injury, and small-sample roles.
- **Keep two versions of every player distribution:** unconditional, and conditional on playing. Availability is a separate probability, and mixing the two silently is a common error.
- **Represent each counting stat as a full discrete probability mass function** over integers (e.g., points 0–80), not a parametric summary. Bins such as P(<20), P(20–24), … are derived from it.
- **Represent the joint distribution as samples**, the way weather ensembles are: an array of simulations × players × stats. PRA, P+A, double-doubles, team totals, margin, and "Player X ≥ 30 *and* team wins" are then computed consistently from the same draws.

**The shapes differ by stat, which is another reason not to force one family:**

- **Points:** lumpy (2s, 3s, free throws), over-dispersed relative to Poisson, and right-skewed. The natural form is a compound distribution: Σ (attempt value × make). Actuarial collective-risk models compute such compound distributions exactly with Panjer's recursion. That is a useful fast *direct* alternative or cross-check to simulation.
- **Rebounds, assists, turnovers, 3PM:** near Poisson, often over-dispersed (negative binomial). Sometimes under-dispersed; Conway–Maxwell–Poisson handles both.
- **Minutes:** a mixture of DNP / early injury exit / foul trouble / normal rotation / blowout rest / overtime. It is strongly multimodal for bench players.

**Direct-model forms that output full distributions:**

- a discrete softmax over integer outcomes (structurally a language-model output head over "point tokens"; train with ranked probability score or with ordinal smoothing so that 29 counts as close to 30)
- distributional gradient boosting (NGBoost; negative binomial / zero-inflated heads)
- quantile regression with monotone rearrangement
- mixture density networks

**Coverage guarantees:** conformal prediction (Vovk et al. 2005) wrapped around any of these gives finite-sample interval coverage under exchangeability. Exchangeability is only approximate in time series, so use rolling or weighted conformal methods.

---

## 11. Can possession-level Monte Carlo generate these distributions?

Yes, and it's cheap. A slate of 15 games × 20,000 simulations × ~200 possessions is about 60 million possession steps. Vectorized code on a single machine handles that in seconds to minutes. Four important caveats:

1. **Monte Carlo error is almost never the bottleneck.** Standard error of a probability near 0.3 is 0.014 at 1k simulations, 0.005 at 10k, and 0.0014 at 100k (Demo C). By contrast, a one-standard-deviation error in a player's true scoring mean (plausibly ±1.5 points) moves P(≥30) by about ±7 percentage points. **Hundreds of thousands of simulations with fixed parameters is precision spent on the wrong term.** Draw parameters from the posterior *per simulation*. For extreme tails (e.g., P(50+)), use importance sampling rather than brute force.
2. **Game-level latents must be drawn once per simulated game** (form, team shooting night, referee crew, pace shock). Without them, totals are under-dispersed even when every event is calibrated (Demo B: the nominal 80% interval covers ~78%, and P(35+) is 0.079 against a true 0.110).
3. **Simulated behavior must include strategic and administrative dynamics:** rotations, foul trouble, end-game fouling, garbage-time substitutions, overtime. Otherwise the simulator gets star minutes in lopsided games wrong. That is the single biggest joint effect on player stats.
4. **Exposure bias.** Components trained on real histories are run on simulated ones. If simulated states drift (e.g., unrealistic foul accumulation), later conditionals are evaluated off-distribution. Monitor simulated vs. real summary statistics: minutes distributions, pace, foul counts, free-throw rates, lead-change distributions.

---

## 12. Evaluation

**Scoring rules** (all strictly proper unless noted; see Gneiting & Raftery 2007):

| Metric | Use | Notes |
|---|---|---|
| **Log score** (log loss, cross-entropy) | Binary and categorical events; per-component event models; integer pmfs | Local. Harsh on assigning near-zero to what happens, so it needs smoothed tails for simulated pmfs. Equals code length. |
| **Brier score** | Binary events (win; over/under a threshold) | Murphy (1973) decomposition into reliability − resolution + uncertainty is very useful for diagnosis. |
| **Ranked probability score (RPS)** | Ordered bins (e.g., the five point ranges) | Respects ordering; log loss over bins doesn't. |
| **CRPS** | Full distributions of integer or continuous stats | Generalizes absolute error: for a point forecast, CRPS = MAE. Threshold-weighted CRPS (Gneiting & Ranjan 2011) emphasizes tails. |
| **Energy score / variogram score** | *Joint* forecasts (PRA, teammates, player vs. team) | The only way to score the correlation structure. Variogram score (Scheuerer & Hamill 2015) is more sensitive to correlation errors. |
| **MAE / RMSE** | Point summaries only | Use MAE for the *median* and RMSE for the *mean*. Each scoring function is consistent for one functional (Gneiting 2011). Mixing them rewards the wrong point forecast. |

**Calibration and sharpness.** Aim for "maximize sharpness subject to calibration" (Gneiting, Balabdaoui & Raftery 2007).

- **PIT histograms**, randomized for count data (Czado, Gneiting & Held 2009). Flat means calibrated; U-shaped means too narrow; hump-shaped means too wide; sloped means biased.
- **Reliability diagrams** for binary events and derived thresholds.
- **Central-interval coverage** at 50/80/95%.
- **Stratify everything**: starters vs. bench, minutes bucket, rookies, post-trade, return from injury, a teammate out, back-to-backs, blowouts, playoffs.

**Backtesting protocol:**

- **As-of data only.** Use injury status, starting lineups and ratings *as they were known* at forecast time. This needs a bitemporal data store. The most common leakage sources are season-end ratings, final injury status, and actual starting lineups.
- **Rolling origin:** refit or filter daily, forecast the next day, never look ahead. The total log score over the season is the prequential code length.
- **Paired significance testing** (Diebold–Mariano, with standard errors clustered by game date). Differences are tiny relative to noise (Section 3a).
- **Skill scores** against baselines: climatology (league or player average with naive dispersion), persistence (last-N average), a strong direct model, and the betting market where it exists (a benchmark, not a goal).
- **Multiple seasons,** including at least one after a rule change.

---

## 13. Data

| Category | What | Public availability | Value |
|---|---|---|---|
| **Box scores** | Per player-game counting stats | Fully public (NBA.com stats API, Basketball-Reference), ~1946 onward with varying detail | Baseline targets; long history |
| **Play-by-play** | Event log: shots, makes/misses, rebounds, turnovers, fouls, substitutions, timeouts; clock, score | Public from 1996–97 (NBA.com). Parsed possession-level datasets with lineups: pbpstats.com and similar | **Core of the possession engine.** Needs cleaning (substitution timing, team rebounds, period-start lineups). |
| **Lineups / stints** | Who was on the floor for every possession | Derivable from play-by-play; public lineup endpoints | Essential for usage, lineup effects, and adjusted plus-minus-style ratings |
| **Shot data** | x/y location, shot type, distance, action type | Public (NBA.com shot chart endpoints) | Shot-quality models; spatial skill (Miller et al. 2014) |
| **Tracking aggregates** | Touches, drives, passes, catch-and-shoot, pull-ups, closest defender distance, speed/distance | Public aggregates on NBA.com since ~2013–14 | Faster-stabilizing skill signals |
| **Matchup data** | Partial possessions of offensive player vs. defender | Public aggregated matchups on NBA.com (roughly 2017–18 onward) | Useful but **confounded and sparse** |
| **Play types** | Synergy-style: pick-and-roll ball handler, isolation, spot-up, … | Aggregates on NBA.com; full event-level data is commercial (Synergy) | Tendency and efficiency by play type |
| **Raw optical tracking** | 25 Hz x/y (and, in the current system, skeletal pose) for players and ball | **Proprietary.** SportVU (all arenas from 2013–14), then Second Spectrum (official from 2017–18), then Sony Hawk-Eye (from 2023–24, as I understand it). About 600 games from 2015–16 SportVU leaked publicly and are widely used in research. | Largest potential improvement for skill estimation, defense, and within-possession modeling |
| **Injury / availability** | Official NBA injury report (published several times daily), transactions, G League assignments | Public, but *timing* is the hard part: you must store when you knew | **Single most important pre-game input for minutes** |
| **Schedule / travel / rest** | Dates, arenas, time zones, back-to-backs | Public | Moderate |
| **Officials** | Crew assignments (announced morning of game) | Public | Small but real (foul rates → free throws, foul trouble) |
| **Betting markets** | Spreads, totals, player props, line movement | Semi-public (historical odds archives, commercial feeds) | **Benchmark** and a sense of "information efficiency"; optionally a feature (kept separate so the scientific model stays independent) |
| **Coaching / roster metadata** | Coach tenure, contracts, draft position, pre-NBA stats (college, international) | Mostly public | Priors for rookies and imports; role-change detection |

**Proprietary data that would materially improve the model,** in my estimated order of value:

1. **Team-internal availability and medical information:** who is really playing and minute restrictions. This dominates minutes error.
2. **Raw tracking:** better shot-quality-adjusted skill estimates, defensive attribution, matchups, fatigue proxies (speed decline).
3. **Event-level play-type tagging with video.**
4. **Wearables / load data** (practice and game load, sleep). Only teams have this.
5. **Coaches' rotation plans.** Essentially unavailable, but the thing that most directly determines minutes.

---

## 14. Biggest practical obstacles

In rough order of impact on forecast error:

1. **Availability and minutes uncertainty.** This is the largest single error source for player-stat forecasts. Late scratches, load management, minute restrictions, and garbage time all hit it. *Mitigation:* explicit availability probabilities, injury-report nowcasting, forecasts conditional on each lineup scenario, and re-forecasting at lineup lock.
2. **Small samples with many parameters.** Five-man lineups and specific matchups mostly have tiny samples. *Mitigation:* build lineup effects from player-level components (embeddings plus low-order interactions) with strong regularization. Never estimate a free parameter per lineup.
3. **Role changes** (trades, teammate injuries, coaching changes). *Mitigation:* the structural model's compositional generalization, plus Bayesian online change-point detection (Adams & MacKay 2007) on usage and minutes.
4. **Confounding.** Matchups are chosen, rest is chosen, garbage time is selected. *Mitigation:* model the selection explicitly (the rotation policy is part of the generative model), and use causal methods where claims are counterfactual.
5. **Concept drift.** Rules, style (the three-point revolution), and officiating emphasis all shift. *Mitigation:* time-decayed estimation, state-space parameters, prequential monitoring and drift alarms.
6. **Garbage time and end-game strategy.** These distort per-possession rates and minutes. *Mitigation:* model them as parts of the state (score × time) rather than deleting the data.
7. **Data quality.** Substitution timing errors, scorekeeper subjectivity on assists and blocks, team vs. player rebound attribution. *Mitigation:* a measurement-error layer; per-arena scorekeeper effects as random effects on assists and blocks.
8. **Tracking data access.** Hard legal and commercial constraint.
9. **Dependence structure.** Game-level latents and teammate correlations are hard to identify, but are required for correct tails.
10. **Computation.** Not a real obstacle for play-by-play-level models. It becomes one only for tracking-based transformers and full Bayesian posteriors at scale. Use variational or Laplace approximations for the nightly filter, and full MCMC for periodic re-estimation of hyperparameters.

---

## 15. Ideas from outside sports analytics that could genuinely help

**Weather prediction (the closest methodological match):**

- **Ensemble forecasting.** Perturb both initial conditions (latents) and model parameters. This is Section 11.
- **Data assimilation.** Ensemble Kalman filters (Evensen 1994) for nightly skill updates on high-dimensional, nonlinear latent states.
- **Post-processing.** MOS, EMOS, and Bayesian model averaging (Raftery et al. 2005) to calibrate simulator output.
- **Verification culture.** Rank histograms, skill scores relative to climatology and persistence.

**Actuarial science:**

- **Credibility theory** (Bühlmann): blending individual and class experience is empirical Bayes shrinkage.
- **Frequency × severity decomposition:** attempts × points per attempt.
- **Panjer recursion:** exact compound distributions without simulation.
- **Copulas for dependent risks.**

**Psychometrics:**

- **Item response theory.** P(make) = σ(shooter ability − shot difficulty), extended with defender and context terms. It is the right skeleton for shot-making.
- **Reliability theory.** Tells you how many attempts each rate stat needs before it means anything.

**Finance and econometrics:**

- **Stochastic volatility:** some players are more *variable*, and variance itself changes over time.
- **Regime switching** (Hamilton 1989) for role changes.
- **Factor models / matrix factorization** for player skill embeddings.
- **Shrinkage covariance estimators** (Ledoit–Wolf) for teammate stat correlations.
- **Tail dependence in copulas.** The Gaussian copula's failure in 2008 is a cautionary tale for PRA and teammate joint tails.
- **Forecast combination** (the "combination puzzle": simple averages are hard to beat).

**Signal processing and control:**

- **Kalman and particle filters** for within-game and between-game latent tracking.
- **Change-point detection.**
- **Model predictive control** as a model of coaching behavior: a coach solves a rolling constrained optimization over rotations.
- **System identification** for fatigue dynamics.

**Physics and statistical mechanics:**

- **Brownian motion for score differentials** (Stern 1994); **random walks with a restoring force** (Gabel & Redner 2012).
- **Maximum entropy** (Jaynes 1957). Given observed marginals (each player's usage, each pair's co-occurrence), the least-assuming joint model is an Ising/Potts pairwise model. Neuroscientists use exactly this for neural populations (Schneidman et al. 2006). It is a principled, parameter-thrifty way to model lineup interactions.
- **Hawkes (self-exciting) point processes** (from seismology) as the proper null model for "runs" and momentum tests.

**Biology and medicine:**

- **Survival analysis and competing risks** (Cox 1972; Fine–Gray) for possession terminations, substitutions, foul-outs and injury exits.
- **Banister's fitness–fatigue impulse-response model** (sports science, 1975) for load and rest effects.
- **Epidemiological nowcasting under reporting delays** for injury-report information that arrives in stages.
- **Dynamic time warping** (from speech recognition) for play recognition in trajectories.

**Operations research:**

- **Rotation as a constrained allocation problem:** 240 minutes, foul constraints, rest targets.
- **Inverse optimization** to infer a coach's implicit objective from observed rotations. This gives a far more robust minutes model than regression on past minutes, especially after roster changes.

**Game theory and economics:**

- **Minimax / indifference tests** for exploitability (Section 8).
- **The Lucas critique** for strategic nonstationarity.
- **Synthetic control / difference-in-differences** for estimating the effect of a teammate's absence or a rule change.

**Language modeling (what *does* transfer):**

- Tokenization design for events.
- Self-supervised pretraining (next-frame or next-event prediction) on tracking data, then fine-tuning for skill estimation.
- Learned embeddings (Alcorn's (batter|pitcher)2vec and baller2vec).
- Scheduled sampling against exposure bias.
- Retrieval: nearest-neighbor "comparable players," as in PECOTA / CARMELO-style projections.

**Communications:**

- The **noisy-channel** view of scorekeeping (a measurement model).
- **Rate–distortion / information bottleneck** for choosing the state.
- **Kelly's information-rate result** for benchmarking against markets.

---

## 16. How this compares with how sophisticated NBA prediction systems are usually built

From public descriptions (much industry work is proprietary, so this is partial):

- **Team and game models:**
  - rating systems (Elo variants, Bayesian or adjusted efficiency ratings, simple rating systems)
  - player-based team strength from projected minutes × player impact metrics (RAPM-derived: RPM, EPM, LEBRON, RAPTOR/DARKO-style)
  - adjustments for rest, travel and injuries, often calibrated against market lines
- **Player projections:**
  - projected minutes × per-minute or per-possession rates, regressed toward priors
  - opponent/pace adjustments
  - distributions from assumed families or empirical residuals
  - simulation with correlations, common in daily fantasy and props tooling
  - modern implementations lean heavily on gradient-boosted trees
- **Research systems:** EPV-style tracking models; spatial shot models; matchup HMMs.

| Component of the proposed architecture | Assessment |
|---|---|
| Minutes × rates core, shrinkage, opponent adjustment | **Conventional.** Keep it; it's the backbone that works. |
| Kalman-style skill filtering | **Conventional in the best public systems** (DARKO-style). Upgrade to multivariate, correlated skills with aging drift. |
| Possession-level competing-risks simulator with lineups | **Exists** (academic Markov sims, proprietary sims) but **uncommon as the primary engine for player distributions**. Potentially advantageous for joint/coherent distributions, injuries and trades, and live forecasting. |
| Game-level random effects to fix aggregate dispersion | **Underused and advantageous.** A cheap fix for the most common calibration failure. |
| Rotation policy as an explicit hazard/optimization model | **Genuinely different.** Most systems forecast minutes directly. |
| Weather-style post-processing / blending | **Genuinely different in sports** and likely the single best return on effort. |
| Entropy / usable-information diagnostics | **Genuinely different** as a research tool for state selection. Not a forecasting component. |
| Prequential log score as the backbone metric | **Uncommon, advantageous.** Many projection systems still report MAE/RMSE. |
| Large transformer over play-by-play | **Unnecessary** at play-by-play data scale (hypothesis E2). |
| Transformer over raw tracking | **Advantageous only with tracking access**, mainly through better skill estimates. |
| RL | **Unnecessary** for prediction. |
| 100k+ simulations per game with fixed parameters | **Unnecessary.** Spend the budget on parameter draws. |

---

## 17. Experiments that could falsify the idea

Each experiment is designed so that a *negative* result is informative. All use the same as-of information for every model. Anything else is an unfair comparison.

**E1: Structural vs. direct on the actual targets.**

- *Hypothesis:* distributions emerging from the simulator beat a strong direct distributional model (NGBoost / quantile gradient boosting with the same pre-game features, including teammate availability) on CRPS/RPS/log score for PTS, REB, AST, 3PM, TOV, MIN and PRA.
- *Protocol:* rolling-origin backtest over ≥3 seasons; Diebold–Mariano test clustered by date.
- *Falsified if:* the simulator isn't better on average, **and** a stacked blend gives it ~0 weight (weight confidence interval includes 0).

**E2: Is a small Markov state sufficient?**

- *Protocol:* next-possession outcome model with (a) the Tier-1/2 state; (b) (a) plus summaries of the last k possessions (k = 1, 3, 10); (c) (a) plus a transformer or RNN over the full game history; (d) (a) plus a game-level random effect updated within the game.
- *Falsifies "history matters" if:* (b) and (c) don't beat (a) by a statistically significant margin in held-out bits per possession, or all their gain is captured by (d).
- *Falsifies "a small state suffices" if:* (c) beats (a) and (d) substantially. Then find out *which* history features it uses, via attribution or probing.

**E3: Do event-level gains transfer to box-score gains?**

- *Protocol:* build 10–20 simulator variants that differ in component quality. For each, measure (i) prequential bits per possession and (ii) box-score CRPS.
- *Falsifies "compression-based evaluation is the right guide" if:* rank correlation across variants is weak or negative.

**E4: Is low entropy exploitable?**

- *Protocol:* flag high-usable-information contexts (held-out) in the first half of a season. In the second half, test whether (i) offensive efficiency in those contexts declines relative to matched controls, and (ii) expected values of the used actions are unequal (indifference test).
- *Falsified if:* neither holds. A label-shuffle placebo gives the null distribution of "discovered" predictability.

**E5: Dispersion honesty.**

- *Protocol:* PIT histograms and interval coverage of simulated totals with and without game-level latents.
- *Falsifies "the structural model is doing the work" if:* calibration requires game-level variance terms so large, and so heavily tuned to aggregates, that the possession machinery contributes nothing beyond the variance fit. Test by replacing the engine with a Poisson/negative-binomial rate model plus the same latents; if that matches, the engine is unnecessary.

**E6: The compositional-generalization test (the structural model's strongest claim).**

- *Protocol:* restrict evaluation to games where a top-2 usage teammate is out, a player was traded in the last 15 games, or there was a coaching change.
- *Falsified if:* the simulator is no better than the direct model *here*. Then the main argument for the architecture fails.

**E7: Learning curves.**

- *Protocol:* train on 1, 2, 4 and 8 seasons.
- *Falsifies "structure is a useful inductive bias" if:* the simulator's relative advantage doesn't shrink as data grows (it should be largest with little data), or it never has one.

**E8: Market benchmark (scientific, not for betting).**

- *Protocol:* compare log score and CRPS on game margins, totals and player props against closing-line-implied distributions.
- *Informative either way:* if the model is always worse and blending with the market doesn't help, the model contains no information beyond public consensus.

**E9: Momentum null.**

- *Protocol:* within-game permutation tests of shot sequences with shot-difficulty controls, compared against a Hawkes process and a static game latent.
- *Falsifies the need for sequence memory if:* a static game latent explains the dependence.

**E10: Drift robustness.**

- *Protocol:* compare prequential degradation after the 2018–19 and 2022–23 rule changes.
- *Tests whether:* structural components adapt faster than direct models, as they should if structure is real.

**My pre-registered expectations,** so I can be proven wrong:

- E1: roughly a tie on average for high-minute starters; simulator better for bench players and on joint events; the blend beats both by a small but significant margin.
- E2: history beyond the state plus a game latent adds under 1 millibit per possession.
- E3: weak correlation.
- E4: mostly null, with a few persistent exceptions.
- E6: clear simulator advantage.

---

## 18. The system I would build for maximum predictive accuracy

### Stage 0: Data layer

- **Bitemporal warehouse:** every record has *event time* and *knowledge time*. Every forecast is reproducible "as of" any timestamp. This is non-negotiable for honest backtests.
- **Identity resolution:** players, teams, coaches, officials, arenas.
- **Play-by-play → possession table:**
  - lineup reconstruction, fixing period-start lineups and substitution-timing errors
  - possession start type, duration, terminal event, points
  - free-throw sequences, and-ones
  - team vs. player rebounds; assist attribution
  - score and clock state; team and personal fouls; bonus; timeouts
- **Joins:** shot locations; tracking aggregates; injury-report snapshots (each version stored); officials; schedule and travel; markets in a separate schema.

### Stage 1: Slow layer, latent skills (updated nightly)

For each player, a latent vector θ covering:

- usage propensity by context (half-court vs. transition)
- shot-zone mix
- shot-making by zone (IRT-style, relative to expected difficulty)
- free-throw rate and FT%
- turnover propensity
- offensive and defensive rebounding strength
- assist/creation propensity
- fouling propensity
- defensive impacts on opponents' usage, shot mix and make probability
- stamina

Model details:

- **Dynamics:** θ_{t+1} = θ_t + aging drift(age, minutes) + η_t, with innovation variance larger across off-seasons and after injuries. The filter is an extended/ensemble Kalman filter, or variational Bayes, over the possession-level likelihoods from Stage 3. This is a joint, multi-output, RAPM-like regression carried through time.
- **Priors:**
  - hierarchical by position/archetype (embeddings)
  - rookies/imports: translation models from college, international and G League stats, and draft position
  - known cross-skill correlations (e.g., FT% informs 3P% true talent)
- **Faster-stabilizing signals:** attempt rates, shot-quality-adjusted make rates (where tracking aggregates exist) and rebounding opportunities, each weighted by its reliability.

### Stage 2: Medium layer, availability, rotation, game context (per game)

- **P(active):** injury-report status × historical conversion of each status, the player's history, rest patterns (back-to-backs), team context (tanking, seeding), a nowcast updated with each report.
- **Rotation policy:**
  - pre-game minute targets from an inverse-optimization / hierarchical model of the coach
  - in-game substitution hazards as a function of stint length, fouls, score × time, and the minute target
  - hard constraints: 5 players on court; minutes sum automatically
  - learned per coach with partial pooling across coaches
- **Game-level latents** (drawn per simulation):
  - team pace shock
  - team shooting-night effect
  - player form effects (small; hierarchical)
  - referee crew foul rate
  - in-game injury hazard

  Variances are estimated from data by matching held-out dispersion, fitted *jointly* rather than tuned at the end.

### Stage 3: Fast layer, the possession engine

Competing-risks semi-Markov model. Each component is a regularized GLM on player-embedding features. Small neural sub-models are used where interactions are proven to help (ablation-gated).

1. **Start type** is determined by the previous possession's end.
2. **Terminal event and duration:** a discrete-time hazard over 1-second bins for {field goal attempt, shooting foul, non-shooting foul (into bonus), turnover (live/dead), end of period}, depending on lineup, start type, score × time, and shot clock.
3. **User** (who takes the shot / commits the turnover / draws the foul): a softmax over the 5 offensive players. Logits = player usage × context + a lineup-interaction term (a set-structured network or pairwise MaxEnt term). This is the attention-shaped component.
4. **Shot zone and type:** categorical, given the user, lineup and defense.
5. **Make probability:** IRT-style σ(shooter skill in zone − zone difficulty − defender/lineup defensive impact + context + game latent).
6. **Free throws:** player FT% with a small pressure/fatigue context.
7. **Rebound:** P(offensive rebound ∣ both lineups, shot type/zone) and then *which* player (softmax), with a separate team-rebound pathway.
8. **Assist attribution:** P(assisted ∣ shot type, zone, user), with the assister chosen by a teammate softmax. This is where a passing-graph / GNN term earns its keep, if anywhere. Include a scorekeeper (arena) random effect.
9. **State update:** score, clock, fouls, bonus, stint lengths, then consult the rotation policy for substitutions at dead balls.

### Stage 4: Simulation

- For each game: S = 10–20k simulations.
- **Each simulation draws:** (i) parameters from the approximate posterior (or from a bank of posterior draws), (ii) availability outcomes, (iii) game-level latents, (iv) the game.
- Overtime handled natively.
- Output: a joint sample tensor [simulation × player × stat] plus team totals, margin and win.
- For tail bins with little mass, use importance sampling (tilt the shooting latent), or smooth the pmf with a fitted compound distribution.

### Stage 5: Direct distributional models (in parallel, not as a fallback)

- Per stat: NGBoost / distributional gradient boosting on player-game features (projected minutes distribution, recent rates, filtered skills, opponent, pace, teammate availability, rest). Output: full pmfs.
- A Panjer-recursion compound model for points as a fast semi-structural cross-check.

### Stage 6: Post-processing and blending (weather-style)

- **Combine** simulator and direct pmfs per stat by log-linear or linear pooling. Weights are learned prequentially per stat and per context stratum (e.g., more weight on the simulator when roles changed).
- **PIT-based recalibration** of each marginal (isotonic on the CDF), fitted on a rolling window.
- **Reconciliation:** adjust player marginals so that implied team totals agree with the calibrated team-total distribution. Use the MinT idea from hierarchical forecasting (Wickramasuriya, Athanasopoulos & Hyndman 2019), adapted to distributions, or simply reweight simulation samples.
- **Keep the joint:** apply marginal calibration through a *copula on the simulated samples*, i.e., remap each margin's quantiles while keeping the simulation ranks. Joint quantities (PRA, teammate combos) then inherit the calibrated margins *and* the simulator's dependence.
- **Optional market-informed variant:** anchor team margin and total to market-implied distributions and propagate through the joint samples. Always report it separately from the independent model.

### Stage 7: Outputs

For each player:

- P(active)
- the minutes distribution
- full pmfs for PTS, REB, AST, 3PM, TOV, STL, BLK, PRA and other combinations
- the requested bins (e.g., P(<20), P(20–24), P(25–29), P(30–34), P(35+))
- quantiles
- joint probabilities on request

For each game: win probability, margin and total distributions, and each team's box-score distribution.

### Stage 8: Evaluation and monitoring (continuous)

- Nightly prequential scores per component (bits) and per target (CRPS / RPS / log score).
- PIT and coverage dashboards by stratum; energy/variogram scores for joints.
- Drift alarms: rolling calibration slope, change in blend weights.
- Ablation harness to run E1–E10 as standing experiments, not one-offs.

### Stage 9: Live mode

- The same engine, initialized from the current in-game state.
- Latents updated within the game by a particle filter: a hot shooting night, an injury exit, foul trouble.
- Re-simulate the rest of the game each possession.

This is where the sequence architecture is most clearly better than direct models.

### Suggested build order (so each step is testable)

1. Data layer + possession table + as-of backtest harness + baselines. Without the harness, nothing else can be evaluated.
2. Direct distributional models (Stage 5). This is the bar everything else must beat.
3. Slow-layer skill filter.
4. A minimal possession engine (Tier-1 state, no rotation model; actual minutes supplied as an oracle for diagnosis only).
5. Rotation and availability models.
6. Game-level latents.
7. Blending and calibration.
8. Experiments E1–E10.
9. Only then: tracking-data components, neural interaction terms, entropy diagnostics for tactical research.

---

## References (selected)

- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Shannon, C. E. (1951). Prediction and Entropy of Printed English. *BSTJ*.
- Markov, A. A. (1913). Example of a statistical investigation of the text of *Eugene Onegin*…
- Rosenblatt, F. (1958). The Perceptron. *Psychological Review*. Minsky & Papert (1969), *Perceptrons*.
- Rumelhart, Hinton & Williams (1986). Learning representations by back-propagating errors. *Nature*.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.
- Widrow & Hoff (1960). Adaptive switching circuits (LMS).
- Kelly, J. L. (1956). A new interpretation of information rate. *BSTJ*.
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica*.
- Dawid, A. P. (1984). Statistical theory: the prequential approach. *JRSS A*.
- Efron & Morris (1975). Data analysis using Stein's estimator and its generalizations. *JASA*.
- Bühlmann, H. (1967). Experience rating and credibility. *ASTIN Bulletin*.
- Glickman, M. (1999). Parameter estimation in large dynamic paired comparison experiments. *Applied Statistics*.
- Herbrich, Minka & Graepel (2006). TrueSkill. *NeurIPS*.
- Stern, H. (1994). A Brownian motion model for the progress of sports scores. *JASA*.
- Gilovich, Vallone & Tversky (1985). The hot hand in basketball. *Cognitive Psychology*.
- Miller & Sanjurjo (2018). Surprised by the hot hand fallacy? *Econometrica*.
- Bocskocsky, Ezekowitz & Stein (2014). The hot hand: a new approach to an old "fallacy." MIT Sloan SAC.
- Oliver, D. (2004). *Basketball on Paper*.
- Rosenbaum, D. (2004). Measuring how NBA players help their teams win (adjusted plus-minus). Sill, J. (2010). Improved NBA adjusted +/- using regularization and out-of-sample testing. MIT Sloan SAC.
- Shirley, K. (2007). A Markov model for basketball. NESSIS.
- Štrumbelj & Vračar (2012). Simulating a basketball match with a homogeneous Markov model and forecasting the outcome. *International Journal of Forecasting*.
- Vračar, Štrumbelj & Kononenko (2016). Modeling basketball play-by-play data. *Expert Systems with Applications*.
- Gabel & Redner (2012). Random walk picture of basketball scoring. *JQAS*.
- Merritt & Clauset (2014). Scoring dynamics across professional team sports. *EPJ Data Science*.
- Miller, Bornn, Adams & Goldsberry (2014). Factorized point process intensities. *ICML*.
- Franks, Miller, Bornn & Goldsberry (2015). Characterizing the spatial structure of defensive skill in professional basketball. *Annals of Applied Statistics*.
- Cervone, D'Amour, Bornn & Goldsberry (2016). A multiresolution stochastic process model for predicting basketball possession outcomes. *JASA*.
- Alcorn & Nguyen (2021). baller2vec: A multi-entity transformer for multi-agent spatiotemporal modeling.
- Crutchfield & Young (1989). Inferring statistical complexity. *PRL*.
- Tishby, Pereira & Bialek (1999). The information bottleneck method.
- Xu, Zhao, Song, Stewart & Ermon (2020). A theory of usable information under computational constraints. *ICLR*.
- Nemenman, Shafee & Bialek (2002). Entropy and inference, revisited. *NeurIPS*.
- Walker & Wooders (2001). Minimax play at Wimbledon. *AER*. Palacios-Huerta (2003). Professionals play minimax. *REStud*. Chiappori, Levitt & Groseclose (2002). *AER*.
- Lucas, R. (1976). Econometric policy evaluation: a critique.
- Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- Gneiting, Balabdaoui & Raftery (2007). Probabilistic forecasts, calibration and sharpness. *JRSS B*.
- Gneiting (2011). Making and evaluating point forecasts. *JASA*.
- Gneiting & Ranjan (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES*.
- Czado, Gneiting & Held (2009). Predictive model assessment for count data. *Biometrics*.
- Scheuerer & Hamill (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review*.
- Diebold & Mariano (1995). Comparing predictive accuracy. *JBES*.
- Glahn & Lowry (1972). The use of Model Output Statistics (MOS) in objective weather forecasting.
- Gneiting, Raftery, Westveld & Goldman (2005). Calibrated probabilistic forecasting using ensemble MOS. *Monthly Weather Review*.
- Raftery, Gneiting, Balabdaoui & Polakowski (2005). Using Bayesian model averaging to calibrate forecast ensembles. *Monthly Weather Review*.
- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model (EnKF).
- Wickramasuriya, Athanasopoulos & Hyndman (2019). Optimal forecast reconciliation (MinT). *JASA*.
- Adams & MacKay (2007). Bayesian online changepoint detection.
- Duan et al. (2020). NGBoost. *ICML*.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World* (conformal prediction).
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*. Schneidman et al. (2006). Weak pairwise correlations imply strongly correlated network states. *Nature*.
- Banister et al. (1975). A systems model of training for athletic performance.

*Citations are from memory. Bibliographic details, and especially exact years for industry systems and tracking-provider transitions, should be verified before being relied on formally.*
