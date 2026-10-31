# 04. Pre-registered expectations

*Canonical: `registry/expectations.yaml` (frozen). This page summarises it. Written before any real data was seen, so that hindsight bias is visible later.*

**Strength labels:**

- **strong**: I would be surprised (< ~15%) to be outside the range.
- **weak**: a central guess.
- **speculation**: little basis; recorded so that the surprise can be measured.

Probabilities are subjective chances of each pre-registered decision.

## Headline expectations

| Question | My expectation | Strength |
|---|---|---|
| Which experiments will produce **large** gains? | Availability-aware minutes (L3 vs L2: 5–15% on MIN, mode U); player identity (L1 vs L0: 35–55%); oracle minutes shows that 25–45% of PTS CRPS is minutes error | strong / weak |
| Which will produce **tiny** gains? | Structural simulator vs strong direct model on marginals (−1% to +1.5%); game latents on core CRPS (0–0.5%); parameter uncertainty overall (±0.3%); neural vs GLM/GBM at play-by-play scale (±0.5%) | weak |
| Which will probably **fail**? | H10 (neural gains at PBP scale; p_falsified ≈ 0.60); H6d (coaching-change advantage, p_supported ≈ 0.15); naive cross-league pooling for the NBA (harmful, p ≈ 0.70) | weak / speculation |
| Where should **simulation** beat direct prediction? | Joint quantities (teammates' minutes, PRA, blowout starter minutes); star-absence and trade strata; tail calibration **if** latents are modelled; live forecasting | weak (live: strong) |
| Where should **direct** models win? | Marginals of stable-role starters; minutes in stable rotations; scorer-judgement stats (AST/STL/BLK); leagues with poor lineup data (NCAA) | weak / strong for NCAA |
| How much long-range **sequence memory**? | < 0.5 millibit per possession segment beyond a compact state + game latent. "Momentum" is mostly between-game heterogeneity. H3b diagnosis NO_HISTORY_SIGNAL or APPARENT_HISTORY_EXPLAINED_BY_LATENT with p ≈ 0.75 | strong |
| Will **compression** track distribution quality? | Only partly: Spearman ρ between event bits and core CRPS across variants in [0.2, 0.7]. Rebound/assist actor ablations and the game latent are the expected discordant variants | weak |
| Is low **entropy** exploitable? | Mostly no. Usable-information discoveries replicate (~60%) but rarely improve player-stat distributions (~25%). Placebos stay ≤ 5% (strong) | weak / strong |
| Will **joint** forecasts improve? | vs independence: yes (energy-score skill 5–15%, p ≈ 0.80); vs a Gaussian copula: 0–5% (p ≈ 0.45) | weak |
| Will **parameter uncertainty** matter? | Only for low-sample players: 80% coverage 70–76% (plug-in) → 77–82% (posterior predictive), p ≈ 0.60 | weak |

## Ladder gaps (core CRPS skill, mode U, NBA)

| Gap | Range | Strength |
|---|---|---|
| L1 vs L0 | 35–55% | strong |
| L2 vs L1 | 0–3% (5–15% in role-change strata) | weak |
| L3 vs L2 | 2–8% (MIN alone 5–15%) | weak |
| L4 vs L3 | 0.5–2% (3–10% for < 10 appearances) | weak |
| L5 vs L4 | 1–3% | weak |
| L6 vs L5 | 0.5–3% | weak |
| L8 inside L5/L6 | 0–1.5% | weak |
| L9 vs L6 | −6% to −1% | weak |
| L10 vs L6 | −3% to +1% | speculation |
| L11 vs L10 | 0–0.5% core; 2–6% upper-tail twCRPS | weak |
| L12 vs L11 | ±0.3% | weak |
| L13 vs L12 | ±0.5% | weak |
| L14 vs best single | 0.5–2% | weak |
| Oracle minutes (PTS) | 25–45% (starters 15–30%) | weak |
| Oracle availability (core, U) | 5–15% | weak |

## NBA vs WNBA vs NCAA

- **Similar:** hot hand/momentum is small everywhere; compact-state sufficiency holds similarly (weak).
- **Different:**
  - Home advantage: NCAA > NBA ≈ WNBA (strong).
  - Per-possession efficiency variance across teams: NCAA ≫ NBA (strong).
  - Roster stability is lowest in the NCAA, so parameter-uncertainty and translation-prior effects are largest there (strong).
  - The structural-model advantage is smallest in the NCAA because of lineup data quality (weak).
  - Game outcomes are more predictable in the NCAA than in the NBA (weak). WNBA: uncertain (speculation).
  - NCAAW quarter structure and per-quarter bonus differ from NCAAM halves (rules, strong).
- **Transfer by component:**
  - Likely to transfer: free-throw shooting, shot-make vs distance shape, shot-clock hazard shape, rebound-share mechanics.
  - Unlikely to transfer: team strength, usage allocation, rotation policy, pace level.

## Probabilities per hypothesis

See the `hypotheses` section of the YAML (p_supported, p_falsified, …). After the confirmation run, the scorecard puts each outcome next to these probabilities. Averaged over hypotheses, the log score of my own pre-registered probabilities is itself reported, as a measure of how well I understood the problem in advance.
