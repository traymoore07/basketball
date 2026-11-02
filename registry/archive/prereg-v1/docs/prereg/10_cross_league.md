# 10. NBA, WNBA, and NCAA: one corpus, three leagues, or something in between?

*Pre-registered design for using the three leagues (four competitions: NBA, WNBA, NCAA men's, NCAA women's). Serves H11a/H11b and constrains every other hypothesis.*

## 1. Decision

**Option E, a specific combination.** The four options are not equivalent, and more games are not automatically more information.

| Role | Option | How it is used |
|---|---|---|
| Reference for every comparison | **B**: separate leagues | Every model is first fitted per league. This is the H11 reference and the fallback. |
| Primary pooled approach | **C**: related populations under a hierarchical model | Parameters are expressed in **rule-normalised units** and partially pooled across leagues with league- (and era-) level deviations whose variances are estimated from data. |
| Targeted transfer | **D**: transfer-learning sources | (i) Player-level *translation priors* (NCAA → NBA/WNBA; NCAAW → WNBA). (ii) Pre-training of neural event models (H10) on the pooled corpus, then fine-tuning per league. |
| Comparison arm only | **A**: one combined corpus | Naive pooling (league as at most a feature). Expected to show negative transfer (`04`). It exists to measure that. |

Scores are **never pooled across leagues**. Each league has its own development and lockbox windows and its own scorecard.

## 2. Why the leagues differ (what the hierarchy must absorb)

Rule facts are recorded as **data** in RULESET, looked up by `ruleset_id`, and must be verified against official rulebooks by the data team. The values below are my working knowledge and are not to be hard-coded.

| Dimension | NBA | WNBA | NCAAM | NCAAW |
|---|---|---|---|---|
| Game structure | 4 × 12 min | 4 × 10 min (halves before 2006) | 2 × 20 min halves | 4 × 10 min quarters (halves before 2015–16) |
| Shot clock | 24 s (14 s reset after OREB since 2018–19) | 24 s (30 s before 2006) | 30 s (35 s before 2015–16) | 30 s |
| Three-point distance | 7.24 m arc (shortened 1994–97) | 6.75 m (since 2013; shorter before) | 6.75 m (since 2019–20; shorter before) | 6.75 m (since 2021–22; shorter before) |
| Foul limit | 6 | 6 | 5 | 5 |
| Team-foul bonus | per quarter | per quarter | per half, 1-and-1 then double bonus | per quarter, two shots (since 2015–16) |
| Overtime | 5 min | 5 min | 5 min | 5 min |
| Season length | 82 (with exceptions) | ~34–44, changing by era | ~30–35 + tournaments | ~30–35 + tournaments |
| Teams / talent spread | 30; compressed talent | 12–15; compressed but very top-heavy | ~360 D-I; enormous spread | ~360 D-I; enormous spread |
| Schedule | balanced-ish | balanced-ish | highly unbalanced (conferences) | highly unbalanced |
| Roster stability | moderate (trades, free agency) | moderate; small rosters, hardship contracts | low (graduation, transfer portal, eligibility changes) | low |
| Home advantage | small / declining | small | large | moderate–large |
| Data quality (lineups, sub timing) | high (modern era) | good | variable; lineups often missing or wrong | variable |

Consequences:

- **Time and counts are rule-relative.** Per-minute rates are not comparable across 40- and 48-minute games. Pooling uses per-possession rates, shot-clock fractions (elapsed / shot clock), and team-foul state relative to the bonus rule.
- **Some components plausibly share physics:**
  - free-throw shooting;
  - make probability as a function of distance (after a line-distance adjustment);
  - the rebound-share structure given shot type;
  - shot-clock hazard shape on the normalised clock.
- **Others plausibly do not:**
  - team strength;
  - usage allocation;
  - rotation policy and minutes;
  - pace level;
  - the foul-trouble response (5 vs 6 fouls).
- **Eras are leagues too.** The NBA of 2002 and of 2024 differ as much as some league pairs. The hierarchy therefore uses `league × era(ruleset)` as the lowest pooled level, with era deviations shrunk toward the league.

## 3. Hierarchy (pre-registered structure)

```
global
 └─ gender (men / women)
     └─ level (professional / college)
         └─ league (NBA, WNBA, NCAAM, NCAAW)
             └─ era (ruleset_id)
                 └─ team-season  ─ player
```

For each engine component, parameter θ_{league,era} = θ_global + δ_gender + δ_level + δ_league + δ_era. The δ variances are estimated (partial pooling). A component "transfers" when its δ variances are small relative to within-league estimation error. X3 below tests this per component.

## 4. Experiments that measure transfer (instead of assuming it)

| ID | Question | Protocol | Reading |
|---|---|---|---|
| **X1** (H11a) | Does pooling help each target league? | Same architecture, three training regimes: league-only (B), hierarchical (C), naive pooled (A). Scored on each league's lockbox separately. | SUPPORTED per league per the H11a MPE. Naive pooling is expected to be ≤ league-only. HARMFUL is reported. |
| **X2** | Is an auxiliary season worth a target season? | Learning curves: target data truncated to k = 1, 2, 4, all seasons, each with and without auxiliary leagues. | Exchange rate: the number of target seasons that one auxiliary season substitutes for (interpolated). Zero means no transfer. |
| **X3** | Which components transfer? | Event level: for each component, compare league-specific vs pooled-with-deviation vs fully pooled parameters by held-out bits. | A table of components × leagues. Pre-registered expectation in `04`: FT%, shot-distance shape, rebound shares, and shot-clock hazard transfer; usage, rotation, and team strength do not. |
| **X4** (H11b) | Do pre-league stats help rookies? | NCAA → NBA/WNBA translation priors vs position/draft priors on each rookie's first 20 appearances. PERSON_LINK knowledge times enforced (L-F3). | SUPPORTED per H11b MPE. |
| **X5** | Do old eras help or hurt? | Within-league: training with era-decay weights {none, half-life 3, 6, 12 seasons}, hierarchy with era deviations. | Chooses era weighting on development only. The lockbox tests the chosen option vs recent-only. |
| **X6** | Negative-transfer monitor | For every pooled model: per-league calibration (PIT-KS, coverage) vs league-only. | Any league with HARMFUL CRPS or calibration degradation > 0.02 flags the pooled model for that league. |
| **X7** | Men's vs women's structure | Variance of δ_gender per component; separate vs gender-pooled rebound/usage components. | Descriptive. Informs whether NCAAM and NCAAW should share anything but the global level. |

**Guardrails**

- **Calendar-time as-of across leagues** (L-C6). The single as-of clock guarantees that a WNBA season ending in September cannot inform a June NBA forecast. T1 runs on every pooled pipeline.
- **Data-quality weighting.** Games with low lineup/substitution quality (NCAA) contribute to event-level components that do not need lineups. Lineup-dependent components use only games with quality flag ≥ "exact" or "dead_ball_batch". The weighting scheme is fixed before P6.
- **No league is evaluated on the pooled score.** A pooled model that helps the NCAA and hurts the NBA is reported as exactly that.

## 5. Expectations (copied from `04` for convenience; the YAML is canonical)

- NBA gains little from pooling (0–0.5%), and naive pooling is harmful (p ≈ 0.70).
- The WNBA gains more (0.5–2%): smaller league, fewer games (speculation).
- In the NCAA, within-NCAA hierarchy (conference/level) matters more than cross-league data.
- Translation priors help rookies by 3–8% over their first 20 games.
- Home advantage: NCAA > NBA ≈ WNBA (strong prior).
- The structural model's advantage is smallest in the NCAA, because of lineup data quality (weak prior).
