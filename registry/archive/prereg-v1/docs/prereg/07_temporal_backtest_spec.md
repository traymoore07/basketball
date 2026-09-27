# 07. Temporal backtesting: living through basketball history

*Implementation: `src/hoopslab/backtest/runner.py` (rolling origin, as-of requests, lockbox guard), `src/hoopslab/asof.py` (visibility).*

No random train/test splits anywhere. The only protocol is:

```
for each forecast time t, in calendar order:
    view  = store.view(t)                      # records with knowledge_time <= t
    reqs  = as-of roster of every team with a game at t + lead
    for each model: model.fit(view); forecast = model.predict(view, reqs)   # logged, hashed
    (results for games before t were revealed simply by becoming visible)
score all forecasts afterwards against outcomes from the full store
```

## 1. Forecast timestamps and information sets

| Info set | Forecast time | What is visible | Use |
|---|---|---|---|
| **pre** (primary) | scheduled tip − 60 min | injury reports published by then; no announced starters | all confirmatory hypotheses |
| morning | tip − 10 h | earlier report snapshots | sensitivity: value of late information |
| **lock** | tip − 5 min | announced starting lineups, late scratches | mode L / S forecasts; reported separately |
| live | during the game | events with wall-clock knowledge times | out of scope for confirmatory tests; reserved for later |

The "pre" offset of 60 minutes is pre-registered for NBA and WNBA. For NCAA, whose availability reporting differs, the offset is also tip − 60 min, but availability-conditional claims require injury-report coverage above 80% for the season (`11`).

## 2. Windows

Let each league's seasons with Level-A data be s₁ … s_N (complete seasons only).

| Window | Seasons | Purpose |
|---|---|---|
| **Burn-in** | s₁ … s₃ | priors, initial fits; never scored |
| **Development** | s₄ … s_{N−2} | building, tuning, exploratory tests, instrument checks on real data |
| **Lockbox (confirmation)** | s_{N−1}, s_N | confirmatory tests, run once after P6 (config freeze) |

Rules:

- If N < 6 for a league, burn-in shrinks to 2 seasons and development must contain at least 1 season. If that is impossible, the league enters only exploratory analysis.
- Event-level experiments need Level B from the first burn-in season. If PBP begins later than box scores, event-level windows are defined on the PBP-covered seasons with the same 3 / … / 2 structure.
- The lockbox is sealed in code: `BacktestSpec.lockbox`, with `LockboxError` unless `unseal_manifest_hash` equals the frozen registry hash.
- Seasons interrupted or reshaped (e.g. shortened or bubble seasons) are kept, flagged as a stratum, and never used as the only confirmation season. If one would be, the lockbox extends back one season.

## 3. Update cadence

| Component | Cadence | Notes |
|---|---|---|
| Availability, injury status, rosters, rotation inputs | every forecast time | cheap as-of reads |
| Direct models (L1–L6) | refit daily (every forecast day) | incremental where possible |
| Latent-skill filters (L8) | filter update after each game day | Kalman-style |
| Structural simulator parameters (L10–L12) | full re-estimation weekly (`refit_days = 7`); daily availability/rotation | pre-registered compute compromise; H1 sensitivity: daily refit on a 20% sample of days |
| Event-level science models | refit monthly (prequential) | E-HIST, E-SHANNON, E-COMP |
| Hyperparameters | fixed per season at the season boundary, chosen on data strictly before it | never tuned on the evaluation season |
| Post-processing (L14, H13) | rolling window of the previous 60 game days | as-of PITs only |

## 4. Season boundaries and offseason

- **Variance inflation:** latent-skill states get extra process variance across the offseason. The amount is a hyperparameter set on burn-in/development data.
- **Aging drift:** deterministic aging-curve drift applied at the boundary. The curve is estimated as-of from prior seasons.
- **Rule changes:** a new `ruleset_id` becomes effective at its date. Models may include ruleset indicators. The first season after a major rule change is a pre-registered stratum ("post-rule-change"), reported separately in E-LADDER and H1.
- **Preseason games:** excluded from scoring. They may be used as inputs only if their knowledge times are honest.
- **Postseason:** scored as a separate stratum. Regular season is primary. Rotations tighten in the postseason, a known distribution shift, reported but not confirmatory.
- **In-season tournaments / play-in / conference and national tournaments (NCAA):** included. Phase is a feature and a reporting stratum. Neutral sites are flagged.

## 5. Entities that change

| Situation | Handling |
|---|---|
| **Rookies / first-time players** | Priors from as-of pre-league information: draft slot, PERSON_LINK-linked NCAA/international stats (if the link's knowledge_time ≤ t), position. Flagged `low_sample` until 10 appearances. Their first 20 appearances form the H11b stratum. |
| **Traded / signed players** | Skill state carries over. Team context switches at the ROSTER spell's knowledge time. `own_move_21d` and `roster_change_21d` strata. |
| **Two-way / 10-day / hardship / G-League assignments** | Roster spells with transaction types. Players with no NBA appearances remain in the request population if rostered (mode U). |
| **Returning from long absence** | `return_from_absence` stratum (≥ 5 consecutive missed games). Skill-state variance inflated by missed time. |
| **Coaching changes** | COACH_TENURE spell change becomes visible at announcement. `coaching_change_20g` stratum. Rotation models partially pooled across coaches with a coach effect that resets. |
| **Expansion teams** | Team-level priors from roster composition (expansion draft) at knowledge time. First-season team effects flagged. |
| **Relocation / rename** | New TEAM spell, same franchise_id. Nothing resets. |

## 6. Cross-league timing

Pooled or transfer models (H11) respect **calendar** time. A WNBA game played in August 2027 cannot inform an NBA forecast issued in May 2027, even though the two "seasons" are indexed differently. The as-of view makes this automatic, because all leagues share one clock. T1 is run on pooled pipelines to prove it.

## 7. What is logged per forecast

- Model name
- Rung
- Configuration hash
- Code commit
- Forecast time
- Data snapshot hash
- Accessed tables (from `AsOfView.accessed`)
- Request keys
- Full pmfs and p(appear)
- Joint samples (for simulators, on a stratified subsample of games when storage requires)

Forecasts are append-only and written before the outcomes are joined.

## 8. Scoring and clustering

Losses are per unit (player-game, team-game, event). Clusters are game-days, which share model state and news. Uncertainty comes from a moving-block bootstrap over game-days with a 7-day block (`hoopslab.metrics.compare`). Event-level experiments bootstrap over games.
