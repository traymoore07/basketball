# P1 report: baseline ladder L0-L4 on NBA `nba_stats` (development window 2000-2020)

*Exposure class O-dev. Written 2026-10-30 from the complete 23-season run 2 (results/p1_ladder/). The
preliminary 8-season summary of run 1 (results/p1_ladder_run1_chunkbug/) was retained for audit only
and is not used anywhere in this report. Nothing in the frozen specification (models, half-life grid,
targets, metrics, strata, decision thresholds) was changed in response to any observed outcome.*

## 0. Fixed report block

```
PREREGISTRATION INTEGRITY:  PASS (python -m hoopslab.registry verify: "PRE-REGISTRATION INTACT (tier 1,
                            tier 2, archive, logs)" before the run (both machines) and after the run)
CORPUS VERSION:             BASKETBALL_CORPUS_v1.0.2/researcher_isolated, built 2026-10-10 19:25:57 UTC;
                            researcher_isolated_v1.0.2.duckdb sha256 ca30422bbbea9f247b4577dc1f8c3ab9
                            8e0c3e804db6ff83fad9b74e138ba5be (2,456,039,424 bytes); release manifest
                            9,317 files verified OK
ADAPTER HASH:               src/hoopslab/adapters/corpus_v102.py sha256 8fbca2812e789faab7f082f5b65327
                            8910de5f7f000a9c6d6d3a57fc5b2c7d64 (attested 2026-10-30T14:01:26Z, unchanged
                            since C9)
OUTCOME EXPOSURE:           O-dev (EXPOSURE_LOG rows 8-10)
HOLDOUT SEAL:               INTACT (rows with season >= 2023 in every in-game table: 0, checked before
                            the run on both machines; lockbox 2021-2022 guarded by LockboxError, never
                            entered; the P1 driver asserts every forecast time against the lockbox)
NBA P1 POPULATION:          as-of ROSTER spells, forecast time = scheduled tip - 60 min, refit at every
                            forecast time; 23 seasons 1998-2020 (1998-1999 burn-in, forecast only for
                            the L2 half-life selection, never reported; 2000-2020 development, 21
                            seasons reported); 28,990 games forecast, 953,481 player-game requests
                            (870,895 in 2000-2020); regular-season development scoring units: 816,775
                            (mode U) / 514,616 (mode A); 3,378 game-day clusters regular, 942 postseason;
                            2 games excluded (DI-1 / M24, 2005 placeholder-team rows); 0 games missing
                            (coverage assertion passed in all 23 seasons)
L0 STATUS:                  COMPLETE (Climatology; 23/23 seasons; leakage T1/T2/T3/T6/T7 pass 23/23)
L1 STATUS:                  COMPLETE (PlayerMean, no recency; 23/23; leakage pass 23/23)
L2 STATUS:                  COMPLETE (PlayerMean half-life grid 5/10/20/40 appearances, selected per
                            season on strictly prior scored seasons; h = 5 selected in all 21 seasons:
                            OPTIMUM AT PRE-REGISTERED SEARCH BOUNDARY - wider search not tested in P1)
L3 STATUS:                  COMPLETE (MinutesRate shrink=False; runs as frozen under M1: no
                            INJURY_REPORT exists, so availability = own appearance history; 23/23)
L4 STATUS:                  COMPLETE (MinutesRate shrink=True, posterior predictive; L4-plugin variant
                            also run; 23/23)
PRIMARY TARGET RESULTS:     core CRPS skill vs L0, mode U regular season: L1 +20.49%, L2 +29.80%,
                            L3 +29.72%, L4 +29.74%; mode A: L1 +20.51%, L2 +27.15%, L3 +29.57%,
                            L4 +29.55% (per-target tables in section 3)
CALIBRATION FINDINGS:       L0 near-uniform PIT by construction; L1 strongly under-dispersed and biased
                            high in mode U (MIN cov80 0.644, dispersion 1.35, bias +2.4 min); L2 repairs
                            most of it (MIN cov80 0.728) but is over-dispersed on counting targets in
                            mode A (PRA cov80 0.876); L3/L4 near-nominal on PTS/REB/AST/FG3M/TOV/PRA in
                            mode A (cov80 0.80-0.84) but over-dispersed on MIN in mode A (cov80 0.891,
                            dispersion 0.77); a positive mode-U bias (+1.3 to +2.2 on MIN/PRA) survives
                            every rung and vanishes in mode A; low-sample stratum: L4 posterior
                            predictive raises 80% coverage over the plug-in by 0 to 1.1 points only
BEST SIMPLE BASELINE BY TARGET (lowest CRPS, credit-chain rungs):
                            mode U: MIN L2, PTS L3, REB L3 (= L2 within 0.0001), AST L4, FG3M L2 = L4,
                            TOV L3, PRA L3; mode A: MIN L2, PTS L4, REB L4, AST L4, FG3M L3, TOV L3,
                            PRA L4 (L4 and L3 differ by < 0.1% everywhere outside the low-sample stratum)
MARGINAL GAIN BY LADDER STEP (core CRPS skill vs the immediately simpler rung, 95% cluster-bootstrap CI,
                            Holm within the family, frozen MPE 1%; final decision after the frozen no-harm
                            and calibration-floor constraints):
                            mode U: L1 vs L0 +20.49% [+19.78, +21.25] PROMISING (raw SUPPORTED,
                                    downgraded by the calibration floor on 6 targets);
                                    L2 vs L1 +11.76% [+11.61, +11.93] SUPPORTED;
                                    L3 vs L2 -0.17% [-0.26, -0.07] HARMFUL;
                                    L4 vs L3 +0.03% [+0.01, +0.04] FALSIFIED;
                                    L4 vs best simpler (L2) -0.14% [-0.23, -0.04]
                            mode A: L1 vs L0 +20.51% [+20.29, +20.74] PROMISING (raw SUPPORTED;
                                    no-harm log:MIN, floor on 5 targets);
                                    L2 vs L1 +8.36% [+8.16, +8.57] PROMISING (raw SUPPORTED; no-harm
                                    log:TOV, floor on 5 targets);
                                    L3 vs L2 +3.08% [+2.93, +3.23] PROMISING (raw SUPPORTED; no-harm
                                    log:MIN, rps:MIN, floor on MIN);
                                    L4 vs L3 -0.01% [-0.03, +0.01] FALSIFIED
PRE-DATA EXPECTATIONS VS OBSERVED:
                            section 7: 4 of the 5 ladder-gap expectations are contradicted (L1 vs L0
                            below the strong 35-55% range; L2 vs L1 far above 0-3%; L3 vs L2 harmful
                            in mode U where 2-8% was expected and MIN 5-15%; L4 vs L3 at zero where
                            0.5-2% was expected); parameter-uncertainty "tiny gain" and the low-sample
                            L4 gain in mode A agree
NEW DATA ISSUES DISCOVERED: DI-1 placeholder team '0' rows (M24, structural, 2 forecastable games);
                            DI-2 step change in the as-of roster population in 2004 (41-42 requests per
                            game in 1998-2003, 28-33 from 2004 on) coinciding with the drop of L1-vs-L0
                            skill from 33-35% to 13-21%: recorded, not investigated (section 8)
MISMATCHES / BUGFIXES / AMENDMENTS:
                            mismatches: M24 new (DI-1, SCOPE-RESTRICTION, structural); M1 pre-existing
                            (no INJURY_REPORT: L3 runs degenerate as frozen). Bugfixes: C9 (adapter
                            roster gap-close, before any outcome), C11 (driver chunking, run 1
                            superseded), C15 (frozen no-harm / calibration-floor constraints were not
                            applied to the decision labels). Engineering: C10, C12, C13, C16.
                            Governance: C14. AMENDMENTS: none.
H6 ORIGINAL CONFIRMATORY STATUS:
                            UNCHANGED - NO H6 OUTCOMES EXAMINED
READY FOR NEXT RESEARCH PHASE:
                            NO. P1 is complete, reproducible and frozen-compliant, but two of the three
                            frozen P1 sanity gates FAIL (L1 >> L0 on every core target; L3 >= L2 on MIN
                            in mode U). docs/prereg/03 says a failed gate "means a bug" until shown
                            otherwise. Neither failure was investigated in P1 (no diagnostic beyond the
                            frozen strata is authorised); both have documented candidate causes (DI-2;
                            M1 plus L3's slower minutes estimator) that need an authorised diagnostic
                            phase, not a model change, before L5.
```

## 1. What ran, and how it is reproduced

| item | value |
|---|---|
| code commit of the run | `2c3c28a405901d4b93ed2c469f6c4714f8392b69` (run 2, every season); this report and C15/C16 are committed on top of it |
| driver | experiments/p1_ladder/run_season.py sha256 `39800ecfa073de8c0008c46796b91a64d4503cd34ace89ed571b86a1735a720a` (C11 version; identical hash on both machines) |
| models wrapper | experiments/p1_ladder/models.py sha256 `63c3ce1f014a0ac679f3720c8104fc7efaed008950d0c7e53b8c76cef53654b3` (bit-identical to `hoopslab.models.baselines`, check_shared_equals_vanilla.py) |
| configuration | experiments/p1_ladder/config.json sha256 `3f5378cc0261dfeb24e32b6ffc1a2358c353896cb7b4607ef3c9324721c6e5aa`. Note: its `forecast_points` sentence still describes run 1's fixed 7-day chunks; the C11 driver derives the chunk length from the measured GAME v1 lag (7 d) as 6 d. The file was left untouched so that its attested hash stays valid; the driver docstring and done_*.json (`game_v1_lag_days`, `chunk_days`) carry the truth |
| seeds | PIT randomisation 0; cluster bootstrap 12345 (B = 2000, block 7 game-days); leakage tips rng 1000 + season |
| train / validation periods | rolling origin: at every forecast time the model is refit on everything the as-of store exposes before that time (burn-in 1997-1999 is history only; 1998-1999 forecast solely for the L2 half-life choice); validation = development seasons 2000-2020; lockbox 2021-2022 sealed; holdout >= 2023 absent |
| forecast timestamps | every P1-phase scheduled tip in each season window minus 60 min (results/p1_ladder/requests_{S}.parquet, column forecast_time; tips in nba_stats are date-only and land at 12:00 UTC, so forecast time = 11:00 UTC) |
| phases forecast | regular, postseason, play_in, in_season_tournament; preseason never forecast |
| exclusions | DI-1: games with a placeholder team id `nba:nba_stats:0` (2 in the P1 population, both 2005 "playoffs" rows 0040401000 / 0040401001, no box score); verified by experiments/p1_ladder/verify_di1.py over all 23 seasons (results/p1_ladder/di1_verification.json: 17 source rows, 15 store rows, 2 forecastable, 0 requests involving the placeholder team, `all_finished_seasons_consistent: true`) |
| environment of the backtest | TrayServer: Python 3.12.0, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, duckdb 1.5.5, pyarrow 26.0.0, Windows 11 10.0.26200, OMP/OPENBLAS/MKL threads = 1 per worker, 8 season workers; preflight results/p1_ladder/preflight_server.json ALL PASSED at 2c3c28a |
| environment of leakage, preflight, summary scripts | i9: Python 3.13.15 with the same pandas/numpy/scipy/duckdb versions (results/p1_ladder/preflight.json ALL PASSED); summary and tables were computed on the server in the venv above and copied back unchanged |
| artifacts | per season: requests_{S}.parquet, forecasts_{S}_{model}.npz (float32 pmfs, 9 models), scores_{S}_part*.parquet (883 parts), strata_{S}.parquet, done_{S}.json, log_{S}.txt, stdout_{S}.txt; launch_log.txt; cluster_sums.parquet; pit_summaries.parquet; summary.json; summary_tables.md; di1_verification.json; leakage_*.json; di1_evidence/. Large files are git-ignored (C10/C12) and kept on both the i9 and TrayServer (results/p1_ladder/, 16.3 GB) |
| dry-run evidence for C11 | done_2001_dry.json (1260/1260 games), done_2005_dry.json (1314/1314 with 2 DI-1 exclusions) |

Run bookkeeping per season (from done_{S}.json; "req" = player-game requests, "games" = games with
requests = expected P1-phase games minus DI-1, "excl" = DI-1 games):

| season | forecast days | req | games | missing | excl | req/game |
|---|---|---|---|---|---|---|
| 1998 | 202 | 49,298 | 1,260 | 0 | 0 | 39.1 |
| 1999 | 124 | 33,288 | 791 | 0 | 0 | 42.1 |
| 2000 | 208 | 52,219 | 1,264 | 0 | 0 | 41.3 |
| 2001 | 209 | 52,873 | 1,260 | 0 | 0 | 42.0 |
| 2002 | 205 | 53,199 | 1,260 | 0 | 0 | 42.2 |
| 2003 | 208 | 52,318 | 1,277 | 0 | 0 | 41.0 |
| 2004 | 213 | 40,147 | 1,271 | 0 | 0 | 31.6 |
| 2005 | 210 | 39,345 | 1,314 | 0 | 2 | 29.9 |
| 2006 | 211 | 38,172 | 1,319 | 0 | 0 | 28.9 |
| 2007 | 206 | 38,109 | 1,309 | 0 | 0 | 29.1 |
| 2008 | 209 | 37,937 | 1,316 | 0 | 0 | 28.8 |
| 2009 | 208 | 37,624 | 1,315 | 0 | 0 | 28.6 |
| 2010 | 209 | 37,227 | 1,312 | 0 | 0 | 28.4 |
| 2011 | 206 | 37,584 | 1,311 | 0 | 0 | 28.7 |
| 2012 | 166 | 32,418 | 1,074 | 0 | 0 | 30.2 |
| 2013 | 208 | 39,295 | 1,315 | 0 | 0 | 29.9 |
| 2014 | 207 | 39,532 | 1,319 | 0 | 0 | 30.0 |
| 2015 | 205 | 39,871 | 1,311 | 0 | 0 | 30.4 |
| 2016 | 210 | 39,799 | 1,316 | 0 | 0 | 30.2 |
| 2017 | 204 | 40,195 | 1,309 | 0 | 0 | 30.7 |
| 2018 | 212 | 42,385 | 1,312 | 0 | 0 | 32.3 |
| 2019 | 213 | 42,626 | 1,312 | 0 | 0 | 32.5 |
| 2020 | 193 | 38,020 | 1,143 | 0 | 0 | 33.3 |

Leakage (docs/prereg/06, required before any result is reported): T1 truncation, T2 future
perturbation, T3 id relabel, T6 population, T7 feature recompute, each on 20 random forecast times per
season, through one composite forecaster stacking all nine models: 23/23 seasons pass every detector;
T5 static access scan passes (results/p1_ladder/leakage_*.json, committed at 2c3c28a).

## 2. Scoring protocol (frozen, docs/prereg/08)

Primary: CRPS (exact integer form) per unit; core aggregate = equal-weight mean over the seven core
targets of relative CRPS skill, regular season, development window. Secondary: log score in bits
(1e-4 uniform mixture), RPS on the pre-registered bins. Diagnostic: randomised PIT (KS distance,
dispersion ratio 12 Var(PIT), coverage 50/80/95), MAE of the median, RMSE of the mean, upper-tail twCRPS
(PTS 25, PRA 40, REB 10, AST 8; zero for MIN/FG3M/TOV by construction). Modes: U (unconditional, the
appearance probability is part of the forecast) and A (conditional on appearing); never averaged.
Comparison statistic: game-day cluster bootstrap (B = 2000, block 7, seed 12345) of the relative core
skill, Holm within the four immediately-simpler comparisons, MPE = 1% core skill, decision rule
SUPPORTED / HARMFUL / FALSIFIED / PROMISING / INCONCLUSIVE, then the two binding constraints of 08
section 5 (no-harm on log score and RPS at the aggregate level; calibration floor: candidate's PIT-KS or
|cov80 - 0.8| worse than the reference's by more than 0.02 on any core target of the comparison, same
mode). Credit rule: a rung is compared with the immediately simpler rung and with the best simpler rung
by core skill on the development window.

## 3. Primary results per rung and target

### 3.1 Mean CRPS per unit, regular season 2000-2020 (lower is better)

Mode U, 816,775 units per target:

| rung | MIN | PTS | REB | AST | FG3M | TOV | PRA |
|---|---|---|---|---|---|---|---|
| L0 | 8.0332 | 4.0006 | 1.7059 | 0.9915 | 0.3643 | 0.6068 | 6.2063 |
| L1 | 6.3543 | 3.1766 | 1.3540 | 0.7524 | 0.2897 | 0.5185 | 4.8323 |
| L2 (h=5) | **4.9940** | 2.7476 | 1.2148 | 0.6802 | **0.2706** | 0.4828 | 4.1536 |
| L3 | 5.1557 | **2.7419** | **1.2147** | 0.6790 | 0.2712 | **0.4799** | **4.0990** |
| L4 | 5.1557 | 2.7426 | 1.2152 | **0.6781** | **0.2706** | 0.4800 | 4.1012 |
| L4-plugin | 5.1557 | 2.7436 | 1.2161 | 0.6786 | 0.2708 | 0.4802 | 4.1023 |

Mode A, 514,616 units per target:

| rung | MIN | PTS | REB | AST | FG3M | TOV | PRA |
|---|---|---|---|---|---|---|---|
| L0 | 6.5468 | 4.4195 | 1.8847 | 1.2499 | 0.5190 | 0.7349 | 6.3259 |
| L1 | 5.2880 | 3.5463 | 1.4856 | 0.9137 | 0.3976 | 0.6479 | 4.9794 |
| L2 (h=5) | **3.8969** | 3.2460 | 1.4352 | 0.8649 | 0.3760 | 0.6379 | 4.5804 |
| L3 | 4.1154 | 3.0871 | 1.3604 | 0.8341 | **0.3692** | **0.6179** | 4.1875 |
| L4 | 4.1154 | **3.0851** | **1.3600** | **0.8331** | 0.3697 | 0.6189 | **4.1864** |
| L4-plugin | 4.1154 | 3.0843 | 1.3594 | 0.8325 | 0.3695 | 0.6185 | 4.1852 |

Bold = lowest among the credit-chain rungs (L4-plugin is a diagnostic variant, not a rung). The full
set of secondary and diagnostic metrics for every rung and the four L2 half-lives is in
results/p1_ladder/summary_tables.md sections A.U and A.A. Log score in bits is preserved there as the
frozen framework states it (not interpreted here).

### 3.2 Credit chain, core skill (equal-weight mean over the seven targets)

Every comparison below: regular season, development window 2000-2020, 3,378 game-day clusters, 95%
cluster-bootstrap CI, Holm-adjusted p within the four-comparison family, MPE 1%.

Mode U:

| candidate | reference | kind | core effect | 95% CI | p (Holm) | raw decide() | no-harm violations | calibration-floor violations | final decision |
|---|---|---|---|---|---|---|---|---|---|
| L1 | L0 | immediately simpler | +20.49% | [+19.78%, +21.25%] | 0.0020 | SUPPORTED | none | AST, MIN, PRA, PTS, REB, TOV | **PROMISING** |
| L2 | L1 | immediately simpler | +11.76% | [+11.61%, +11.93%] | 0.0020 | SUPPORTED | none | none | **SUPPORTED** |
| L3 | L2 | immediately simpler | -0.17% | [-0.26%, -0.07%] | 1.0000 | HARMFUL | rps:MIN, rps:FG3M | n/a | **HARMFUL** |
| L4 | L3 | immediately simpler | +0.03% | [+0.01%, +0.04%] | 0.0080 | FALSIFIED | rps:TOV | n/a | **FALSIFIED** |
| L4 | L2 | best simpler | -0.14% | [-0.23%, -0.04%] | (not in the Holm family) | | | | reference for anything built on L4 in mode U is L2 (credit rule: L3 and L4 are non-additive) |

Mode A:

| candidate | reference | kind | core effect | 95% CI | p (Holm) | raw decide() | no-harm violations | calibration-floor violations | final decision |
|---|---|---|---|---|---|---|---|---|---|
| L1 | L0 | immediately simpler | +20.51% | [+20.29%, +20.74%] | 0.0020 | SUPPORTED | log:MIN | AST, MIN, PRA, PTS, REB | **PROMISING** |
| L2 | L1 | immediately simpler | +8.36% | [+8.16%, +8.57%] | 0.0020 | SUPPORTED | log:TOV | AST, PRA, PTS, REB, TOV | **PROMISING** |
| L3 | L2 | immediately simpler | +3.08% | [+2.93%, +3.23%] | 0.0020 | SUPPORTED | log:MIN, rps:MIN | MIN | **PROMISING** |
| L4 | L3 | immediately simpler | -0.01% | [-0.03%, +0.01%] | 0.7876 | FALSIFIED | rps on PTS, REB, AST, FG3M, TOV, PRA | n/a | **FALSIFIED** |

In mode A the best simpler rung for L4 is L3 itself, so no separate best-simpler row exists.

Core skill vs L0: mode U L1 +20.49%, L2 +29.80%, L3 +29.72%, L4 +29.74%; mode A L1 +20.51%,
L2 +27.15%, L3 +29.57%, L4 +29.55%.

### 3.3 Per-target CRPS skill of each step (95% cluster-bootstrap CI)

Mode U:

| step | MIN | PTS | REB | AST | FG3M | TOV | PRA |
|---|---|---|---|---|---|---|---|
| L1 vs L0 | +20.90% [+19.76, +22.10] | +20.60% [+19.79, +21.46] | +20.63% [+19.90, +21.42] | +24.12% [+23.53, +24.77] | +20.47% [+20.18, +20.79] | +14.55% [+13.92, +15.19] | +22.14% [+21.31, +23.08] |
| L2 vs L1 | +21.41% [+21.14, +21.75] | +13.50% [+13.32, +13.71] | +10.28% [+10.12, +10.48] | +9.59% [+9.40, +9.79] | +6.60% [+6.41, +6.78] | +6.89% [+6.74, +7.05] | +14.05% [+13.84, +14.27] |
| L3 vs L2 | **-3.24% [-3.44, -3.06]** | +0.21% [+0.09, +0.33] | +0.01% [-0.12, +0.14] | +0.18% [+0.07, +0.30] | -0.23% [-0.31, -0.14] | +0.61% [+0.53, +0.69] | +1.31% [+1.14, +1.51] |
| L4 vs L3 | +0.00% (identical) | -0.03% [-0.04, -0.01] | -0.04% [-0.07, -0.02] | +0.12% [+0.10, +0.15] | +0.21% [+0.17, +0.25] | -0.03% [-0.09, +0.02] | -0.05% [-0.07, -0.04] |
| L4 vs L2 (best simpler) | -3.24% [-3.44, -3.07] | +0.18% [+0.06, +0.30] | -0.04% [-0.17, +0.11] | +0.30% [+0.20, +0.42] | -0.02% [-0.11, +0.07] | +0.58% [+0.49, +0.66] | +1.26% [+1.09, +1.45] |

Mode A:

| step | MIN | PTS | REB | AST | FG3M | TOV | PRA |
|---|---|---|---|---|---|---|---|
| L1 vs L0 | +19.23% [+18.81, +19.67] | +19.76% [+19.45, +20.08] | +21.18% [+20.97, +21.41] | +26.89% [+26.64, +27.14] | +23.39% [+23.19, +23.60] | +11.84% [+11.66, +12.05] | +21.29% [+20.97, +21.62] |
| L2 vs L1 | +26.31% [+25.95, +26.74] | +8.47% [+8.24, +8.73] | +3.39% [+3.12, +3.67] | +5.35% [+5.14, +5.58] | +5.42% [+5.24, +5.61] | +1.53% [+1.38, +1.71] | +8.01% [+7.63, +8.39] |
| L3 vs L2 | **-5.61% [-5.86, -5.36]** | +4.90% [+4.74, +5.05] | +5.21% [+4.96, +5.46] | +3.56% [+3.40, +3.71] | +1.81% [+1.72, +1.90] | +3.14% [+3.04, +3.25] | +8.58% [+8.26, +8.91] |
| L4 vs L3 | +0.00% (identical) | +0.06% [+0.04, +0.08] | +0.03% [-0.01, +0.06] | +0.12% [+0.08, +0.15] | -0.14% [-0.18, -0.09] | -0.17% [-0.23, -0.10] | +0.03% [+0.01, +0.05] |

L4 vs L3 on MIN is exactly zero in both modes because the shrinkage of L4 acts on the per-minute rates
only; the minutes component is shared (observed in run 1, unchanged, stands as a property of the frozen
implementation).

### 3.4 L2 half-life grid and selection

Core skill vs L1 over the whole development window: mode U h=5 +11.76%, h=10 +10.80%, h=20 +9.39%,
h=40 +7.62%; mode A +8.36%, +8.36%, +7.79%, +6.97%. The frozen selection rule (lowest core CRPS ratio
vs L1 on the scored seasons strictly before s; burn-in 1998-1999 for s = 2000) chose h = 5 in every one
of the 21 seasons; the criterion ratio was monotone in h in every season (summary_tables.md section C).
Reported as: **OPTIMUM AT PRE-REGISTERED SEARCH BOUNDARY - wider search not tested in P1.** No claim is
made that 5 appearances is the true optimum.

### 3.5 L4 posterior predictive vs plug-in (diagnostic, not a rung)

Core skill of L4-plugin vs L4: mode U -0.05% [-0.05%, -0.04%]; mode A +0.04% [+0.04%, +0.04%].
Integrating the gamma posterior changes the core CRPS by less than a tenth of a percent in either
direction.

## 4. Calibration (randomised PIT, seed 0, regular season 2000-2020)

Dispersion ratio = 12 Var(PIT): 1 = calibrated spread, > 1 = forecasts too narrow (under-dispersed),
< 1 = too wide. Bias = mean forecast minus mean outcome. Full tables: summary_tables.md D.U, D.A, E.

Mode U (816,775 units):

| target | L0 cov80 / disp / KS / bias | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| MIN | 0.822 / 0.963 / 0.030 / +0.90 | 0.644 / 1.346 / 0.137 / +2.36 | 0.728 / 1.126 / 0.098 / +1.90 | 0.794 / 0.952 / 0.076 / +2.19 | 0.795 / 0.950 / 0.076 / +2.19 |
| PTS | 0.799 / 1.011 / 0.019 / +0.14 | 0.679 / 1.251 / 0.130 / +0.82 | 0.780 / 1.026 / 0.058 / +0.64 | 0.765 / 1.063 / 0.071 / +0.78 | 0.766 / 1.057 / 0.073 / +0.77 |
| REB | 0.802 / 0.999 / 0.019 / +0.12 | 0.738 / 1.123 / 0.090 / +0.40 | 0.823 / 0.933 / 0.047 / +0.30 | 0.788 / 1.005 / 0.054 / +0.37 | 0.790 / 1.000 / 0.058 / +0.36 |
| AST | 0.801 / 0.993 / 0.014 / +0.05 | 0.760 / 1.086 / 0.056 / +0.15 | 0.819 / 0.947 / 0.029 / +0.15 | 0.801 / 0.989 / 0.030 / +0.17 | 0.806 / 0.973 / 0.038 / +0.17 |
| FG3M | 0.784 / 1.035 / 0.030 / -0.07 | 0.787 / 1.030 / 0.011 / -0.01 | 0.809 / 0.975 / 0.012 / +0.04 | 0.802 / 0.995 / 0.017 / +0.05 | 0.809 / 0.977 / 0.019 / +0.04 |
| TOV | 0.807 / 0.975 / 0.027 / +0.08 | 0.784 / 1.024 / 0.068 / +0.17 | 0.826 / 0.935 / 0.029 / +0.11 | 0.813 / 0.964 / 0.032 / +0.13 | 0.818 / 0.951 / 0.040 / +0.13 |
| PRA | 0.799 / 1.014 / 0.023 / +0.31 | 0.684 / 1.249 / 0.124 / +1.38 | 0.787 / 0.996 / 0.059 / +1.10 | 0.772 / 1.036 / 0.067 / +1.32 | 0.774 / 1.033 / 0.067 / +1.30 |

Mode A (514,616 units):

| target | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| MIN | 0.840 / 0.897 / 0.038 / +0.30 | 0.611 / 1.447 / 0.109 / -0.39 | 0.753 / 1.109 / 0.026 / +0.02 | 0.891 / 0.767 / 0.059 / +0.11 | 0.891 / 0.767 / 0.059 / +0.11 |
| PTS | 0.803 / 0.985 / 0.017 / -0.23 | 0.678 / 1.292 / 0.073 / -0.30 | 0.833 / 0.871 / 0.094 / -0.06 | 0.808 / 0.976 / 0.038 / -0.02 | 0.809 / 0.974 / 0.037 / -0.04 |
| REB | 0.810 / 0.973 / 0.010 / -0.01 | 0.761 / 1.092 / 0.027 / -0.08 | 0.878 / 0.759 / 0.105 / -0.02 | 0.831 / 0.919 / 0.029 / +0.01 | 0.831 / 0.918 / 0.026 / -0.00 |
| AST | 0.804 / 0.988 / 0.011 / -0.02 | 0.764 / 1.084 / 0.037 / -0.11 | 0.844 / 0.864 / 0.075 / -0.02 | 0.822 / 0.941 / 0.032 / -0.02 | 0.828 / 0.930 / 0.027 / -0.02 |
| FG3M | 0.781 / 1.044 / 0.061 / -0.14 | 0.776 / 1.048 / 0.040 / -0.12 | 0.809 / 0.968 / 0.034 / -0.01 | 0.802 / 0.994 / 0.014 / -0.01 | 0.809 / 0.978 / 0.014 / -0.02 |
| TOV | 0.808 / 0.983 / 0.017 / +0.05 | 0.786 / 1.036 / 0.014 / +0.02 | 0.842 / 0.877 / 0.072 / -0.00 | 0.827 / 0.930 / 0.034 / +0.01 | 0.832 / 0.922 / 0.025 / +0.01 |
| PRA | 0.808 / 0.977 / 0.015 / -0.26 | 0.683 / 1.278 / 0.065 / -0.49 | 0.876 / 0.733 / 0.130 / -0.09 | 0.838 / 0.907 / 0.035 / -0.02 | 0.839 / 0.905 / 0.034 / -0.06 |

Findings, measured and not corrected:

- **Systematic bias.** In mode U every rung above L0 forecasts too high on MIN (+1.9 to +2.4 min),
  PTS (+0.6 to +0.8), PRA (+1.1 to +1.4) and, less, REB/AST/TOV. In mode A the same rungs are nearly
  unbiased (L3/L4 within +/-0.1 on every target). The only component that exists in mode U and not in
  mode A is the appearance probability; the pattern is consistent with p_appear being too high for
  rostered players who do not play. Recorded as a future question (not tested).
- **Excessive sharpness (under-dispersion).** L1 everywhere, severely on MIN (dispersion 1.35 U,
  1.45 A; 80% intervals cover 64% / 61%), PTS and PRA (1.25-1.29). L2 keeps a milder version on MIN
  (1.13 U, 1.11 A).
- **Excessive dispersion.** L2 on the counting targets in mode A (PRA 0.73, REB 0.76, PTS 0.87, AST
  0.86, TOV 0.88; 80% intervals cover 83-88%); L3/L4 on MIN in mode A (0.77, cov80 0.891). L3/L4 on
  the counting targets in mode A are mildly wide (0.90-0.98).
- **Tail behaviour.** Upper-tail twCRPS (summary_tables.md A.*) improves monotonically L1 -> L3 -> L4
  on PTS/REB/AST/PRA, but L2 (h = 5) is worse than L1 in the upper tail on every tail target in both
  modes (PTS 0.184 vs 0.157 in U) while winning on CRPS: the 5-appearance window buys centre accuracy
  at the cost of the upper tail.
- **Population-specific failures (low_sample stratum, < 10 as-of appearances; table E).** Mode U:
  all rungs near nominal (cov80 0.80-0.82). Mode A (14,091 units): every rung is under-dispersed and
  under-covers: L2 cov80 0.65-0.76, L3 0.68-0.76, L4 0.71-0.80, L4-plugin 0.70-0.80. The posterior
  predictive (L4) raises cov80 over the plug-in by +1.1 points on PTS (0.696 -> 0.707), +0.9 REB,
  +1.0 PRA, +0.3 AST, +0.3 FG3M, -0.1 TOV.
- **Where calibration improves along the ladder.** L1 -> L2 repairs most of the under-dispersion;
  L2 -> L3 brings PTS/REB/AST/FG3M/TOV/PRA in mode A to within 0.04 of nominal 80% coverage and cuts
  KS from 0.07-0.13 to 0.01-0.04. L0's PIT is near-uniform by construction (it is the empirical
  distribution), which is why the frozen calibration floor downgrades L1 vs L0.
- **Where it stays poor.** MIN in mode U keeps a +2 min bias at every rung; MIN in mode A is too
  narrow at L1/L2 and too wide at L3/L4; the low-sample population in mode A is under-covered at every
  rung.

## 5. Frozen strata

### 5.1 Per season (mode U, regular; core skill vs previous rung)

| season | clusters | L1 vs L0 | L2 vs L1 | L3 vs L2 | L4 vs L3 | L4 vs L0 |
|---|---|---|---|---|---|---|
| 2000 | 162 | +35.37% | +9.76% | -0.81% | -0.17% | +40.94% |
| 2001 | 164 | +32.64% | +12.42% | -0.95% | -0.08% | +40.24% |
| 2002 | 163 | +32.92% | +12.18% | -0.84% | -0.04% | +40.44% |
| 2003 | 164 | +33.16% | +11.89% | -0.56% | -0.08% | +40.55% |
| 2004 | 164 | +20.71% | +13.86% | -1.68% | +0.01% | +30.56% |
| 2005 | 164 | +18.78% | +12.33% | -0.82% | -0.01% | +28.22% |
| 2006 | 164 | +19.87% | +12.72% | -0.23% | -0.07% | +29.85% |
| 2007 | 163 | +17.46% | +12.57% | -0.71% | -0.03% | +27.38% |
| 2008 | 162 | +18.31% | +12.68% | +0.54% | -0.04% | +29.06% |
| 2009 | 163 | +17.05% | +11.80% | -0.17% | -0.01% | +26.76% |
| 2010 | 164 | +18.29% | +10.84% | +0.16% | -0.10% | +27.23% |
| 2011 | 164 | +17.59% | +11.40% | +0.20% | -0.02% | +27.18% |
| 2012 (lockout) | 120 | +12.71% | +12.72% | -0.33% | +0.06% | +23.72% |
| 2013 | 163 | +15.29% | +12.56% | +0.12% | +0.01% | +26.13% |
| 2014 | 163 | +15.03% | +12.86% | +0.00% | +0.07% | +26.10% |
| 2015 | 162 | +14.54% | +10.86% | +0.02% | +0.18% | +24.08% |
| 2016 | 161 | +15.40% | +12.35% | +0.54% | +0.10% | +26.42% |
| 2017 | 162 | +16.44% | +11.68% | +0.95% | +0.09% | +27.04% |
| 2018 (14 s reset) | 168 | +15.84% | +11.33% | +0.06% | +0.14% | +25.59% |
| 2019 | 168 | +17.28% | +9.93% | +0.34% | +0.15% | +25.91% |
| 2020 (bubble) | 150 | +16.75% | +8.78% | +0.03% | +0.27% | +24.33% |

The L1-vs-L0 gain is 33-35% in 2000-2003 and 13-21% from 2004 on; the step coincides with the change
in the as-of roster population (DI-2, section 8). L2 vs L1 is 9-14% in every season. L3 vs L2 is
negative in 10 of the first 13 seasons (2000-2012) and non-negative in every season from 2013 on
(largest +0.95% in 2017). L4 vs L3 is within +/-0.3% everywhere. The interrupted seasons (2012, 2020) and the
post-rule-change season (2018) show no pattern beyond their smaller cluster counts.

### 5.2 Postseason (mode U, 942 clusters)

L1 vs L0 +28.10%, L2 vs L1 +16.40%, L3 vs L2 +1.79%, L4 vs L3 -0.18%. Both recency (L2) and the
minutes-times-rate structure (L3) gain more in the postseason than in the regular season.

### 5.3 low_sample stratum (< 10 as-of appearances): L4 vs L3

| mode / stratum | clusters | core effect | 95% CI | MIN | PTS | REB | AST | FG3M | TOV | PRA |
|---|---|---|---|---|---|---|---|---|---|---|
| U, low | 3,330 | +1.97% | [+1.63%, +2.30%] | 0 | +0.64% | +1.98% | +2.54% | +3.88% | +5.01% | -0.28% |
| U, rest | 3,378 | -0.01% | [-0.03%, +0.00%] | 0 | -0.04% | -0.09% | +0.08% | +0.15% | -0.14% | -0.05% |
| A, low | 2,712 | +4.53% | [+4.06%, +4.99%] | 0 | +4.46% | +5.28% | +5.28% | +5.69% | +7.68% | +3.29% |
| A, rest | 3,378 | -0.10% | [-0.12%, -0.09%] | 0 | -0.03% | -0.09% | +0.02% | -0.24% | -0.34% | -0.05% |

Shrinkage helps only where there is little player history, and there by 2-8% per counting target.

### 5.4 role_change_alarm stratum (CUSUM usage alarm in the last 10 appearances): L2 vs L1

| mode / stratum | clusters | core effect | 95% CI |
|---|---|---|---|
| U, alarm | 3,375 | +20.09% | [+19.50%, +20.74%] |
| U, rest | 3,378 | +11.14% | [+11.00%, +11.31%] |
| A, alarm | 3,357 | +13.20% | [+12.61%, +13.87%] |
| A, rest | 3,378 | +8.02% | [+7.83%, +8.24%] |

This is the X-L2-L1 note only. No H6 statistic of any kind was computed (no interaction, no
simulator, no L6/L12).

## 6. P1 sanity gates (docs/prereg/03; a failure "means a bug" until shown otherwise)

| gate | result | detail |
|---|---|---|
| L1 >> L0 on every core target (expected 35-55%) | **FAIL** | per-target L1 vs L0 (mode U): MIN 20.9%, PTS 20.6%, REB 20.6%, AST 24.1%, FG3M 20.5%, TOV 14.6%, PRA 22.1%. The implemented check (summarize.py, written before run 1's outcomes) requires > 20% on every target; TOV fails it, and every target is below the 35% floor of the expectation. Per season, 2000-2003 are inside 33-35% (section 5.1); the drop from 2004 coincides with DI-2 |
| L3 >= L2 on MIN in mode U | **FAIL** | L3 vs L2 on MIN = -3.24% [-3.44%, -3.06%] (mode A: -5.61%). Context: under M1 the corpus has no INJURY_REPORT, so L3's availability input reduces to the player's own appearance history, and its minutes estimator (frozen defaults half-life 15 appearances, prior strength 150) moves more slowly than the selected L2 (half-life 5). L3 is nonetheless the better distribution on MIN by coverage in mode U (0.794 vs 0.728) |
| L4-plugin cov80 in low_sample <= L4 cov80 | PASS | PTS, mode A, low_sample: plug-in 0.696 <= posterior predictive 0.707 |

Neither failure was investigated beyond the frozen strata (no diagnostic analysis is authorised in P1),
and nothing was changed in response. Both are left for an authorised diagnostic phase.

## 7. Pre-data expectations vs observed (docs/prereg/04, registry/expectations.yaml; nothing rewritten)

| PRE-DATA EXPECTATION | OBSERVED P1 RESULT | AGREEMENT / CONTRADICTION | NOTES |
|---|---|---|---|
| X-L1-L0: core skill of L1 vs L0 in [35%, 55%], **strong** | +20.49% U [19.78, 21.25]; +20.51% A | **CONTRADICTION** (outside a strong range) | 2000-2003 seasons sit at 33-35%, inside the range; the step down in 2004 coincides with DI-2. The gate built on this expectation fails |
| "MIN in mode U gains most" from player identity (X-L1-L0 note) | per-target L1 vs L0 in U: AST +24.1% largest, MIN +20.9% mid-pack, TOV +14.6% smallest | **CONTRADICTION** | |
| X-L2-L1: core skill of L2 vs L1 in [0%, 3%], weak | +11.76% U [11.61, 11.93]; +8.36% A | **CONTRADICTION** (far above) | recency is worth roughly four times the upper bound expected; MIN alone +21.4% U / +26.3% A |
| X-L2-L1 note: 5-15% inside role-change strata | role_change_alarm: +20.09% U [19.50, 20.74], +13.20% A; rest +11.14% U, +8.02% A | **CONTRADICTION** on magnitude in U (above 15%); the direction (alarm > rest) agrees; A inside the range | X-L2-L1 note only; no H6 statistic |
| X-L3-L2: core skill of L3 vs L2 in [2%, 8%], weak; "MIN (mode U) alone 5-15%" | core: -0.17% U [-0.26, -0.07] HARMFUL; +3.08% A [2.93, 3.23]; MIN: -3.24% U, -5.61% A | **CONTRADICTION** in mode U (core and MIN); mode A core inside the range; MIN contradicted in both modes | the availability information the expectation assumed does not exist in the corpus (M1); the compound minutes x rate structure still gains 1.8-8.6% per counting target in mode A |
| X-L3-L2 context (04 headline): availability-aware minutes = a "large" gain | see above | **CONTRADICTION** | |
| X-L4-L3: core skill of L4 vs L3 in [0.5%, 2%], weak | +0.03% U [0.01, 0.04]; -0.01% A [-0.03, 0.01]; FALSIFIED at MPE 1% in both modes | **CONTRADICTION** (below the range; confidently below practical relevance) | |
| X-L4-L3 note: 3-10% for players with < 10 appearances | low_sample: +1.97% U [1.63, 2.30]; +4.53% A [4.06, 4.99] | mode A **AGREEMENT**; mode U below the range | per counting target in A: +3.3% to +7.7% |
| 04 headline: parameter uncertainty overall +/-0.3% ("tiny") | L4-plugin vs L4: -0.05% U, +0.04% A | **AGREEMENT** | |
| 04 headline / H9-type range: low-sample 80% coverage plug-in 70-76% -> posterior predictive 77-82% | PTS mode A low_sample: plug-in 69.6% -> L4 70.7%; REB 76.6 -> 77.5; AST 80.0 -> 80.3; FG3M 79.8 -> 80.1; TOV 78.3 -> 78.2; PRA 72.5 -> 73.5 | direction mostly agrees; magnitude far smaller (about +1 point, not +6); plug-in PTS below its range | reported as the P1 diagnostic only; H9 itself concerns L12 vs L11 and is not examined |
| 03 sanity gate: L1 >> L0 on every target | FAIL (section 6) | **CONTRADICTION** | |
| 03 sanity gate: L3 >= L2 on MIN in U | FAIL (section 6) | **CONTRADICTION** | |
| 03 sanity gate: L4-plugin coverage <= L4 in low_sample | PASS | AGREEMENT | |
| 03 synthetic ordering on MIN (L0 > L1 > L2 > L3 = L4 CRPS) | real data: L0 8.03 > L1 6.35 > L2 4.99 < L3 5.16 = L4 5.16 | **CONTRADICTION** at the L2 -> L3 step | |

Contradictions are reported as stated; none is softened by re-reading the expectation.

## 8. The eleven questions

1. **L0 -> L1.** +20.5% core CRPS skill in both modes (per target 14.6% TOV to 24.1% AST in U). Raw
   SUPPORTED, final PROMISING because L1's distributions are much worse calibrated than climatology
   (frozen calibration floor on six targets; in mode A also a harmful log score on MIN).
2. **L1 -> L2.** +11.8% U (SUPPORTED), +8.4% A (PROMISING: harmful log score on TOV and the floor on
   five targets). The largest single step after identity; MIN +21%/+26%.
3. **L2 -> L3.** -0.17% U (HARMFUL; MIN -3.2% outweighs +0.2 to +1.3% on PTS/AST/TOV/PRA);
   +3.08% A (PROMISING; +1.8 to +8.6% on every counting target, MIN -5.6%).
4. **L3 -> L4.** +0.03% U, -0.01% A: FALSIFIED at the 1% MPE in both modes. Inside the low-sample
   stratum +2.0% U / +4.5% A.
5. **Targets that improve most quickly.** MIN (L0 -> L2: CRPS 8.03 -> 4.99 in U, 6.55 -> 3.90 in A)
   and PRA (6.21 -> 4.10 U); AST gains most from identity alone.
6. **Targets that remain difficult.** TOV (smallest identity gain, 14.6%; smallest recency gain in A,
   1.5%; L1 -> L4 only 7.4% in U) and FG3M (6.6% recency gain, L3/L4 gains within 0.2%); in absolute
   terms MIN in mode U keeps a CRPS of 5.0-5.2 minutes and a +2 min bias at every rung.
7. **Where calibration improves.** L1 -> L2 removes most of the under-dispersion on every target;
   L2 -> L3 brings the counting targets in mode A to within 0.04 of nominal 80% coverage and KS
   0.01-0.04; L3 -> L4 slightly improves coverage in the low-sample stratum (about +1 point).
8. **Where it remains poor.** MIN at every rung (biased high in U; too narrow at L1/L2 and too wide at
   L3/L4 in A); the positive mode-U bias on PTS/PRA at every rung; the low-sample population in mode A
   (80% intervals cover 65-80% at every rung); L2's upper tail (worse than L1 on every tail target).
9. **Simple information sources carrying most of the usable signal.** Player identity (20.5 points of
   the 29.8-point core gain in U) and recency with a short window (another 9.3 points in U, 6.6 in A);
   together they are 100% of the mode-U gain of the whole ladder (L4 is not better than L2 in U). The
   minutes x rate decomposition adds 2.4 points in mode A only. Availability information could not be tested (M1).
10. **Complex-looking additions with effectively zero gain.** Hierarchical shrinkage (L4 vs L3:
    +0.03% / -0.01%, FALSIFIED) outside the low-sample stratum, and posterior-predictive integration
    (L4 vs L4-plugin, +/-0.05%). In mode U the minutes-rate structure (L3) is a net loss.
11. **Contradicted expectations.** Yes: four of the five ladder-gap expectations (section 7),
    including the only *strong* one (X-L1-L0), the "MIN gains most from identity" note, and two of the
    three sanity gates. The parameter-uncertainty and low-sample (mode A) expectations agree.

## 9. New data issues discovered (data failure vs model failure)

| id | what | class | evidence | consequence in P1 |
|---|---|---|---|---|
| DI-1 | 17 nba_stats schedule rows name the placeholder team '0' (13 preseason, 2 all-star, 2 "playoffs" rows in 2005 with no box score); no roster spell can exist, the as-of request set is empty and the frozen MinutesRate raises on it | **data (structural)**; mismatch M24, SCOPE-RESTRICTION | results/p1_ladder/di1_evidence/ (original traceback from run 1, launch log lines, probe), di1_verification.json (all 23 seasons post-patch, 0 requests involving team '0'), done_2005.json excluded_games | deterministic exclusion `drop_unknown_team_games` applied identically to every season and to the leakage tips; 2 games excluded (2005 postseason); criterion = structural invalidity, never a forecast or score; no other outcome-based exclusion exists |
| DI-2 | the as-of ROSTER request population steps down in 2004: 41-42 requests per game in 1998-2003, 28-33 from 2004 on (section 1 table); the L1-vs-L0 gain steps down at the same season (33-35% -> 13-21%) | **data observation (unclassified)**; not a model failure (identical frozen code in every season) | done_{S}.json request counts; strata per season | none in P1: recorded here and in EXPOSURE_LOG row 10 as a structural count; **future question**: which corpus roster-spell source changes in 2004 (nba_stats roster coverage vs appearance-derived spells), and whether the pre-2004 population carries rostered-but-inactive players that make climatology weaker |
| (known) M1 | no INJURY_REPORT in the corpus | mismatch recorded at onboarding | registry row M1 | L3 runs as frozen with no availability information; bears on the L3 results and the failed MIN gate |
| (not a data issue) | twCRPS = 0 for MIN, FG3M, TOV | by design (thresholds frozen only for PTS/PRA/REB/AST) | 08 metric spec | none |

No impossible minutes distributions, missing targets, duplicated player-games or corrupted segments
were met in the run (coverage assertion passed, 0 missing / 0 extra games in all 23 seasons; no
LockboxError, no traceback in any season log).

## 10. Mismatches, bugfixes, amendments (governance classification)

| id | class | what | when relative to outcomes |
|---|---|---|---|
| M24 | mismatch, SCOPE-RESTRICTION (structural) | DI-1 placeholder-team rows; exclusion rule in experiments/p1_ladder/exclusions.py | found during run 1 (2005 crash); the rule depends on team identity only |
| M1 | mismatch (pre-existing) | INJURY_REPORT absent; L3 degenerate as frozen | onboarding, before any outcome |
| C9 | bugfix (adapter) | roster spells never gap-closed (unit mismatch); re-attested | before any outcome |
| C11 | bugfix (driver) | fixed 7-day chunks skipped every 7th game day; lag-derived 6-day chunks + coverage assertion; run 1 archived and superseded, all 23 seasons re-run | after run 1's preliminary observations; no model/metric/stratum/threshold touched; the preliminary observations (L3 < L2 on MIN U; h = 5 at the boundary; L4 = L3 on MIN; larger L2 gain under role-change alarm) all recur in run 2 and stand as observations |
| C15 | bugfix (summary script) | the frozen constraints of 08 section 5 (no-harm, calibration floor) were listed but not applied to the decision labels; experiments/p1_ladder/constraints.py applies them as a pure function of summary.json; raw labels retained | after run 2's raw labels were seen; the constraints and the 0.02 floor are frozen text, nothing was tuned |
| C10, C12, C13, C16 | engineering | git-ignore of large artifacts; new P1 scripts; this report and the committed summaries | - |
| C14 | governance | attestation refresh after M24 | - |
| amendments | **none** | no frozen tier-1/tier-2 file changed; `python -m hoopslab.registry verify` INTACT after the run | |

## 11. Confirmations requested before finalising

1. All 23 seasons completed (exit 0, done_{S}.json present); the only structural exclusion is DI-1
   (2 games, 2005), documented.
2. Every authorised leakage detector (T1, T2, T3, T6, T7) completed over the full intended range
   (1998-2020, 20 tips per season) and passed.
3. T5 static scan passes.
4. The holdout was never accessed (preflight 5_holdout_sealed on both machines: 0 rows >= 2023; the
   package physically excludes in-game data >= 2023; the lockbox 2021-2022 guard never fired).
5. All P1 outputs come from one driver and one exclusion implementation (run_season.py sha
   39800ecf..., exclusions.py; di1_verification.json: every season post-patch, consistent).
6. The final summary (summary.json / summary_tables.md) was regenerated from the complete 23-season set
   (21 scored); the preliminary 8-season summary was not used.
7. The preliminary summary is retained for audit under results/p1_ladder_run1_chunkbug/ (README) and
   excluded from every table here.
8. Code/config hashes (section 1) and EXPOSURE_LOG rows 8-10 are current.
9. Registry verification and the complete test suite pass after the run (recorded in the commit
   message of this report).
10. No unlogged tier-1/tier-2 change: preflight check 9 passed on both machines before the run; the
    post-run verify is INTACT; every post-freeze edit has a POSTFREEZE_CHANGELOG row (C9-C16).

## 12. Future research questions recorded (not tested in P1)

- Why the as-of roster population and the L1-vs-L0 gain both step in 2004 (DI-2).
- Whether the positive mode-U bias at every rung is an appearance-probability bias.
- Whether L2's half-life optimum lies below 5 appearances (boundary result).
- Why L3's minutes estimator loses to L2's recency estimator on MIN while its rates win on the
  counting targets in mode A (frozen defaults half-life 15 / prior 150 vs the selected h = 5).
- Whether L2's upper-tail loss (twCRPS worse than L1) is a window-length effect.
- The low-sample population's under-coverage in mode A at every rung.

Stop. No L5, no entropy/history experiments, no possession models, no H6 analysis, no holdout access.
