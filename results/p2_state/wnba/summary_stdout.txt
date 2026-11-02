# E-STATE portability summary (WNBA wnba_stats segments 2016-2020, test months 2018-05..2020-10, logit only; spec section 7)

## 1. Frozen multinomial logit: events 86,821, games 488, months 15

| state | blocks | columns (median) | held-out bits/event | add-one gain (mb) [CI] | residual history r_k (mb) [CI] | converged |
|---|---|---|---|---|---|---|
| S0 | S | 480 | 2.5481 | - | +1.954 [+1.202, +2.759] | 1.00 / 1.00 |
| S1 | + team fouls / bonus | 496 | 2.5401 | +7.999 [+6.832, +9.223] | -0.618 [-1.104, -0.100] | 1.00 / 1.00 |
| S2 | + gap since previous segment | 498 | 2.5233 | +16.868 [+15.216, +18.466] | -0.149 [-0.688, +0.375] | 1.00 / 1.00 |
| S3 | + late clock x margin | 546 | 2.5178 | +5.532 [+4.448, +6.599] | -0.255 [-0.756, +0.247] | 1.00 / 1.00 |
| S4 | + finer score + early clock x margin | 619 | 2.5218 | -4.006 [-4.887, -3.128] | -0.364 [-0.869, +0.124] | 1.00 / 1.00 |
| S5 | + player fouls | 645 | 2.5234 | -1.633 [-2.133, -1.120] | -0.391 [-0.937, +0.133] | 1.00 / 1.00 |
| S6 | + rotation (stints, lineup changes) | 649 | 2.5237 | -0.346 [-0.578, -0.100] | -0.416 [-0.950, +0.096] | 1.00 / 1.00 |
| S7 | + previous segment class | 658 | 2.5236 | +0.151 [-0.260, +0.545] | -0.868 [-1.322, -0.398] | 1.00 / 1.00 |

Bridge to E-HIST (beyond the latent): S0: +2.537 [+1.850, +3.231]; S6: +0.327 [-0.120, +0.770]; S7: -0.118 [-0.461, +0.204]. Latent gain: S0: -0.582 [-0.908, -0.263]; S6: -0.743 [-1.046, -0.435]; S7: -0.749 [-1.055, -0.435]

## 2. Information decomposition (S0 -> S7 + full history = +23.697 [+21.115, +26.261] mb)

| component | mb/event [CI] | share of total |
|---|---|---|
| team fouls / bonus | +7.999 [+6.832, +9.223] | 33.8% |
| gap since previous segment | +16.868 [+15.216, +18.466] | 71.2% |
| late clock x margin | +5.532 [+4.448, +6.599] | 23.3% |
| finer score + early clock x margin | -4.006 [-4.887, -3.128] | -16.9% |
| player fouls | -1.633 [-2.133, -1.120] | -6.9% |
| rotation (stints, lineup changes) | -0.346 [-0.578, -0.100] | -1.5% |
| previous segment class | +0.151 [-0.260, +0.545] | 0.6% |
| remaining within-game history (GL + L1 + L3 + L10 + EW) after S7 | -0.868 [-1.322, -0.398] | -3.7% |

## 3. Previous-class proxy test and short memory

* PC on S0: +0.465 [+0.101, +0.839]; PC on S6 (= g7): +0.151 [-0.260, +0.545]; PC beyond full history on S0: +0.270 [-0.045, +0.597]
* S0 + PC vs S4 (= S0 + fouls + gap + clock x score): +25.928 [+23.432, +28.362] (positive = S4 better)
* full history on S0 + PC: +1.760 [+1.058, +2.471]; on S6: -0.416 [-0.950, +0.096]; on S7: -0.868 [-1.322, -0.398]

## 4. Controls (expected <= 0)

* S6 with its state blocks permuted within cells vs S0: -8.510 [-9.719, -7.367]
* full history permuted within cells on S6: -0.599 [-0.890, -0.302]

## 5. Per season (mb/event)

| season | events | games | g1 FB | g2 GAP | g3 TM | g4 SC | g5 PF | g6 ROT | g7 PC | r0 | r6 [CI] | r7 [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 28,951 | 164 | +10.10 | +16.11 | +4.68 | -5.45 | -1.66 | -0.59 | -0.46 | +2.73 | -0.28 [-1.16, +0.54] | -0.61 [-1.40, +0.09] |
| 2019 | 34,960 | 196 | +8.34 | +14.30 | +6.76 | -2.74 | -1.31 | -0.21 | +0.45 | +2.08 | -0.39 [-1.24, +0.46] | -0.93 [-1.65, -0.25] |
| 2020 | 22,910 | 128 | +4.82 | +21.74 | +4.74 | -4.13 | -2.10 | -0.24 | +0.47 | +0.79 | -0.63 [-1.77, +0.46] | -1.10 [-2.09, -0.14] |

## 6. Where the primary residual r6 sits (logit, pooled means; diagnostic)

**by phase**: 4Q/OT: share 0.246, r0 -0.01, r6 -1.37 (contrib -0.34); periods 1-3: share 0.754, r0 +2.59, r6 -0.10 (contrib -0.08)
**by margin**: 0-5: share 0.466, r0 +1.82, r6 -0.21 (contrib -0.10); 6-10: share 0.256, r0 +1.95, r6 -0.94 (contrib -0.24); 11-15: share 0.146, r0 +2.14, r6 -0.55 (contrib -0.08); 16+: share 0.132, r0 +2.23, r6 +0.02 (contrib +0.00)
**by start_type**: after_dead_ball: share 0.159, r0 +3.56, r6 -0.74 (contrib -0.12); after_dreb: share 0.275, r0 +1.49, r6 -0.87 (contrib -0.24); after_live_turnover: share 0.078, r0 +2.41, r6 -0.57 (contrib -0.04); after_make: share 0.365, r0 +1.33, r6 -0.51 (contrib -0.19); after_oreb: share 0.123, r0 +2.50, r6 +1.39 (contrib +0.17)
**by y**: 0: share 0.147, r0 -2.43, r6 -1.43 (contrib -0.21); 1: share 0.004, r0 -11.65, r6 -2.07 (contrib -0.01); 2: share 0.026, r0 +24.22, r6 +0.60 (contrib +0.02); 3: share 0.056, r0 +20.86, r6 +0.09 (contrib +0.00); 4: share 0.283, r0 +1.73, r6 +0.87 (contrib +0.25); 5: share 0.259, r0 -0.80, r6 -1.69 (contrib -0.44); 6: share 0.147, r0 -0.27, r6 -0.87 (contrib -0.13); 7: share 0.077, r0 +4.02, r6 +1.32 (contrib +0.10)
**by def_in_bonus**: 0: share 0.912, r0 +1.38, r6 -0.43 (contrib -0.39); 1: share 0.088, r0 +7.86, r6 -0.31 (contrib -0.03)

## 8. Frozen H2 (registry decision_override), S* = S0 against R = S7 (logit)

loss +24.565 [+22.124, +27.079] mb; I(S7) over class frequencies +0.477 [-4.675, +5.496] mb; threshold 0.50 mb -> **FALSIFIED**

## 9. Regime (docs/p2/02_estate_spec.md section 6)

{"mnl_r6": "A", "gbm_r6": null, "mnl_r7": "A", "gbm_r7": null, "regime": "A (logit only)"}
