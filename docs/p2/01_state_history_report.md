# P2 report: compact state and within-game history (E-HIST, H3a / H3b) on NBA `nba_stats` possession segments, development seasons 2000-2020

*Exposure class O-dev, event level (EXPOSURE_LOG rows 13-14). Written 2026-10-31 from the complete prequential run
(results/p2_hist/). The P1 L2-A player-stat baseline is the reference throughout and was not retuned. Nothing in the
frozen specification (feature sets, penalty, target, unit, contrasts, MPE, diagnosis rule, controls) was changed in
response to any observed outcome. The only post-freeze changes are the two recorded in POSTFREEZE_CHANGELOG C20-C22:
the optimiser iteration cap (set on convergence flags, before any pooled number existed) and the segment-parser
rebound rule (a data bug found by the pre-registered controls on the timing month, before any pooled number existed).*

## 0. Fixed report block

```
STATE EXPERIMENT STATUS:            PASS. Every pre-registered run completed on corrected segments: frozen
                                    multinomial logit on all 185 prequential months (5,668,402 held-out events,
                                    26,937 games), LightGBM replication on every third month (62 months,
                                    1,891,555 events), GRU sequence family on all 185 months, negative control,
                                    two positive controls, hot-hand contrast, H3b slow-variable attribution on
                                    every sixth month (31 months, 951,008 events), per-season table.
COMPACT STATE DEFINITION:           S (frozen, hoopslab.models.event.EventDesign): start type (after_make,
                                    after_dreb, after_live_turnover, after_oreb; the frozen encoding maps
                                    after_dead_ball to the baseline), offence-is-home flag, score margin in
                                    6-point buckets clipped to +-3 (7 levels), period (1-4, overtime pooled),
                                    clutch flag (last 4 minutes of regulation or later and |margin| <= 5),
                                    offence team, defence team, bag of the 5 on-court offence players, bag of the
                                    5 on-court defence players. Target: the segment's 8-class terminal outcome.
BEST STATE-ONLY MODEL:              frozen L2 multinomial logit, set A: 2.4533 bits/event held out (LightGBM
                                    A 2.4443 on its 62-month subset; neural A_nn 2.4680, 14.7 mb worse than the
                                    logit on identical events). The simplest family is the best state-only model.
HISTORY GAIN (bits/event):          +0.00347 bits/event = +3.47 millibits [+3.39, +3.55] (A -> E, logit,
                                    95% cluster bootstrap over games, B = 2000); LightGBM +4.66 [+4.44, +4.87];
                                    neural family +4.38 [+4.20, +4.57]. Positive in all 21 seasons
                                    (+2.30 .. +4.46 mb), playoffs as regular season.
HISTORY GAIN AFTER GAME LATENT:     +3.35 mb [+3.27, +3.43] (F -> G, logit). The game-level latent itself is
                                    worth +0.11 mb [+0.09, +0.14] (A -> F) and adds +0.001 mb [-0.013, +0.015]
                                    on top of history (E -> G): the latent explains none of the history signal.
FULL-HISTORY MODEL:                 set E (S + L1 + L3 + L10 + EW, logit) 2.4499 bits/event; GRU over the whole
                                    (game, offence) sequence 2.4636 (13.8 mb WORSE than the logit E on identical
                                    events); G_gru 2.4632.
MARKOV-SUFFICIENCY RESULT:          REJECTED for the frozen compact state S at the 0.5 mb MPE. Frozen diagnosis
                                    GENUINE_HISTORY in the pooled logit, in every season, in LightGBM and in
                                    the neural family. H3a SUPPORTED; H3b FALSIFIED (frozen rule: no slow
                                    variable on the pre-registered list brings the residual's CI below the MPE;
                                    team fouls come closest, +3.46 -> +1.03 mb [+0.92, +1.15]).
EVENT-LEVEL GAIN:                   +3.47 mb/event = 0.14% of the state-only cross-entropy; about 0.37 bits per
                                    team-game (105 segments) and 0.73 bits per game. 55% of it sits on the
                                    three free-throw classes (7% of events): +14 to +20 mb each; the turnover
                                    class gains nothing; field-goal classes +1.4 to +2.8 mb.
DOWNSTREAM PLAYER-STAT GAIN:        NONE REALISABLE AT T-60. A pre-game forecast has no within-game history to
                                    condition on; the event-level gain can reach a player-stat forecast only
                                    through an in-game simulator (not built, by instruction) or a live lane
                                    (out of scope). Between-game analogue, EXPLORATORY, not a claim: L2-A plus a
                                    shrunk last-3-game residual (results/p2_hist/downstream_form.md, 490,535
                                    units 2001-2020): MIN +1.67% CRPS skill [+1.63, +1.69], PRA +0.12%,
                                    PTS -0.01%, REB 0.00%, AST +0.01%. H3c proper waits for its reference L8.
DOES HISTORY ADD PRACTICALLY        YES at the event level (unambiguous: 6.9 x MPE, every season, every family,
USEFUL INFORMATION:                 controls behave) but SMALL (0.14%), and mostly not "history": ~70% of the
                                    residual is the team-foul / bonus state, which the compact state omits and
                                    which is itself worth +7.6 mb; the previous segment's class adds +0.62 mb
                                    beyond everything. NO for the pre-game player-stat lane.
DOES SEQUENCE MODELING BEAT         NO. The GRU is 13.8 mb worse than the frozen logit with five summary
SIMPLE STATE MODELS:                history blocks, and the history gain inside the neural family (+4.4 mb) is
                                    the same size as the logit's (+3.5 mb) and LightGBM's (+4.7 mb). Everything
                                    whole-game history carries is captured by L1/L3/L10 plus two exponentially
                                    weighted summaries plus (not frozen) the previous segment's class.
DOES THIS JUSTIFY A FULL            NO, not on this evidence. The information history carries is a few
POSSESSION SIMULATOR:               millibits per event and is mostly a missing slow STATE variable (team fouls
                                    and bonus, +7.6 mb; time since the offence's previous segment, +8.9 mb), which
                                    belongs in E-STATE (H2) and in any future simulator's state, not in a
                                    history mechanism. Whether a simulator pays off at the player-stat level is
                                    H5, untested here. Recommended next step: E-STATE with fouls/bonus, gap and
                                    previous-segment class as candidate Tier-2 blocks; a simulator only after.
P1 L2-A BASELINE:                   UNCHANGED (results/p1_ladder/ untouched; shadow model card hash unchanged)
HOLDOUT:                            UNTOUCHED (0 rows with season >= 2023 in possessions / play_by_play /
                                    player_game_box; lockbox 2021-2022 present in the package, never parsed:
                                    no segments_2021/2022 file exists on either machine)
SHADOW FIREWALL:                    INTACT (no shadow forecast, lock, outcome or score read; the executor
                                    session reported operational status only: "runner alive, no locks yet")
READY FOR NEXT RESEARCH PHASE:      YES. Next phase recommended: E-STATE (H2) with the three blocks above as
                                    candidates; then E-SHANNON. Not yet: possession simulator, entropy
                                    exploitation, H6.
```

## 1. What ran, and how it is reproduced

**Unit and target (frozen, docs/prereg/01 E-HIST).** Possession *segment*: an offensive-rebound continuation is a
separate segment. The supplied `possessions` table holds whole possessions, so segments are parsed from the
corpus play-by-play inside each possession (`experiments/p2_hist/segments.py`; mismatch M18 foresaw this).
Terminal outcome in 8 classes: 0 TO, 1 FT trip 0 of 2, 2 FT 1 of 2, 3 FT 2 of 2 (and-one and technical trips
capped at 2 made), 4 2PA miss, 5 2PA make, 6 3PA miss, 7 3PA make. Parser gate per season: points reconcile
>= 99.36% and field-goal attempts >= 99.98% of the supplied possession totals in all 24 parsed seasons
1997-2020 (results/p2_hist/preflight.json). Seasons 2021-2022 (lockbox) were never parsed; >= 2023 is absent.

**Feature sets (frozen, `hoopslab.experiments.history.SETS`).** A = S; B = S + L1 (offence's previous possession
points); C = + L3; D = + L10; E = D + EW (exponentially weighted history, half-lives 3 and 10 possessions: the
tractable stand-in for whole-game history); F = S + GL (game-level latent: offence's running shooting residual,
shrunk); G = F + L1 + L3 + L10 + EW. Contrasts: history_signal = bits(A) - bits(E), latent_signal = bits(A) - bits(F),
history_residual = bits(F) - bits(G). MPE 0.5 millibit per event. Diagnosis by `diagnose_history` (frozen rule in
section 3).

**Models.** (i) The frozen `MultinomialLogit` (L2 = 1e-4, L-BFGS) on the frozen `EventDesign` blocks; iteration
cap 1000 instead of the class default 300 (C20: sets D/E/G did not converge at 300 on the timing month; the
penalty, objective and features are unchanged; every pooled fit converged, meta .json per month). (ii) LightGBM
replication with fixed parameters written before any outcome (31 leaves, learning rate 0.05, 300 rounds,
min 50 per leaf, lambda_l2 1, feature/bagging fraction 0.8), every third test month. (iii) Neural family with a
fixed architecture written before any outcome (`run_gru.py`): per (game, offence) sequence; input = dense compact
state + summed 16-d player embeddings (offence and defence) + previous segment's class one-hot; GRU hidden 32;
Adam 2e-3, 4 epochs, seed 0. Nested references in the same family: A_nn (no memory, no previous class) and F_nn
(A_nn + GL) so that the neural contrasts compare like with like.

**Protocol.** Prequential by calendar month over development seasons 2000-2020: for each test month every model is
refit on the 730 days before the month's first game day (1997-1999 serve as training history only, never scored);
held-out -log2 P(observed class) per event; 185 months; 5,668,402 scored segments. No plug-in entropy anywhere.
Statistics: paired per-event bit differences, cluster bootstrap over games, B = 2000, seed 12345
(`hoopslab.metrics.compare`), one-sided p for "candidate better".

**Controls (frozen).** Negative: history blocks permuted across events within state cells (start type x margin
bucket x period), refit, must be <= 0. Positive canary 1: the same offence's next segment class (weak, by
construction only one step ahead). Positive canary 2 (`run_canary.py`, every sixth month): the next segment in
the game, either team: its start type and the margin change; must be large. Hot-hand contrast: pooled vs
within-game P(make | previous make) - P(make | previous miss).

**Attribution (frozen H3b step).** On every sixth test month, the slow variables are added one at a time to F and
to G and the residual re-measured (`slow_vars.py`, `run_attrib.py`): team fouls in the period (defence and
offence, one-hot 0-6+) plus the bonus flag; stint length of the offence lineup (log seconds); lineup-change flag;
seconds since the offence's previous segment started (log); all four together; and, beyond the frozen list, the
previous segment's class (PC). Timeouts remaining could not be built: nba_stats timeout rows carry team_id 0
(unattributable), so that slow variable is reported as not testable on this corpus.

**Compute.** Logit months 2000-2015 on the i9 (8 workers), 2016-2020 and attribution on TrayServer
(C:\Users\tdm79\bball\basketball, same venv stack); GBM, GRU, canary 2 on the i9. Per-event bits are git-ignored
(results/p2_hist/bits/, 1.1 GB); pooled summaries are committed (results/p2_hist/summary.json,
summary_tables.md). Reproduce: `python experiments/p2_hist/segments.py --seasons 1997 ... 2020`,
`slow_vars.py`, `launch.py --family mnl|gbm|gru`, `run_attrib.py`, `run_canary.py`, `summarize_hist.py`.
Hashes of the frozen event modules, the experiment scripts, the adapter and the corpus: results/p2_hist/preflight.json.

**Data notes.** 193 of 5,668,402 keys (game_id, possession_seq) collide (possession_seq = possession number x 10 +
continuation index; a possession with more than nine continuations wraps); they affect the GRU-logit join by 432
rows and nothing else. Lineups are complete (5 + 5) in every season. The frozen START encoding has no level for
after_dead_ball (14.6% of segments), which therefore shares the baseline with after_make; kept as frozen.

## 2. Primary results: the frozen logit, pooled over 185 months

Mean held-out bits per event: A 2.4533, B 2.4528, C 2.4526, D 2.4526, E 2.4499, F 2.4532, G 2.4499.

| contrast | millibits/event | 95% CI | p | n |
|---|---|---|---|---|
| gain_B (A -> B, previous possession) | +0.545 | [+0.518, +0.574] | 0.0005 | 5,668,402 |
| gain_C (A -> C, previous 3) | +0.672 | [+0.637, +0.706] | 0.0005 | 5,668,402 |
| gain_D (A -> D, previous 10) | +0.732 | [+0.695, +0.769] | 0.0005 | 5,668,402 |
| gain_E_vs_D (whole-game summaries beyond 10) | +2.734 | [+2.659, +2.808] | 0.0005 | 5,668,402 |
| **history_signal (A -> E)** | **+3.466** | **[+3.386, +3.547]** | 0.0005 | 5,668,402 |
| latent_signal (A -> F) | +0.113 | [+0.087, +0.139] | 0.0005 | 5,668,402 |
| **history_residual (F -> G)** | **+3.354** | **[+3.274, +3.429]** | 0.0005 | 5,668,402 |
| gain_G_vs_E (latent beyond history) | +0.001 | [-0.013, +0.015] | 0.4443 | 5,668,402 |
| NEGATIVE CONTROL: permuted history (E_perm vs A) | -0.038 | [-0.050, -0.026] | 1.0000 | 5,668,402 |
| NEGATIVE CONTROL: permuted history beyond latent (G_perm vs F) | -0.157 | [-0.185, -0.130] | 1.0000 | 5,668,402 |
| POSITIVE CONTROL canary 1 (same offence's next class) | +3.179 | [+3.098, +3.261] | 0.0005 | 5,668,402 |
| POSITIVE CONTROL canary 2 (next segment's start type + margin change; 31 months) | +1,345.4 | [+1,340.0, +1,350.6] | 0.0005 | 951,008 |

Reading. Both negative controls are at or below zero (the permuted blocks cost 0.04-0.16 mb, as a null feature
should under L2). Canary 2 moves the cross-entropy by 1.35 bits per event, so the pipeline sees information when
it is there. Canary 1 is small by construction (one class of one future segment, weak by itself); it is kept
because it was the frozen one. The memory ladder is monotone: one possession +0.55, three +0.67, ten +0.73, and
the whole-game summaries add another +2.73 on top of ten possessions. The game-level latent (running shooting
residual) is worth 0.11 mb and nothing on top of history.

**Frozen diagnosis: GENUINE_HISTORY** (history_residual CI_low +3.27 > 0 and estimate >= MPE).

### 2.1 Per season

| season | events | games | history_signal (mb) [CI] | latent (mb) | history_residual (mb) [CI] | gain_B (mb) | diagnosis |
|---|---|---|---|---|---|---|---|
| 2000 | 267,977 | 1264 | +4.46 [+4.02, +4.91] | +0.15 | +4.34 [+3.92, +4.76] | +0.54 | GENUINE_HISTORY |
| 2001 | 260,893 | 1260 | +4.18 [+3.72, +4.63] | +0.21 | +4.01 [+3.56, +4.43] | +0.64 | GENUINE_HISTORY |
| 2002 | 261,134 | 1260 | +3.60 [+3.15, +4.03] | +0.03 | +3.59 [+3.17, +4.02] | +0.60 | GENUINE_HISTORY |
| 2003 | 265,633 | 1277 | +4.17 [+3.79, +4.56] | +0.07 | +4.02 [+3.65, +4.42] | +0.51 | GENUINE_HISTORY |
| 2004 | 261,790 | 1271 | +3.78 [+3.38, +4.18] | +0.05 | +3.71 [+3.32, +4.11] | +0.75 | GENUINE_HISTORY |
| 2005 | 272,938 | 1314 | +4.36 [+3.96, +4.79] | +0.06 | +4.27 [+3.86, +4.70] | +0.51 | GENUINE_HISTORY |
| 2006 | 270,600 | 1319 | +4.01 [+3.59, +4.44] | +0.01 | +3.98 [+3.56, +4.40] | +0.62 | GENUINE_HISTORY |
| 2007 | 271,707 | 1309 | +3.67 [+3.24, +4.11] | +0.10 | +3.58 [+3.16, +4.02] | +0.54 | GENUINE_HISTORY |
| 2008 | 274,149 | 1316 | +4.02 [+3.63, +4.40] | +0.09 | +3.92 [+3.53, +4.30] | +0.55 | GENUINE_HISTORY |
| 2009 | 271,624 | 1315 | +3.53 [+3.16, +3.91] | +0.25 | +3.36 [+2.99, +3.73] | +0.71 | GENUINE_HISTORY |
| 2010 | 272,643 | 1312 | +3.19 [+2.79, +3.61] | +0.18 | +3.05 [+2.67, +3.46] | +0.53 | GENUINE_HISTORY |
| 2011 | 271,058 | 1311 | +3.50 [+3.12, +3.85] | +0.13 | +3.35 [+2.98, +3.69] | +0.53 | GENUINE_HISTORY |
| 2012 | 221,785 | 1074 | +3.38 [+2.99, +3.77] | +0.18 | +3.17 [+2.79, +3.54] | +0.60 | GENUINE_HISTORY |
| 2013 | 272,908 | 1314 | +2.99 [+2.63, +3.35] | +0.28 | +2.74 [+2.40, +3.08] | +0.42 | GENUINE_HISTORY |
| 2014 | 278,140 | 1319 | +3.24 [+2.87, +3.63] | +0.15 | +3.15 [+2.79, +3.51] | +0.53 | GENUINE_HISTORY |
| 2015 | 277,286 | 1311 | +2.94 [+2.59, +3.29] | +0.05 | +2.89 [+2.55, +3.23] | +0.52 | GENUINE_HISTORY |
| 2016 | 281,323 | 1316 | +3.09 [+2.74, +3.43] | +0.05 | +3.04 [+2.70, +3.37] | +0.63 | GENUINE_HISTORY |
| 2017 | 280,408 | 1309 | +2.74 [+2.40, +3.08] | +0.12 | +2.65 [+2.33, +2.97] | +0.57 | GENUINE_HISTORY |
| 2018 | 284,369 | 1312 | +2.85 [+2.51, +3.19] | +0.07 | +2.70 [+2.37, +3.03] | +0.39 | GENUINE_HISTORY |
| 2019 | 294,201 | 1312 | +2.88 [+2.57, +3.21] | +0.05 | +2.81 [+2.50, +3.12] | +0.36 | GENUINE_HISTORY |
| 2020 | 255,836 | 1142 | +2.30 [+2.00, +2.62] | +0.10 | +2.21 [+1.91, +2.52] | +0.44 | GENUINE_HISTORY |

The signal is present in every season and drifts down over the window (+4.5 mb in 2000 to +2.3 mb in 2020),
consistent with the foul-driven reading in section 4 (fewer free-throw trips per segment in the modern era).
Playoffs (354,665 events) +3.46 mb and regular season +3.47 mb: no difference.

## 3. Replications: LightGBM and the neural sequence family

**LightGBM (62 months, 1,891,555 events).** Mean bits: A 2.4443, B 2.4435, D 2.4432, E 2.4397, F 2.4441, G 2.4397.
history_signal +4.66 mb [+4.44, +4.87]; latent_signal +0.21 [+0.09, +0.33]; history_residual +4.43 [+4.22, +4.62];
canary 1 +4.49. Diagnosis GENUINE_HISTORY. The tree model finds the same structure, slightly more of it.

**Neural family (185 months, identical test events as the logit).** Mean bits: logit A 2.4533 / E 2.4498 /
F 2.4532; A_nn 2.4680, F_nn 2.4680, E_gru 2.4636, G_gru 2.4632.

| contrast | millibits/event | 95% CI | p |
|---|---|---|---|
| history_signal_nn (A_nn -> E_gru) | +4.380 | [+4.199, +4.572] | 0.0005 |
| latent_signal_nn (A_nn -> F_nn) | -0.026 | [-0.181, +0.131] | 0.6187 |
| history_residual_nn (F_nn -> G_gru) | +4.857 | [+4.669, +5.041] | 0.0005 |
| architecture: A_nn vs logit A | -14.715 | [-14.996, -14.420] | 1.0000 |
| architecture: E_gru vs logit E | -13.805 | [-14.106, -13.508] | 1.0000 |

Diagnosis GENUINE_HISTORY. Two readings. First, the history gain is family-independent: +3.5 (logit), +4.7
(GBM), +4.4 (GRU) mb. Second, the sequence model is not better than the simple model: the GRU is 13.8 mb worse
than the logit on E, because the logit's team and player identity blocks (one coefficient per team and player per
class) beat 16-d embeddings trained for 4 epochs, and because the recurrent memory finds nothing that
L1/L3/L10/EW plus the previous class do not already carry. The within-family controls behaved: pooled over the same
events, training and scoring the GRU on order-shuffled sequences costs 4.46 mb against E_gru (the whole gain,
so the gain is sequence information, not capacity), and removing the previous-class input costs 2.23 mb.

**Frozen diagnosis rule** (`diagnose_history`, MPE 0.5 mb): NO_HISTORY_SIGNAL if CI_high(history_signal) < MPE;
GENUINE_HISTORY if CI_low(history_residual) > 0 and estimate >= MPE; APPARENT_HISTORY_EXPLAINED_BY_LATENT if
history_signal >= MPE, latent_signal >= MPE and CI_high(history_residual) < MPE; otherwise INCONCLUSIVE. H3b is
SUPPORTED by the first or third label and FALSIFIED by GENUINE_HISTORY.

## 4. Where the history information is (diagnostics on the frozen per-event bits; nothing refit)

### 4.1 By outcome class (logit, pooled)

| class | share | history gain A -> E (mb) | residual F -> G (mb) | gain_B (mb) | contribution to the pooled +3.47 |
|---|---|---|---|---|---|
| 0 TO | 0.130 | -0.21 | -0.38 | -0.03 | -0.03 |
| 1 FT 0/2 | 0.007 | +14.17 | +14.18 | +0.93 | +0.10 |
| 2 FT 1/2 | 0.034 | +16.73 | +16.71 | +1.69 | +0.57 |
| 3 FT 2/2 | 0.058 | +20.40 | +20.49 | +2.33 | +1.19 |
| 4 2PA miss | 0.301 | +2.83 | +2.78 | +0.49 | +0.85 |
| 5 2PA make | 0.283 | +1.36 | +1.63 | +0.09 | +0.38 |
| 6 3PA miss | 0.120 | +2.57 | +1.62 | +1.07 | +0.31 |
| 7 3PA make | 0.067 | +1.37 | +0.74 | +0.75 | +0.09 |

Over half of the pooled gain is on the three free-throw classes (7% of events). History tells the model *whether
the next segment is likely to end at the line*, far more than whether a shot goes in: the make-vs-miss gains are
1-3 mb and the turnover class gains nothing. This is the signature of the team-foul state (bonus), which the
compact state does not carry.

### 4.2 By game phase, margin and start type (logit, pooled; results/p2_hist/where_is_the_gain.txt)

| slice | share of events | history gain (mb) | residual (mb) | contribution (mb) |
|---|---|---|---|---|
| periods 1-3 | 0.749 | +3.93 | +3.76 | +2.94 |
| 4Q/OT, more than 3 min left | 0.176 | +1.58 | +1.59 | +0.28 |
| 4Q/OT, 60-180 s left | 0.044 | +2.73 | +2.73 | +0.12 |
| 4Q/OT, last 60 s | 0.031 | +4.11 | +4.60 | +0.13 |
| margin 0-5 | 0.710 | +3.88 | +3.76 | +2.76 |
| margin 6-10 | 0.139 | +2.88 | +2.78 | +0.40 |
| margin 11-15 | 0.077 | +2.43 | +2.32 | +0.19 |
| margin 16+ | 0.074 | +1.71 | +1.69 | +0.13 |
| start after_make | 0.393 | +4.30 | +4.11 | +1.69 |
| start after_dreb | 0.270 | +2.94 | +2.82 | +0.80 |
| start after_dead_ball | 0.146 | +3.95 | +4.01 | +0.58 |
| start after_oreb | 0.121 | +2.13 | +2.08 | +0.26 |
| start after_live_turnover | 0.069 | +2.16 | +2.09 | +0.15 |

Only 7% of the pooled gain comes from the last three minutes of regulation or overtime. The signal is a
whole-game, close-game, every-start-type phenomenon, not a late-game fouling regime. The exploratory time x
margin check (section 5.2) tests that reading directly.

### 4.3 Hot-hand contrast (frozen diagnostic)

4,317,194 consecutive field-goal pairs by the same offence: pooled P(make | previous make) - P(make | previous
miss) = -0.0069; within-game game-weighted mean = -0.0136. The within-game number carries the Miller-Sanjurjo
small-sample bias (negative in expectation under no hot hand), and the pooled number mixes between-game
heterogeneity; neither shows a positive hot hand, and the per-class table above shows that history does not
predict makes vs misses beyond 1-3 mb. The history signal is not a hot hand.

## 5. H3b attribution: which slow variable absorbs the residual?

Frozen rule (docs/prereg/01): add the slow variables one at a time to F and to G; the history signal is
"explained" by the first that brings history_residual's CI below the MPE (0.5 mb). Every sixth test month,
frozen logit; the final row set (31 months, 951,008 events) is in results/p2_hist/summary.json section
`attribution`; the table below is that final set.

| contrast | millibits/event | 95% CI | p |
|---|---|---|---|
| history_residual baseline (F -> G) on these months | +3.459 | [+3.260, +3.677] | 0.0005 |
| slow-variable gain: team fouls + bonus (F -> F_fouls) | +7.623 | [+7.272, +7.950] | 0.0005 |
| **history_residual after fouls (F_fouls -> G_fouls)** | **+1.034** | **[+0.923, +1.147]** | 0.0005 |
| slow-variable gain: stint length (F -> F_stint) | +0.223 | [+0.171, +0.271] | 0.0005 |
| history_residual after stint (F_stint -> G_stint) | +3.404 | [+3.194, +3.614] | 0.0005 |
| slow-variable gain: lineup change (F -> F_lineup) | +0.082 | [+0.048, +0.116] | 0.0005 |
| history_residual after lineup (F_lineup -> G_lineup) | +3.476 | [+3.278, +3.684] | 0.0005 |
| slow-variable gain: seconds since previous segment (F -> F_gap) | +8.867 | [+8.549, +9.198] | 0.0005 |
| history_residual after gap (F_gap -> G_gap) | +9.932 | [+9.598, +10.224] | 0.0005 |
| slow-variable gain: all four (F -> F_all) | +17.536 | [+17.031, +18.037] | 0.0005 |
| history_residual after all four (F_all -> G_all) | +5.711 | [+5.483, +5.944] | 0.0005 |
| previous segment's class beyond E (E -> E_PC) | +0.524 | [+0.445, +0.602] | 0.0005 |
| previous segment's class beyond G (G -> G_PC) | +0.516 | [+0.441, +0.597] | 0.0005 |
| previous segment's class beyond G + all slow (G_all -> G_all_PC) | +0.616 | [+0.541, +0.692] | 0.0005 |

(31 months, 951,008 events; bootstrap over games, B = 2000.)

Reading. Team fouls and the bonus flag absorb most of the residual (+3.5 -> +1.0 mb) and are, by themselves,
worth more than twice history (+7.6 mb on top of the compact state). Stint length and lineup change absorb
nothing. Seconds since the offence's previous segment (the gap) is itself worth +8.9 mb but does *not* absorb history:
with the gap in the model the history blocks become *more* informative (residual +9.9 mb), i.e. the gap and
the recent-points history are jointly informative (short gap plus a free-throw trip on the previous possession
identifies a fouling sequence; short gap plus a quick score identifies transition exchanges). With all four slow
variables the residual is +5.7 mb. No variable on the pre-registered list brings the residual below the MPE, so
under the frozen rule the history signal is **not explained** and H3b is FALSIFIED. The previous segment's class
(not frozen, added as the obvious one-step feature the frozen L1 block cannot express) adds a further +0.62 mb
beyond latent, slow variables and history, and +0.52 mb beyond E: it is independent information.

Timeouts remaining: not testable on this corpus (nba_stats timeout rows carry team_id 0; see data notes).

### 5.1 What the attribution says about the compact state

Two as-of-segment variables that the frozen compact state omits are each worth +8 to +9 mb, more than twice
the whole history signal: team fouls in the period with the bonus flag, and the time since the offence's
previous segment started. Both are state, not history: they are known at the segment's first event without any
sequence of earlier outcomes. That is the E-STATE (H2) question, and it is where the next millibits are.

### 5.2 Exploratory post-hoc check: a time x margin block (not pre-registered; no decision rests on it)

Question: is the residual an end-game regime that the frozen state cannot see because it carries the clock only
as period + clutch flag and the margin only in 6-point buckets? `explore_timemargin.py` adds a TM block (seconds
left in the period in six bins crossed with a seven-level signed margin in the fourth period and overtime, plus
the clock bins alone in periods 1-3; 48 cells) to the frozen sets on three attribution months (2005-03, 2012-03,
2016-03; 150,907 events).

| contrast | millibits/event | 95% CI | p | n |
|---|---|---|---|---|
| TM block on the state (A -> A_TM) | +9.740 | [+8.819, +10.693] | 0.0005 | 150,907 |
| history_residual baseline (F -> G) | +3.402 | [+2.924, +3.884] | 0.0005 | 150,907 |
| history_residual with TM (F_TM -> G_TM) | +1.593 | [+1.244, +1.941] | 0.0005 | 150,907 |
| history_residual with TM + all slow (F_all_TM -> G_all_TM) | +5.701 | [+5.150, +6.253] | 0.0005 | 150,907 |
| 4Q/OT only: history_residual baseline (F -> G) | +1.842 | [+1.287, +2.422] | 0.0005 | 37,932 |
| 4Q/OT only: history_residual with TM (F_TM -> G_TM) | +0.834 | [+0.442, +1.242] | 0.0005 | 37,932 |

Reading. The finer clock x margin block is worth +9.7 mb on the compact state by itself (another missing state
variable for E-STATE) and halves the history residual (+3.4 -> +1.6 mb), but does not remove it; in the fourth
period and overtime the residual with the block is +0.8 mb [+0.4, +1.2], at the MPE. The end-game-regime reading
is therefore partly right (the state's clock is too coarse) and mostly wrong (section 4.2: three quarters of the
gain sits in periods 1-3). No decision rests on this block.

## 6. Downstream: can the event-level gain reach a player-stat forecast?

**At T-60, no.** L2-A forecasts a player's game line sixty minutes before tip from his appearance history. Nothing
in sets B-G exists at that time: L1, L3, L10, EW and GL are functions of the game being forecast. The event-level
gain can therefore reach a player-stat distribution only through (i) a possession simulator that plays the game
forward and whose per-possession model carries the history blocks, which is not built (by instruction), or (ii) an
in-game forecasting lane, which is not in the programme. An upper bound on what (i) could add: +3.5 mb per segment
x about 105 segments per team-game = 0.37 bits per team-game of cross-entropy on the *sequence of segment
outcomes*, of which 55% is about free-throw trips. A simulator without a foul-state variable would be mis-specified
by +7.6 mb per segment on the free-throw classes before it ever used history. This bounds the value of a history
mechanism in a future simulator at a few tenths of a bit per team-game; it does not bound the value of a
simulator, which is H5.

**Between-game analogue (EXPLORATORY, labelled, not a hypothesis test).** The within-game question has a
between-game cousin, H3c (does recent form beyond the filtered skill state help?), whose reference L8 does not
exist yet. As a bounded preview on the existing baseline: L2-A's pmf for each target shifted by a shrunk
last-3-game residual (beta per season fitted on the prior season), regular season 2001-2020, 490,535 units
(results/p2_hist/downstream_form.md):

| target | CRPS skill vs L2-A | 95% CI | beta range over seasons |
|---|---|---|---|
| MIN | +1.668% | [+1.634%, +1.692%] | +0.22 .. +0.28 |
| PTS | -0.012% | [-0.015%, -0.012%] | -0.05 .. +0.02 |
| REB | -0.001% | [-0.001%, -0.001%] | -0.04 .. +0.03 |
| AST | +0.007% | [+0.007%, +0.007%] | -0.05 .. 0.00 |
| PRA | +0.115% | [+0.114%, +0.115%] | +0.01 .. +0.09 |

Minutes carry a form signal (a player trending up or down in minutes keeps trending, which L2's five-appearance
half-life already mostly tracks); the box-score targets carry none. This is consistent with the event level: what
history predicts is the *kind* of possession ending (fouls), not the player's shooting. Nothing here changes L2-A.

## 7. Confirmations

- P1 L2-A baseline UNCHANGED: results/p1_ladder/ not rewritten; `shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json`
  implementation hash unchanged; no P1 config or model file edited (preflight hashes).
- HOLDOUT UNTOUCHED: 0 rows with season >= 2023 in possessions / play_by_play / player_game_box, checked before and
  after the run; lockbox 2021-2022 present in the package, never parsed (no segments_2021/2022 file on either
  machine; `segments.py` and `slow_vars.py` refuse seasons > 2020).
- SHADOW FIREWALL INTACT: no file under the shadow executor's SHADOW_HOME was read; the executor session reported
  operational status only (runner alive, no locks); no 2023+ or current-season data, forecast, lock, outcome or score
  entered this lane.
- Registry: `python -m hoopslab.registry verify` INTACT before and after; EXPOSURE_LOG rows 13 (start) and 14
  (outcomes); POSTFREEZE_CHANGELOG C20 (scripts, maxiter), C21 (parser rebound rule; evidence in
  results/p2_hist/artifact_evidence/), C22 (this report, canary 2, exploratory scripts, C21 class label fix).
- Tests: `pytest -q -m "not slow"` green at the committing hash.

## 8. Limitations, recorded

1. Segments are parsed from play-by-play (M18), reconciling 99.4% of points and 99.98% of field-goal attempts
   with the supplied possessions; possessions whose points do not reconcile stay in the data as parsed (the gate
   is a quality bound, not a filter), and segments with no terminal event, no clock or an incomplete lineup are
   dropped (at most 1,016 per season, 0 to 10 per season from 2003 on; counts in preflight.json).
2. The frozen START encoding has no after_dead_ball level (shares the baseline with after_make).
3. The timeout slow variable is not testable (team_id 0 on nba_stats timeout rows).
4. GBM on every third month and attribution on every sixth month (compute); the logit, GRU and per-season tables
   are complete.
5. Optimiser cap 1000 instead of the class default 300 (C20); all pooled fits converged.
6. 193 colliding (game_id, possession_seq) keys (0.003%), affecting only the GRU-logit join rows count.
7. The GRU is a fixed, small, 4-epoch instrument; a tuned sequence model was not the question (H10 is), and the
   nested A_nn/F_nn references keep the within-family contrasts fair.
8. NBA nba_stats only; no league pooling; the lockbox and holdout were not touched, so nothing here is confirmatory.
