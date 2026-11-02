# E-STATE specification (H2): does the history signal vanish when the state is represented correctly?

Written 2026-10-31 before any E-STATE model was fitted. Everything below is fixed now; deviations go to
`registry/POSTFREEZE_CHANGELOG.md`. The E-HIST result (`docs/p2/01_state_history_report.md`) stands unchanged
and is the only outcome knowledge used to order the blocks.

## 1. Question

With the frozen compact state S0 (E-HIST set A), within-game history is worth +3.466 mb/event [+3.386, +3.547]
and +3.354 mb beyond the game latent; adding team fouls alone brought the beyond-latent residual to
+1.034 mb [+0.923, +1.147]. E-STATE asks whether a compact OBSERVABLE state S* exists such that
P(next segment | full within-game history) is not practically better than P(next segment | S*), i.e. whether
the residual history gain falls below the frozen minimum practical effect (MPE = 0.5 mb/event) once the
state carries the variables E-HIST's attribution identified as missing.

## 2. Unit, target, data, protocol (unchanged from E-HIST)

Possession segments of NBA nba_stats games (`experiments/p2_hist/segments.py`, parser C21), 8-class terminal
outcome, development seasons 2000-2020 (burn-in 1997-1999 as training history only), prequential by month with
a 730-day training window, held-out bits per event, game-cluster bootstrap (B = 2000, seed 12345), MPE 0.5 mb.
Primary instrument: the frozen L2 multinomial logit (L2 = 1e-4, L-BFGS, 1000 iterations). Replication:
LightGBM with the E-HIST fixed parameters on the SAME months and IDENTICAL test events (this time every
logit/GBM contrast is paired per event). No neural model is trained.

Months: the 62 E-HIST GBM months (every third test month of 2000-2020, chronological index 0, 3, 6, ...; the 31
attribution months are a subset) if the timing month (1999-12, burn-in) projects the full run at <= 14 h wall
on the two machines; otherwise the 31 attribution months. The choice is a compute decision made before any
E-STATE test-month outcome is seen and is recorded in `results/p2_state/launch_log.txt`.

## 3. Candidate state blocks (all as-of the segment's first event; nothing from the segment or later)

Ordered by E-HIST attribution evidence (fouls, then gap, then the exploratory clock x margin finding, then the
blocks E-HIST could not test), with the contested previous-class block last. Columns in brackets.

| block | definition | cols | source |
|---|---|---|---|
| FB team fouls / bonus | defence and offence team fouls in the current period before the segment (E-HIST `slow_vars.py` subtypes), each one-hot 0..6+; defence in the bonus (>= 5 fouls, >= 4 in OT); NEW: defence "last-two-minutes bonus" flag (<= 120 s left in the period, >= 1 defence team foul inside those two minutes, not already in the bonus) | 16 | play_by_play foul events |
| GAP | log1p(seconds since the offence's previous segment started, E-HIST definition); NEW: first-segment flag | 2 | segments |
| TM clock x margin (late) | the exploratory E-HIST block unchanged: periods >= 4: 6 clock bins (10/30/60/120/180 s left in the period) x 7 signed-margin levels (-8.5/-3.5/-0.5/0.5/3.5/8.5); periods 1-3: the 6 clock bins alone | 48 | segments |
| SC finer score | signed start margin clipped to +-15, one-hot (31); periods 1-3 clock bin x margin level cells (6 x 7) | 73 | segments |
| PF player fouls | personal fouls charged before the segment start, whole game, to each on-court player (every foul event with a player id whose subtype is not a technical variant and not "Defense 3 Second"); per lineup (defence, offence): max fouls on the floor one-hot 0..6, players in foul trouble (fouls >= period + 1 in periods 1-4; >= 5 in OT) capped at 2 one-hot, players with >= 5 fouls capped at 2 one-hot | 26 | play_by_play foul events |
| ROT rotation | log1p offence stint length and offence lineup-change flag (E-HIST); NEW: the same two for the defence lineup | 4 | segments |
| PC previous class | the offence's previous segment class in this game one-hot (8) + none (1) (E-HIST attribution block) | 9 | segments |

Cross-check before any fit: nba_stats foul descriptions carry the player's running personal-foul count
("(P3.T1)"); the share of foul events whose tag equals the count computed here must be >= 0.98 per season,
else the counting rule is investigated and fixed before fitting (recorded in `results/p2_state/state_build_log.txt`).

Within-game history: H = L1 + L3 + L10 + EW (frozen), latent GL (frozen). "Full history" = GL + H.

## 4. Nested states and fits per month

S0 = S (E-HIST A); S1 = S0 + FB; S2 = S1 + GAP; S3 = S2 + TM; S4 = S3 + SC; S5 = S4 + PF; S6 = S5 + ROT;
S7 = S6 + PC. S6 is the richest purely cross-sectional observable state; S7 adds the first-order Markov element.

Logit fits per month (23): S_k and S_k + GL + H for k = 0..7; S0 + PC and S0 + PC + GL + H; S0 + GL, S6 + GL,
S7 + GL (bridge to the E-HIST "beyond the latent" residual); controls: S6 with the six state blocks permuted
across events within the E-HIST state cells (train and test) and S6 + (GL + H permuted within cells).
GBM fits per month (24): S_k and S_k + GL + H for k = 0..7; S0 + PC (+ GL + H); leave-one-block-out from S6
(six fits). Per-event bits are stored per segment (`results/p2_state/bits/{mnl,gbm}_{month}.parquet`).

## 5. Contrasts (millibits per event, game-cluster bootstrap)

* add-one gain g_k = bits(S_{k-1}) - bits(S_k), k = 1..7, with model size, coverage and era availability;
* residual history r_k = bits(S_k) - bits(S_k + GL + H), k = 0..7. PRIMARY: r_6. Secondary: r_7.
  Bridge: bits(S_k + GL) - bits(S_k + GL + H) for k in {0, 6, 7} (E-HIST's history_residual definition);
* short memory: gain of PC on S6 (= g_7) and of H on S7 (= r_7);
* proxy test: gain of PC on S0 vs gain of PC on S6; and bits(S0 + PC) - bits(S4) (previous class versus
  fouls + gap + clock/score state);
* information decomposition: bits(S0) - bits(S7 + GL + H) = sum_k g_k + r_7, each component with CI;
* controls: bits(S0) - bits(S6_perm) and bits(S6) - bits(S6 + GLH_perm), both expected <= 0;
* per season: g_k, r_6, r_7; by period / margin / class for r_6 (diagnostic);
* GBM: the same contrasts; logit vs GBM per set on identical events;
* frozen H2 (registry): loss = bits(S0) - bits(S7) against max(0.5 mb, 10% of I(S7)), where I(S7) is S7's
  usable information over the training-class-frequency model; SUPPORTED / FALSIFIED / INCONCLUSIVE by the
  registry's `decision_override`. (E-HIST already implies this loss exceeds 17 mb, so the frozen H2 with
  S* = S0 is expected to be FALSIFIED; its "supports further research" clause is what E-STATE executes.)

## 6. Decision rules (fixed now)

Regime by the PRIMARY residual r_6, logit and GBM on identical events:

* REGIME A (state sufficiency): CI_high(r_6) < 0.5 mb in both families;
* REGIME C (material long memory): CI_low(r_6) >= 1.0 mb (two MPE) in both families;
* REGIME B (small residual memory): otherwise, including family disagreement (stated as such).

r_7 is reported with the same labels as a secondary reading. The previous class is called a proxy for omitted
state if its gain on S6 has CI_high < 0.5 mb; if its CI_low >= 0.5 mb the residual short-memory information is
documented without a momentum interpretation. "Short memory useful" = YES if g_7 or r_7 has CI_low >= 0.5 mb.
"Long sequence modelling justified" = YES only under REGIME C. Blocks whose add-one gain has CI_high < 0.5 mb
are reported as not useful and left out of the parsimonious state (both states reported; the regime uses S6).
Nothing here changes P1 L2-A; no T-60 player-stat claim is made from an event-level result.

Transparency: the fouls-only beyond-latent residual (+1.03 mb) was known when the 0.5 / 1.0 thresholds were
written; both are multiples of the frozen MPE, not fitted to that number.

## 7. Portability (secondary, after the NBA result)

WNBA wnba_stats is the only other store with the same event format, supplied possessions and lineups
(lineups cover 2016-2020 only). If the segment parser (period length 600 s) passes the E-HIST gate (points and
field-goal attempts reconcile >= 0.99) on 2016-2020, the chain S0..S7 (+ GL + H, logit only) runs on the
prequential months of 2018-2020 and r_6, g_k are reported beside the NBA values. NCAA stores use a different
event format and lack player ids on ~15% of fouls: not attempted, stated as such. No pooling across leagues.

## 8. Not done here

No possession simulator, no entropy-exploitation analysis, no H6, no neural sequence model, no change to the
E-HIST artefacts, no reading of shadow-executor outcomes, no holdout or lockbox access (parser refuses > 2020).
