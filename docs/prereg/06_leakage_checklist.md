# 06. Leakage checklist: how basketball prediction systems accidentally cheat

Each item has an ID, the mechanism, an example, and the defence. Automated detectors are implemented in `src/hoopslab/backtest/leakage.py` and exercised in `tests/test_asof_and_leakage.py`, where deliberately leaky reference models must be caught.

**Automated tests**

| Test | What it does | Catches |
|---|---|---|
| **T1** truncation invariance | Re-run a forecast after *physically deleting* every record with knowledge_time > t; outputs must be bit-identical | any read of the future through the data path |
| **T2** future perturbation | Shuffle outcome columns of all not-yet-known records (including the target game's own box score); outputs must be unchanged | same as T1, cheaper; target-game inclusion |
| **T3** ID relabel invariance | Consistently relabel all opaque IDs; outputs must be identical up to relabelling | ID-content features, ID-order leakage |
| **T4** knowledge-time audit | Outcomes known before they could exist; knowledge_time == ingested_time; injury reports published after tip; unknown-quality share | adapter / provenance errors |
| **T5** static access scan | Model and feature source must not reference raw-store internals, `full_table`, `ingested_time`, or synthetic truth | side channels (which T1/T2 cannot see) |
| **T6** population check | The evaluation population must contain non-appearing rostered players | selection on outcome |
| **T7** feature recompute | Every as-of feature function gives identical output on a truncated store | "statistics recomputed using later games" |
| Runtime guard | `AsOfView` exposes no path to the raw store | accidental direct reads |
| Runtime guard | `LockboxError`: the runner refuses to forecast inside a sealed window without the frozen manifest hash | peeking at the confirmation window |

**Important limitation, found while building the tests.** T1/T2 are only as strong as the pipeline they wrap. A model holding a reference to the full dataset *outside* the factory evades them. `test_side_channel_evades_dynamic_tests_by_design` demonstrates this. Hence **L-X1** below and the static scan T5. The factory signature is `factory(store) -> model`, and everything a model uses must be built from that store.

## A. Outcome and statistic leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-A1 | Season averages that include the target game | "points per game this season" computed from the final season table | features only from AsOfView (T1, T2, T7) |
| L-A2 | End-of-season ratings used for earlier forecasts | full-season RAPM/EPM/BPM, end-of-season Elo | ratings recomputed as-of (T1) |
| L-A3 | Statistics recomputed using later games | league-average normalisation, pace adjustment, or opponent-adjustment fitted on the whole season | normalisers are as-of features (T7) |
| L-A4 | Scalers, encoders, and clusterings fitted on all data | StandardScaler, player archetype clusters, embeddings trained on the full corpus | fit inside `fit(view)` only; cached encoders must be prefix-consistent (as in `PossessionSimulator._encode`) |
| L-A5 | Stat corrections applied retroactively | later-corrected assist/rebound totals used for earlier forecasts | versioned records; latest version with knowledge_time ≤ t |
| L-A6 | Target-adjacent outcome fields | `plus_minus`, `started`, `seconds` of the target game as features | `time_role: outcome` in the contract; T2 perturbs outcome columns |
| L-A7 | Hyperparameters / feature selection tuned on evaluation data | choosing the EW half-life that is best on the test seasons | tuning only on the development window; lockbox guard |

## B. Availability and lineup leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-B1 | Actual starting lineups in pre-game forecasts | `PLAYER_GAME.started` used at T-60 | `started` is an outcome; STARTING_LINEUP has its own knowledge time (≈ T-30); mode L forecasts are issued after lock |
| L-B2 | Actual minutes / DNP-CD known only after the game | "active players" = players with seconds > 0 | request population = as-of roster; T6 |
| L-B3 | Injury status updated after forecast time | final status used at T-60 | INJURY_REPORT snapshots with publication times; T1 catches `LeakyFinalInjuryStatus` |
| L-B4 | Inactive lists announced near tip used earlier | – | knowledge time = announcement time |
| L-B5 | Evaluation restricted to players who played | scoring only appearances in mode U | mode U scores all rostered players (T6) |
| L-B6 | Status→availability calibration using final statuses | learning P(play \| questionable) from post-hoc statuses | calibrate on statuses visible at the same forecast offset (implemented in L3/L4 and the simulator) |
| L-B7 | Minute restrictions revealed post-game | "played 18 minutes on a restriction" | only if published before t, with a knowledge time |

## C. Knowledge-time and calendar leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-C1 | `ingested_time` used as knowledge time | backfilled data treated as "always known", or scraped reports treated as known at scrape time | separate fields; validator warning; T4 |
| L-C2 | Transactions dated by effective date instead of announcement | a trade effective on day d but announced at d-1 20:00 (or the reverse) | ROSTER knowledge_time = announcement; fallback 23:59 local |
| L-C3 | Time-zone errors | date-only reports compared with UTC tip times; games after midnight UTC | UTC timestamps; fallback rules for date-only fields |
| L-C4 | Schedule derived from the final schedule | back-to-back or rest flags computed from a schedule that was later changed (postponements) | GAME is versioned; schedule context derived as-of |
| L-C5 | Standings / seeding / tanking indicators computed from final standings | "team eliminated" flag from end-of-season standings | as-of standings only |
| L-C6 | Cross-league calendar misalignment | a pooled model trained on a WNBA season (May–Oct) that ends after the NBA forecast date; NCAA tournaments overlapping the NBA season | as-of is calendar time for every league (`10_cross_league.md`); T1 on pooled pipelines |
| L-C7 | Unknown knowledge times treated optimistically | – | conservative fallback + optimistic sensitivity run → LEAKAGE-SENSITIVE flag |

## D. Derived-data leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-D1 | Lineups reconstructed from future events (live mode) | period-start lineups inferred from who appears *later* in the period | live mode must reconstruct lineups causally; pre-game use is fine because the whole game is past |
| L-D2 | Possession parsing / garbage-time labels using the final score | "garbage time" defined from the final margin | labels computed from the state at the time |
| L-D3 | Shot zones or play types re-labelled with later definitions | – | acceptable for past games (definitional); record the label version |
| L-D4 | Aggregated matchup data from a provider that recomputes history | – | provider delivery time as knowledge time |

## E. Evaluation leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-E1 | Evaluation population selected on the outcome | only players who played; only games with complete PBP when completeness correlates with outcome (e.g. overtime games parsed differently) | population = as-of roster × scheduled games; completeness filters decided as-of |
| L-E2 | Excluding hard cases | dropping overtime games, blowouts, or games with injuries mid-game | nothing excluded unless pre-registered |
| L-E3 | Strata defined with hindsight | "star out" defined by actual non-appearance; "trade" by later effective date; "role change" detected with future data | strata from AsOfView only (`backtest/strata.py`); the CUSUM uses past games only |
| L-E4 | Multiple looks at the lockbox | iterating after seeing confirmation results | lockbox guard + manifest hash + amendment log |
| L-E5 | Metric shopping | switching primary metric after results | primary metric fixed per target (`08`) |

## F. Identifier and linkage leakage

| ID | Mechanism | Example | Defence |
|---|---|---|---|
| L-F1 | IDs encode order or future state | player IDs assigned by first NBA appearance; game IDs encoding playoff round | opaque IDs; T3 |
| L-F2 | End-of-season team attribution | a player-season row listing the player's final team | ROSTER spells with knowledge times |
| L-F3 | Cross-league links that exist only because of the future | only NCAA players who later reached the NBA are linked, so "has an NBA link" reveals the future | PERSON_LINK knowledge_time = when the link could have been made; NCAA-era features never use the existence of a later link |
| L-F4 | Survivorship in player tables | player tables that contain only players who later had careers | PLAYER rows with knowledge times; as-of rosters |
| L-F5 | Bio / position fields revised later | later height, retroactive position relabels | PLAYER versions |

## G. Market leakage (only if H14 runs)

| ID | Mechanism | Defence |
|---|---|---|
| L-G1 | Closing lines used for earlier forecasts | snapshot selected with knowledge_time ≤ t; no `is_closing` column stored |
| L-G2 | Props offered only for players expected to play (selection) | evaluate on the as-of roster population; props as benchmark only |
| L-G3 | Snapshot timestamps in local time or at delivery time | UTC, provenance documented |

## X. Engineering side channels

| ID | Mechanism | Defence |
|---|---|---|
| L-X1 | Side channels: precomputed feature files, global caches, or notebooks built from the full dataset | factory(store) rule; T5 scan; any artifact must be a function of the as-of view |
| L-X2 | Caches keyed by entity rather than by time | caches valid only within one view (`AsOfView.cache`) or prefix-verified (`PossessionSimulator._encode`) |
| L-X3 | Random seeds derived from outcomes or IDs | seeds from configuration only |
| L-X4 | Synthetic truth reaching models | T5 scans for `truth[` |

**Before any result is reported,** T1, T2, T3, T5, T6, and T7 must pass for every model in the comparison, on at least 20 randomly sampled forecast times per season, and T4 must pass for the dataset.
