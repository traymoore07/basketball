# 00. Lane registry: research, shadow and context lanes

*Written 2026-11-02 under Tray's "create the new lanes and their governance" mission. Nothing here runs an
experiment, reads a prospective outcome, or changes a frozen artefact. The existing lanes (R0, S0) are
restated, not altered. The new lanes (R1, R2, L1, X1) are created with their locations, permitted and
forbidden data, model inheritance, exposure rules, version identifiers and hand-offs. Each lane has a
lane card in its own directory; this file is the index and the firewall map.*

## 1. Lanes

| id | name | kind | operated by | location | status |
|---|---|---|---|---|---|
| R0 | RESEARCH CLAUDE | historical research | Research Claude | this repository (`docs/`, `experiments/`, `src/hoopslab/`, `results/`, `registry/`) | unchanged; P1 done, P2 E-HIST done, P2 E-STATE running |
| S0 | FROZEN SHADOW BASELINE `L2-A-SHADOW-v0.1` | prospective executor | external executor (not Research Claude) | `SHADOW_HOME` outside the repository (docs/shadow/00); code `src/hoopslab/shadow/` at the pinned commit | unchanged; permanent control group; never modified |
| R1 | SEASON-CARRYOVER-RESEARCH | historical research | Research Claude | `research_season/` | CREATED (lane card + draft spec; no run) |
| R2 | ROTATION-OPPORTUNITY-RESEARCH | historical research | Research Claude | `research_rotation/` | CREATED (lane card + draft spec; no run) |
| L1 | PREGAME-CONTEXT-LIVE | live input lane | external executor | contract and validator in this repository (`live_inputs/`, `src/hoopslab/pregame/`); snapshots in `CONTEXT_HOME/live_inputs/` outside the repository | CREATED (contract, validator, fixture; no live source contacted) |
| X1 | CONTEXT-SHADOW | prospective executor (challengers) | external executor | executor package in this repository (`shadow_context/`, `src/hoopslab/pregame/`); forecasts, outcomes and ledger in `CONTEXT_HOME/live_results/` outside the repository | CREATED (version plan, executor skeleton, refusal guards; no challenger model exists yet) |

`SHADOW_HOME` and `CONTEXT_HOME` are environment variables of the executor's machine. Their defaults
(`C:\Users\Tray\Documents\Claude\shadow_lane`, `C:\Users\Tray\Documents\Claude\context_lane`) are outside
the repository and outside every path Research Claude is pointed at (memory rule "stay in task files").
The two may be the same directory tree; the ledgers inside are separate files keyed by model version.

## 2. Permitted and forbidden data per lane

| lane | may read | may never read | writes |
|---|---|---|---|
| R0 | corpus researcher package v1.0.2, development window 1997-2020 (burn-in 1997-1999), under the exposure ledger; lockbox 2021-2022 only in an authorised confirmation run | holdout (>= 2023) in any form; any prospective outcome, forecast error, ranking, cumulative score or calibration from the shadow or context lanes; `SHADOW_HOME`; `CONTEXT_HOME` | model packages (frozen code + card), docs, registry rows, results under `results/` |
| S0 | live pre-game information (schedule, rosters, availability, confirmed starters); completed box scores needed for the current forecast; final box scores of games after their forecasts are locked | nothing in the research lane beyond the frozen package and its documentation | locked forecasts, snapshots, outcomes, score ledger (`SHADOW_HOME`) |
| R1 | exactly R0's data (PLAYER_GAME box rows incl. did-not-play rows, GAME, ROSTER spells, PLAYER listed position, LINEUP_STINT / realised starters at post-game knowledge time), seasons 1997-2020, prequential | as R0; additionally: no 2023+ outcome, no live injury report, no shadow or context ledger; no current-season anything | `results/research_season/`, docs, a frozen artefact `SEASON-AWARE-L2-v0.2` only if the evidence supports it |
| R2 | exactly R0's data; historical absence labels only from the box score itself (status `played` / `dnp_coach`; the nba_stats did-not-play comment "DNP - Coach's Decision" / "DND - Injury/Illness" / "NWT - Personal Reasons" / body-part strings), rostered-without-box-row; realised starters; lineup stints and substitutions (level C) | as R1; no manufactured injury labels from outcomes (an absence label is the recorded fact of not dressing and its stated reason, never inferred from a bad line) | `results/research_rotation/`, docs, frozen components (minutes-distribution and opportunity-redistribution modules) only if the evidence supports them |
| L1 | any live pre-game source the executor chooses (official reports, beat writers, starters feeds, vendor APIs); timestamps of publication and retrieval | nothing from the research lane except the contract and validator | `CONTEXT_HOME/live_inputs/<cutoff>/` snapshots (one per information cutoff) |
| X1 | L1 snapshots; the frozen model packages of every version it runs (v0.1 from `src/hoopslab/shadow/`, later versions from their pinned commits); final box scores after each lock | nothing from the research lane beyond frozen packages and documentation; never a research result that was not frozen into a version | `CONTEXT_HOME/live_results/` : per-version locked forecasts, outcomes, per-version ledgers, cumulative tables |

Research Claude may see about L1 and X1: the code architecture, the frozen model definitions, the
contract, and the executor's **operational health** (runs completed, snapshots validated, counts of games
forecast, refusals, hash verification status). Research Claude may NOT see: prospective current-season
outcomes, individual live forecast errors, live model rankings, cumulative challenger scores, calibration
from current-season games. Tray controls any later exposure, by an exposure-ledger row of class
`O-shadow` naming the aggregate, the model versions and the months (docs/shadow/00 section 2).

## 3. Model inheritance and version identifiers

```
P1 ladder rung L2 = hoopslab.models.baselines.PlayerMean(half_life=5.0)
    |
    +-- L2-A-SHADOW-v0.1          (S0; frozen 2026-10-30; card shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json)
    |       permanent control group; never modified; forecasts every eligible game alongside every challenger
    |
    +-- SEASON-AWARE-L2-v0.2      (R1 artefact, if supported) -> SEASON-AWARE-SHADOW-v0.2 (X1 challenger 1)
    |       changes ONLY the weighting of historical observations: weight = recency x season-transition x
    |       context-similarity (research_season/01_carryover_spec_draft.md). No injury information, no
    |       current-season prospective outcome, same targets, same pmf construction, same T-60 cutoff.
    |
    +-- AVAILABILITY-SHADOW-v0.3  (X1 challenger 2) = v0.2 + tonight's availability / roster status from an
    |       L1 snapshot (who is eligible; which teammates are absent). Historical weighting as v0.2.
    |
    +-- CONTEXT-SHADOW-v0.4       (X1 challenger 3) = v0.3 + R2 components: role given the available roster,
            minutes distribution, opportunity (FGA / usage / assist and rebound chances) redistribution,
            per-minute production -> stat distributions. Decomposed, not one opaque network.
```

Frozen rungs and hypotheses the lanes inherit from (registry, tier 1, untouched): L2 (recency-weighted
mean; half-life fixed on the development window), L3 (availability-aware minutes x per-minute rates),
L4 (hierarchical shrinkage), L5 (teammate-availability "vacated usage" redistribution, GLM), H12 (how much
forecast error is minutes and availability). R1 and R2 do not amend tier-1 files; their questions are
lane-level pre-registrations (draft specs in their directories, frozen by a changelog row before any
evaluation). A result that would change a frozen hypothesis or rung goes through `registry/AMENDMENTS.md`.

Version rule (shared with docs/shadow/00 and 01): a model that has learned from prospective outcomes is a
new version; a version never changes after its card is written; a card carries the implementation hash
(per file and combined), the pinned commit and the lineage; the executor verifies the hash before every
run; older versions keep forecasting the same games until Tray retires them; locked forecasts are never
rewritten; comparisons use only games every compared version forecast at the same cutoff.

## 4. Exposure rules

- Lane creation (this document and its siblings) is exposure class S1 at most: column names and distinct
  values of the nba_stats did-not-play comment were read to decide what R2 may claim (EXPOSURE_LOG row 16).
- R1 and R2 are O-dev lanes: their outcome exposure is logged per experiment as today (start row, outcome
  row), with the experiments whose validation outcomes were observed.
- L1 and X1 produce no research exposure. Their only route into R0 is a Tray-authored `O-shadow` row with an
  aggregate. Operational-health lines (counts, refusals, hash checks, run times) carry no outcome and may be
  relayed.
- A prospective result never retunes a frozen version. The research question "did season awareness help
  prospectively?" is answered in `CONTEXT_HOME/live_results/` by the executor's cumulative tables and reaches
  research only as a logged aggregate.

## 5. Hand-offs

| from | to | artefact | gate |
|---|---|---|---|
| R1 | X1 | `SEASON-AWARE-L2-v0.2` package: code under `src/hoopslab/` at a pinned commit, model card `shadow_context/cards/MODEL_CARD_SEASON-AWARE-SHADOW-v0.2.json` with implementation hash, lineage and the frozen weighting grid point | historical evidence per the frozen R1 decision rule; card hash printed by the executor equals the card; `tests/test_lanes.py` version-registry test passes |
| R2 | X1 | minutes-distribution and opportunity-redistribution components (role taxonomy, P(minutes) quantiles, redistribution rules) packaged as the v0.4 model; the v0.3 model needs only the L1 contract and v0.2 | same as above for v0.3 and v0.4 |
| L1 | S0 | the frozen ELIGIBLE table (4 states) derived from a pregame-context snapshot by `hoopslab.pregame.contract.to_eligible_v01` (deterministic mapping, UNKNOWN never ACTIVE) | validator passes; the mapping is one-way and recorded in the snapshot manifest |
| L1 | X1 | the full pregame-context snapshot (availability, roster, lineup, recent role, team absences) | validator passes; `status_time`, `published_at`, `retrieved_at` all <= cutoff |
| X1 | Tray | cumulative tables per version, target, population, month (`python -m hoopslab.pregame.cli report`) | Tray decides what reaches research |

## 6. Directory map

```
basketball/                      (git; Research Claude's lane R0)
  docs/lanes/00_lane_registry.md this file
  research_season/               R1 lane card + draft spec (results under results/research_season/)
  research_rotation/             R2 lane card + draft spec (results under results/research_rotation/)
  live_inputs/                   L1 lane card + contract + example snapshot (fixture only; no live data)
  shadow_context/                X1 lane card, VERSIONS.json (version plan), cards/ (immutable model cards)
  src/hoopslab/pregame/          L1 contract + validator, X1 version registry + executor CLI skeleton
  shadow/, src/hoopslab/shadow/  S0 (unchanged)
CONTEXT_HOME/  (outside git)     live_inputs/<cutoff>/  live_results/<version>/  ledger/
SHADOW_HOME/   (outside git)     S0's snapshots, forecasts, outcomes, ledger (unchanged)
```

## 7. What was not done

No season-carryover experiment, no rotation-opportunity experiment, no challenger model, no contact with a
live source, no prospective forecast, no reading of `SHADOW_HOME` or `CONTEXT_HOME`, no change to
`L2-A-SHADOW-v0.1`, to the P1 results, to the E-HIST or E-STATE artefacts, or to any tier-1 file.
