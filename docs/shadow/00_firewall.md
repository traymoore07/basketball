# 00. Research / shadow firewall

*Governs every prospective (future-game) activity. Written 2026-10-30 before any live game was forecast.*

## 1. Two lanes

| | RESEARCH LANE (this repository, Research Claude) | PROSPECTIVE SHADOW LANE (external executor) |
|---|---|---|
| May read | corpus researcher package v1.0.2: development window 2000-2020 (and burn-in 1998-1999 for selection), under the exposure ledger; the lockbox 2021-2022 only in an authorised confirmation run | live pre-game information (schedule, rosters, availability, confirmed starters); completed historical box scores needed to build the current forecast; final box scores of games after their forecasts are locked |
| May never read | sealed holdout outcomes (>= 2023); any prospective game outcome; the shadow ledger; the shadow lane's data snapshots | nothing in the research lane beyond the frozen model package and its documentation |
| Writes | model packages (frozen code + model card), docs, registry rows | locked forecasts, snapshots, outcome attachments, score ledger, cumulative metrics |
| Location | `C:\Users\Tray\Documents\Claude\basketball` (git) | `SHADOW_HOME` (default `C:\Users\Tray\Documents\Claude\shadow_lane`), outside the repository, never a git submodule of it; or any machine the executor chooses |

## 2. Direction of flow

```
research lane  --(frozen package: code hash + model card)-->  shadow lane
shadow lane    --(nothing automatic)-->                       research lane
```

- The shadow lane imports `hoopslab.shadow` from a pinned commit; it never writes into the repository.
- Outcomes, scores and cumulative metrics stay in `SHADOW_HOME`. No script in this repository reads
  `SHADOW_HOME`; the only code path that scores forecasts (`hoopslab.shadow.scoring`) takes a box
  score the executor passes in and writes to a ledger the executor owns.
- Research Claude receives shadow results only as a deliberate, logged act by Tray (an exposure-ledger
  row of class O-shadow, naming the aggregate, the model versions and the months). Until such a row
  exists, Research Claude treats every prospective result as unseen. Per-game outcomes are never
  pasted into a research session; aggregates only.
- Nothing learned in the shadow lane may change a frozen model version. A model that has learned from
  prospective outcomes is a new version and enters shadow testing as such.

## 3. What Research Claude has and has not done (as of this document)

- Has not read any 2023+ box score, schedule result or roster. The replay harness
  (`hoopslab.shadow.replay.ResearchCorpusProvider`) refuses cutoffs outside 1999-10-01..2020-10-31 and
  is built on the researcher package, which carries no in-game rows >= 2023.
- Has not connected to any live data source. The input contract is provider-agnostic (tables), and the
  repository has no client for any website or API.
- Has not run the shadow model on any game after 2020-10-31.

## 4. Executor obligations that keep the firewall

1. Run the executor on a copy of the repository at the pinned commit; do not commit from it.
2. Keep `SHADOW_HOME` out of every path Research Claude is pointed at (memory rule "stay in task files").
3. Never send Research Claude a snapshot, a box score, a forecast package with outcomes, or the ledger.
4. When a report is wanted, run `python -m hoopslab.shadow.cli report` (cumulative table only, no game ids)
   and hand it to Tray, who decides whether and what to disclose, logging the exposure row.
5. Model upgrades: pull a new pinned commit with a new model version; keep the old version forecasting
   the same games until Tray retires it; never delete or rewrite a locked forecast or a ledger row.

## 5. Why mode A

The prospective candidate forecasts P(stat | the player appears). The appearance decision is an
information-state problem that the P1 diagnostics showed to be population-sensitive (docs/p1/02); the
shadow lane records the availability state at the cutoff and evaluates by state instead of inferring
an appearance probability. A model that predicts appearance would be a new version with its own
evaluation populations.
