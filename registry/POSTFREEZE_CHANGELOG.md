# Post-freeze changelog (tier-2 files)

Tier-2 files are all files committed at freeze commit `843caaa` other than the 25 hash-frozen tier-1 files. Their pre-data hashes are in `registry/FREEZE_SUPPLEMENT.json`. A change to any of them must be listed here. `tests/test_registry.py` enforces this.

**Classes:**

- `engineering`: performance, refactoring, and new adapters, with no change to any pre-registered decision. **Not allowed** for files marked `decision_bearing`.
- `governance`: pre-registration bookkeeping (registry tooling, logs, indices).
- `bugfix`: the code did not do what the frozen specification says. Describe the discrepancy.
- `implementation-per-spec`: new code implementing a frozen specification (e.g. a ladder rung marked "interface only"). Cite the spec.
- `amendment:A<n>`: part of a logged amendment.

The exposure column is the real-data exposure class at the time of the change (`EXPOSURE_LOG.md`).

| ID | Date (UTC) | File | Class | Description | Exposure at time |
|---|---|---|---|---|---|
| C1 | 2026-09-27 | src/hoopslab/registry/__init__.py | governance | three-tier verification (tier 1 vs latest FREEZE entry; FREEZE entry 0 and archive v1 immutable; tier-2 changes must be logged); `begin-amendment`; `freeze --amend` requires a complete AMENDMENTS row | S0 |
| C2 | 2026-09-27 | tests/test_registry.py | governance | tests for the checks in C1 | S0 |
| C3 | 2026-09-27 | docs/prereg/00_index.md | governance | links to the onboarding protocol and the new registry files | S0 |
| C4 | 2026-09-27 | README.md | governance | status line and onboarding links | S0 |
| C5 | 2026-10-30 | src/hoopslab/adapters/corpus_v102.py | engineering | NEW read-only adapter for BASKETBALL_CORPUS_v1.0.2 (isolated researcher package): one league x source family x season range per store; knowledge times by the contract's conservative fallback rules; ids namespaced by league/family; no pre-registered decision touched | S1/S2 |
| C6 | 2026-10-30 | tests/test_corpus_adapter.py | governance | NEW tests: strict level-A build, T4, T6, outcome invisibility before tip, holdout split enforcement, determinism, namespacing, level-C window; skipped when the release is absent | S1/S2 |
| C7 | 2026-10-30 | experiments/onboarding/*.py | engineering | NEW onboarding scripts (schema dump, null shares, key uniqueness, S2 consistency checks, power inventory, adapter smoke test); they write only under results/onboarding/ | S1/S2 |
| C8 | 2026-10-29 | docs/onboarding/01_corpus_compatibility_audit.md, registry/ADAPTER_ATTESTATION.json, registry/DATA_CONTRACT_MISMATCHES.md (M1-M23), docs/prereg/00_index.md, README.md | governance | onboarding audit written (executability of every frozen experiment: READY 10 / PARTIAL 6 / BLOCKED 4 of 20), adapter attested by sha256, links added; no frozen file changed; STOP before any O-dev outcome | S1/S2 |
| C9 | 2026-10-30 | src/hoopslab/adapters/corpus_v102.py | bugfix | ROSTER gap-close rule never fired: team game tips were converted to integers in the series' own unit (seconds/microseconds under pandas 3) and compared with nanosecond Timestamp.value, so searchsorted always returned the end of the schedule. Effect before the fix: every spell stayed open until a transfer, request populations held ~85 players per team (every player who ever appeared for the team). Fix: convert via dt.as_unit('ns'). After: ~16 per team. Found by the P1 timing probe (request counts, S1) BEFORE any outcome was computed; adapter tests 8/8 pass; attestation refreshed | S1 |
| C10 | 2026-10-30 | .gitignore | engineering | ignore the large P1 run artifacts (per-season forecasts, score parts, requests, strata, logs) under results/p1_ladder/; summaries are committed | S1 |
| C11 | 2026-10-30 | experiments/p1_ladder/run_season.py | bugfix | P1 driver enumerated forecast points in fixed 7-day chunks, reading the schedule as known at chunk start - 1 day; GAME v1 rows are published exactly 7 days before tip, so every game on the 7th day of each chunk was not yet visible and was silently never forecast (about 15% of games per season: 2005 202/1314, 2016 111/1316, 2001 152/1260). Found after run 1 (20 seasons) by the DI-1 investigation, before any final result. Fix: chunk length derived from the measured publication lag (chunk_days = lag_days - 1 = 6), single-lag assertion, end-of-season coverage assertion (every P1-phase game in the window minus DI-1 rows must have requests; fails loudly), --dry-run flag; dry runs 2001 and 2005 cover 1260/1260 and 1314/1314. Run 1 archived under results/p1_ladder_run1_chunkbug/ (README) and NOT used; all 23 seasons re-run. No model, metric, split, stratum, grid or decision rule touched | O-dev |
| C12 | 2026-10-30 | .gitignore | engineering | ignore the archived run-1 artifacts (results/p1_ladder_run1_chunkbug/*, 13 GB) except its README | O-dev |
| C13 | 2026-10-30 | experiments/p1_ladder/exclusions.py, leakage_tests.py, summarize.py, report_tables.py, verify_di1.py | engineering | NEW P1 scripts: DI-1 structural exclusion (M24); leakage detectors T1/T2/T3/T6/T7 on 20 random tips per season for every rung through one composite forecaster + T5 static scan; summary (cluster sums, L2 selection on strictly prior seasons, credit chain with frozen bootstrap/Holm/MPE/no-harm, PIT calibration, frozen strata, gates); markdown tables; DI-1 verification across all seasons. They write only under results/p1_ladder/ | O-dev |
