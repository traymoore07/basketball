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
