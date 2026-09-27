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
