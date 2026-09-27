# Real-data exposure ledger

This ledger separates **what I believed before seeing real data** (frozen: `registry/FREEZE.json`, `registry/archive/prereg-v1/`) from **what the real data later taught me**. Every contact with the real corpus is appended here, *at the time it happens*, in order.

**Exposure classes** (defined in `docs/onboarding/00_onboarding_protocol.md` §2):

- `S0`: none.
- `S1` **structural**: file listings, schemas, column names, dtypes, row counts, null shares, key uniqueness, date ranges, provenance, knowledge-time availability.
- `S2` **data-quality**: internal consistency checks that do not estimate any quantity a hypothesis is about. Examples: PBP points = box points; team seconds sum; lineup has 5 players.
- `O-dev` **outcome, development window**: any statistic or model result on development-window data that bears on a hypothesis, a ladder rung, or an expectation.
- `O-lock` **outcome, lockbox**: confirmation runs only. The runner enforces the manifest hash.

For each `O-*` row, list the experiments whose validation outcomes were observed. Amendments cite this ledger to determine their report label (PRE-OUTCOME vs POST-HOC).

| # | Timestamp (UTC) | Class | What was accessed / computed | Leagues / seasons | Experiments whose outcomes were observed | Commit |
|---|---|---|---|---|---|---|
| 0 | 2026-09-27 | S0 | Pre-registration integrity verification only: git objects, hashes, test suite on synthetic data. **No real corpus accessed.** Candidate corpus locations were not opened. | – | none | (this commit) |
