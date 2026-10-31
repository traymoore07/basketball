# Real-data onboarding protocol (the frozen pre-registration stays frozen)

*This document is NOT part of the frozen pre-registration. It governs how the frozen plan meets the real corpus. Nothing here changes a hypothesis, expected outcome, threshold, primary metric, or decision rule.*

## 1. Pre-registration integrity verification (2026-09-27, before any corpus access)

**Verdict: INTACT.** Every check below passed. Real-data exposure at the time of verification: **S0 (none).**

| # | Check | Method | Result |
|---|---|---|---|
| V1 | Git state | working tree clean; local branch identical to `origin/claude/tender-shannon-5kd1nc`; no commits after the freeze commit | pass |
| V2 | Freeze commit | `843caaabe8a90e6ce552176a9386706f139d7d5e` ("Pre-registered experimental blueprint and laboratory (frozen before real data)") | pass |
| V3 | Freeze manifest integrity | `registry/FREEZE.json`: exactly 1 entry, `amendment: null`, `real_data_seen: false`, 25 files; the manifest hash `ccb065025fb3…9c40` reproduces from the stored per-file hashes | pass |
| V4 | Per-file hashes, three independent sources | for all 25 files: `sha256sum` of the working copy = hash stored in FREEZE entry 0 = SHA-256 of the git blob at `843caaa` | 25/25 pass |
| V5 | No post-freeze edits | `git diff --name-only 843caaa HEAD` empty (before this onboarding commit) | pass |
| V6 | Amendment log | `registry/AMENDMENTS.md`: no entries | pass |
| V7 | Registry validity | `python -m hoopslab.registry validate` → OK; `verify` → FROZEN FILES UNCHANGED | pass |
| V8 | Instruments still work | `pytest -q`: 33 passed (scoring properness, decision rule, as-of engine, leakage detectors T1–T7, lockbox guard, registry, quick synthetic checks) | pass |
| V9 | Synthetic record | `results/synthetic_suite/seed0.json` and `.log` identical to the freeze commit (all 11 checks PASS) | pass |

### Where each pre-registered component lives

| Component | Hash-frozen (tier 1, `FREEZE.json`) | Also relevant (tier 2, `FREEZE_SUPPLEMENT.json`) |
|---|---|---|
| Hypothesis registry | `registry/hypotheses.yaml`, `docs/prereg/02` | – |
| Baseline ladder | `registry/baseline_ladder.yaml`, `docs/prereg/03` | reference implementations `models/baselines.py`, `models/structural.py`, `models/event.py` |
| Expected-results registry | `registry/expectations.yaml`, `docs/prereg/04` | – |
| Target definitions | `src/hoopslab/targets.py`, `docs/prereg/08` §1 | `forecast.py` |
| Metric definitions and decision rule | `metrics/scoring.py`, `metrics/compare.py`, `metrics/information.py`, `docs/prereg/08` | `diagnostics/joint.py` (H8 instruments) |
| Temporal backtest specification | `docs/prereg/07`, `backtest/strata.py` | `backtest/runner.py` (T-60 offset, lockbox guard, request population), `asof.py`, `features.py` |
| Leakage tests | `docs/prereg/06`, `backtest/leakage.py` | `tests/test_asof_and_leakage.py` (the leaky reference models) |
| Synthetic test suite | `docs/prereg/09`, `synthetic/scenarios.py`, `diagnostics/realism.py`, `diagnostics/changepoint.py`, `experiments/history.py` | `experiments/synthetic_suite.py` (pass criteria), `synthetic/world.py`, `sim/engine.py`, `results/synthetic_suite/*` |
| Data contract | `contract/data_contract.yaml`, `docs/prereg/05`, `docs/prereg/11` | `contract/schema.py` (level definitions) |
| Freeze manifest / hashes | `registry/FREEZE.json` (append-only) | `registry/FREEZE_SUPPLEMENT.json` |
| Amendment log | `registry/AMENDMENTS.md` (append-only) | – |

### A gap in the original freeze, found and closed during this verification

The original manifest hashed 25 files. Several committed files that also encode pre-registered decisions were **not** in it: the T-60 offset and lockbox guard (`runner.py`), the E-HIST model family and its fixed L2 penalty (`models/event.py`), the synthetic pass criteria (`synthetic_suite.py`), the H8 instruments (`joint.py`), the data-level definitions (`schema.py`), and the leakage test cases. Their pre-data content was already provable from git, but a silent edit would not have been flagged.

Fix, with no frozen file touched and no content changed:

- `registry/FREEZE_SUPPLEMENT.json` hashes all 43 other files committed at `843caaa`. The hashes are computed from the freeze commit's **git objects**, so they certify pre-data content. 15 of these files are marked `decision_bearing`.
- `registry/archive/prereg-v1/` holds a verbatim copy of the 25 tier-1 files, also materialised from the git objects. It is verified against FREEZE entry 0 on every test run. This preserves the original version regardless of any future amendment.
- `tests/test_registry.py` now fails if:
  - (a) a tier-1 file differs from the latest FREEZE entry;
  - (b) FREEZE entry 0 or the v1 archive changes;
  - (c) a tier-2 file changes without a row in `POSTFREEZE_CHANGELOG.md`;
  - (d) a decision-bearing tier-2 file changes with a class other than `bugfix`, `implementation-per-spec`, or `amendment:A<n>`.

## 2. Exposure classes (what "seeing the data" means)

The ledger is `registry/EXPOSURE_LOG.md`. Every contact is logged when it happens.

| Class | Allowed during onboarding? | Examples |
|---|---|---|
| **S1 structural** | yes | listing files; schemas; dtypes; row counts per league-season; null shares per field; key uniqueness; date ranges; knowledge-time provenance per source; which contract tables and fields exist |
| **S2 data-quality** | yes | checks in `docs/prereg/11` §7: PBP points = box points; team seconds sum to regulation + OT; 5 players on court; possession counts vs box estimate; ID stability. These are pass/fail facts about the data, not estimates of anything a hypothesis is about |
| **O-dev outcome (development)** | **no, not during onboarding** | any number a hypothesis, ladder gap, or expectation is about: CRPS or log score of any model; usable information in bits; history gains; entropy of anything; home advantage; hot-hand statistics; variance components; league comparisons of efficiency |
| **O-lock** | never before P6 | enforced by `LockboxError` |

**Borderline rule.** If a data-quality check *could* be read as an outcome, it is classed as outcome and deferred. Example: "how often does a questionable player play" is the L3 calibration input.

## 3. Handling a data-contract mismatch

The log is `registry/DATA_CONTRACT_MISMATCHES.md`.

1. Record the mismatch with **structural evidence only**.
2. Assign one consequence using the frozen material alone:
   - Can the adapter resolve it cosmetically? → `NONE`.
   - Does a frozen fallback rule cover it (contract §1)? → `ADAPTER-FALLBACK`. The optimistic-imputation LEAKAGE-SENSITIVE rerun still applies.
   - Does the frozen experiment-to-level map (`docs/prereg/05` §3) already allow the experiment at the level that exists? → `LEVEL-DOWNGRADE`.
   - Does the data exist for some leagues or seasons only? → `SCOPE-RESTRICTION`.
   - Otherwise the experiment is `BLOCKED` and reported as **NOT EXECUTABLE (M<n>)**.
   - Only if a substitute definition would let it run → `PROXY-REQUIRED`, which is an amendment (§4). The original is still reported as NOT EXECUTABLE.
3. Never:
   - drop an experiment silently;
   - loosen a threshold because the data is thin (thin data produces INCONCLUSIVE, which is a legitimate pre-registered outcome);
   - redefine a target, stratum, or metric inside an adapter.

## 4. Amendments (your five conditions, operationalised)

| Condition | Mechanism |
|---|---|
| 1. Explicitly documented | a complete row in `registry/AMENDMENTS.md`; `freeze --amend A<n>` refuses without it |
| 2. Explain why the original cannot reasonably be executed | required column; must cite mismatch IDs when data-driven |
| 3. Data-driven necessity vs research preference | required `Class` column: `DATA-DRIVEN NECESSITY` / `ERROR CORRECTION` / `RESEARCH PREFERENCE`. A research preference never replaces the original: the original is still run and reported |
| 4. Preserve the original | `registry/archive/prereg-v1/` (permanent), plus `registry/archive/pre-A<n>/` created by `begin-amendment` before any edit. FREEZE entries are append-only, and entry 0 is test-protected |
| 5. Before looking at affected validation outcomes | the `Outcomes seen` column is copied from `EXPOSURE_LOG.md`. If any affected experiment's outcomes were seen, the amendment is labelled **POST-HOC** in every report of those hypotheses. The lockbox guard makes lockbox-informed amendments impossible without an explicit, logged unseal |

## 5. Onboarding sequence (next steps, not yet started)

1. Locate the corpus and log S1 exposure.
2. Write the adapter (`hoopslab.adapters.files.FileAdapter` mapping, or a new adapter). Changes are logged in `POSTFREEZE_CHANGELOG.md` as `engineering`.
3. Run `build_store(strict=False)`, the T4 knowledge-time audit, and the `docs/prereg/11` §7 coverage report (S1/S2). Record every mismatch.
4. Produce a per-experiment **executability table**: each frozen experiment × league → runnable as frozen / level-downgrade / scope-restricted / blocked (M<n>) / proxy-required.
5. Stop and report before P1 (the ladder), which is the first outcome-exposing step.
