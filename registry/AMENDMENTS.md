# Amendment log

Any change to a file listed in `registry/FREEZE.json` after the freeze is recorded here
*before* running `python -m hoopslab.registry freeze --amend "<reason>"`.

The original pre-registration is preserved verbatim, and permanently, in `registry/archive/prereg-v1/`. That directory was materialised from the git objects of freeze commit `843caaa` and is verified against FREEZE entry 0 by `tests/test_registry.py`. An amendment never edits that archive or FREEZE entry 0. It adds a new archive snapshot and a new FREEZE entry.

## Required procedure (docs/onboarding/00_onboarding_protocol.md §4)

1. `python -m hoopslab.registry begin-amendment A<n>` snapshots the current frozen files into `registry/archive/pre-A<n>/`. It refuses if the frozen files already differ from the last FREEZE entry.
2. Add the row below, with every field filled, **before** editing any frozen file.
3. Edit the frozen files.
4. `python -m hoopslab.registry freeze --amend A<n>` appends a FREEZE entry. It refuses if this log has no complete row for A<n>.

## Field definitions

- **Why the original cannot reasonably be executed:** a concrete obstacle, citing mismatch IDs (`M<n>` in `DATA_CONTRACT_MISMATCHES.md`) where applicable.
- **Class:**
  - `DATA-DRIVEN NECESSITY`: the original plan is impossible or invalid with the data that exists.
  - `ERROR CORRECTION`: the frozen text or code is wrong on its own terms, for example a bug in a metric.
  - `RESEARCH PREFERENCE`: the original is executable, but I would now rather do something else.

  A research preference never replaces the original. The original is still run and reported, and the preferred version is added as a clearly labelled secondary analysis.
- **Outcomes seen:** which validation outcomes of the *affected* experiments had been observed when the amendment was written: `none` / `development window` / `lockbox`. This is copied from `EXPOSURE_LOG.md`, not from memory.
- **Report label:** `PRE-OUTCOME` if outcomes seen = none; otherwise `POST-HOC`. The label travels with every result of every affected hypothesis.

| ID | Date (UTC) | Frozen files touched | Original (archive path) → new | Why the original cannot reasonably be executed | Class | Linked mismatches | Hypotheses / experiments affected | Outcomes seen | Report label |
|---|---|---|---|---|---|---|---|---|---|

*(no amendments)*
