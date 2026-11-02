# Lane L1: PREGAME-CONTEXT-LIVE

*Lane card. Created 2026-11-02 (docs/lanes/00_lane_registry.md). Operated by the external executor, outside
the Research Claude environment. Research Claude created the contract, the validator, the fixture and the
documentation; it never operates the lane on current-season data and never reads a real snapshot.*

**Job.** Answer, before each forecast cutoff: WHO IS AVAILABLE TONIGHT? and WHAT ROLE WILL EACH PLAYER HAVE
IF AVAILABLE? as recorded facts with sources and times, never as inferred probabilities.

**Location.** Contract: `live_inputs/01_pregame_context_contract.md`; code: `src/hoopslab/pregame/contract.py`
(schema, validator, one-way mapping to the frozen baseline's ELIGIBLE table, minutes-distribution output
contract); fixture: `live_inputs/example_snapshot/` (synthetic ids only). Real snapshots: one directory per
information cutoff under `CONTEXT_HOME/live_inputs/<YYYY-MM-DD>T<HHMM>Z/`, outside the repository.

**Allowed.** Any pre-game source the executor chooses (official injury reports and active lists, starters
feeds, beat writers, vendors), each row stamped with its publication time and the executor's retrieval
time, both at or before the cutoff.

**Forbidden.** Turning UNKNOWN into ACTIVE (the validator and the mapping refuse it); marking an inferred
state as official; anything published after the cutoff; writing into the research repository; sending a
snapshot to Research Claude.

**Hand-offs.** To S0 / the v0.1 run inside X1: `python -m hoopslab.pregame.cli eligible --snapshot DIR --out
eligible.csv` writes the frozen four-state ELIGIBLE table (ACTIVE+official -> CONFIRMED_ACTIVE; ACTIVE
unofficial and PROBABLE -> EXPECTED_ACTIVE; QUESTIONABLE / DOUBTFUL / UNKNOWN -> QUESTIONABLE_UNKNOWN; OUT ->
OUT). To X1 challengers v0.3 / v0.4: the full snapshot. `validate` must pass first; the report carries the
table hashes and the `snapshot_id`.

**Status.** CREATED: contract fixed, validator and mapping tested on the fixture (`tests/test_lanes.py`),
no live source contacted, no real snapshot exists.
