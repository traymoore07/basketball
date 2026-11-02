# Prospective shadow lane: L2-A-SHADOW-v0.1

Frozen P1 L2 (PlayerMean, half-life 5 appearances), mode A only: P(stat | the player appears) for
MIN, PTS, REB, AST, FG3M, TOV, PRA. Packaged so an external executor can forecast games that have
not happened yet without touching model code.

- `MODEL_CARD_L2-A-SHADOW-v0.1.json`: version, implementation hash (per file + combined), pmf
  construction, half-life, mode, required inputs, output schema, environment, card SHA-256.
  Regenerate with `python -m hoopslab.shadow.card`; the printed combined hash must match the card.
- Code: `src/hoopslab/shadow/` (interface, model, package, scoring, replay, cli, card).
- Docs: `docs/shadow/00_firewall.md` (research / shadow separation), `01_input_contract.md` (tables,
  edge cases, scoring policy), `02_executor_instructions.md` (step-by-step), `03_build_report.md`.
- Tests: `tests/test_shadow.py` (synthetic); `experiments/shadow/replay_test.py` (development-period
  replay against the P1 forecasts; results under `results/shadow/`).
- The pinned commit for an executor is the commit that carries this card (git log on the card file).

Research Claude never runs this against games after 2020-10-31; the replay provider refuses such cutoffs.
