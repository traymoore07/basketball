# basketball

A pre-registered scientific framework for **probabilistic** forecasting of basketball player statistics and game outcomes across the NBA, WNBA, and NCAA. The aim is calibrated distributions, not point estimates.

**Current status: methodology frozen (freeze commit `843caaa`, manifest `ccb06502…`); integrity re-verified 2026-09-27; no real data seen; no model fitted.** See [`docs/onboarding/00_onboarding_protocol.md`](docs/onboarding/00_onboarding_protocol.md) for how the frozen plan meets the real corpus (exposure ledger, data-contract mismatches, amendment rules).

| Start here | |
|---|---|
| [`docs/research/nba_sequence_modeling.md`](docs/research/nba_sequence_modeling.md) | First research report: which ideas from information theory, probability, and ML are real vs. analogy; proposed architecture |
| [`docs/prereg/00_index.md`](docs/prereg/00_index.md) | **Pre-registered experimental blueprint**: hypotheses, baseline ladder, expectations, data contract, leakage checklist, temporal backtesting, metrics, synthetic test suite, cross-league design, real-data checklist |
| [`registry/`](registry/) | Machine-readable, hash-frozen registry (`hypotheses.yaml`, `baseline_ladder.yaml`, `expectations.yaml`, `FREEZE.json`, `AMENDMENTS.md`) |
| [`src/hoopslab/`](src/hoopslab/) | The laboratory: data contract, as-of engine, adapters, scoring rules, paired inference and decision rule, backtest runner, leakage detectors, synthetic worlds, reference models (ladder L0–L4, structural simulator L10–L12), information-theoretic instruments |

```bash
pip install -e .[dev]
pytest -q -m "not slow"                           # unit tests + leakage detectors
pytest -q -m slow                                 # quick synthetic instrument checks
python -m hoopslab.experiments.synthetic_suite    # full instrument verification (~10 min)
python -m hoopslab.registry verify                # frozen pre-registration intact? (tiers 1-2, archive, logs)
```

[`experiments/toy_demos.py`](experiments/toy_demos.py) holds the toy numerical demonstrations from the first report.
