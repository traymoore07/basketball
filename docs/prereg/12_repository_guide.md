# 12. Repository guide (scaffolding and interfaces)

```
basketball/
├── pyproject.toml                   package `hoopslab` (numpy, scipy, pandas, pyyaml; pytest for tests)
├── registry/                        FROZEN pre-registration artefacts (canonical, machine-readable)
│   ├── hypotheses.yaml              hypothesis registry (H1a … H14)
│   ├── baseline_ladder.yaml         ladder L0–L14 + oracles, credit rule
│   ├── expectations.yaml            pre-registered expectations with strength labels
│   ├── FREEZE.json                  SHA-256 manifest of frozen files (append-only entries)
│   └── AMENDMENTS.md                amendment log (empty at freeze)
├── docs/
│   ├── research/nba_sequence_modeling.md    first research report
│   └── prereg/00–12                         this blueprint
├── src/hoopslab/
│   ├── contract/        data_contract.yaml + schema.py (load, validate, data level)
│   ├── adapters/        base.py (DataAdapter, build_store), synthetic.py, files.py (real-data mapping)
│   ├── asof.py          AsOfStore / AsOfView: knowledge-time visibility, versions, truncate/perturb
│   ├── features.py      as-of feature helpers (history, rosters, injury status, EW weights)
│   ├── targets.py       target definitions, supports, bins, conditioning modes
│   ├── forecast.py      PlayerForecast (p_appear, conditional pmfs, joint samples, team samples)
│   ├── models/          base.py (Forecaster), baselines.py (L0–L4), structural.py (L10–L12),
│   │                    event.py (event-level multinomial models, feature blocks)
│   ├── sim/engine.py    vectorised possession engine (synthetic truth + reference simulator)
│   ├── metrics/         scoring.py (CRPS, log, RPS, Brier, PIT, energy, variogram, twCRPS),
│   │                    compare.py (paired block bootstrap, Holm, decision rule, interactions),
│   │                    information.py (usable information, plug-in contrast, arithmetic coder)
│   ├── backtest/        runner.py (rolling origin, requests, lockbox), leakage.py (T1–T7),
│   │                    strata.py (as-of strata)
│   ├── diagnostics/     joint.py (H8 instruments), realism.py (gates), changepoint.py (CUSUM)
│   ├── synthetic/       world.py (ground-truth worlds → contract tables), scenarios.py (A–H)
│   ├── experiments/     history.py (E-HIST), synthetic_suite.py (instrument verification)
│   └── registry/        validate / render / freeze / verify
├── tests/               scoring properness, comparison logic, as-of + leakage detectors
├── experiments/toy_demos.py         first-report demos
└── results/synthetic_suite/         verification outputs (JSON)
```

## Interfaces that the real dataset and future models plug into

| Interface | Where | Contract |
|---|---|---|
| `DataAdapter.load_tables() -> dict[str, DataFrame]` | `adapters/base.py` | canonical tables per `data_contract.yaml`, honest knowledge times |
| `build_store(adapter, strict=True) -> (AsOfStore, report)` | `adapters/base.py` | validates and reports the data level |
| `AsOfView.table(name)` | `asof.py` | the only data access for models and features |
| `Forecaster.fit(view)`, `Forecaster.predict(view, requests, targets) -> PlayerForecast` | `models/base.py` | full distributions; joint samples optional |
| `BacktestSpec`, `run_backtest(store, spec, models)` | `backtest/runner.py` | rolling origin, as-of requests, lockbox guard |
| `score_forecasts(store, forecasts, targets, modes)` | `backtest/runner.py` | long table of per-unit losses and PITs |
| `compare_losses`, `stratum_interaction`, `apply_decisions` | `metrics/compare.py` | pre-registered inference and decisions |
| Leakage tests `truncation_invariance(store, factory, times)` etc. | `backtest/leakage.py` | `factory(store) -> fresh model` |

## Commands

```bash
pip install -e .[dev]
pytest -q                                            # unit tests + leakage detectors (~1 min)
python -m hoopslab.experiments.synthetic_suite       # instrument verification on synthetic worlds (~10 min)
python -m hoopslab.registry validate | render | verify
```

## Adding a model later (after the freeze)

1. Subclass `Forecaster`, set `name`, `rung`, `data_level`, `info_set`.
2. Read data only through `view.table(...)`. Build every cache from the view.
3. Pass T1, T2, T3, T5, T7 on ≥ 20 sampled forecast times (`tests/` shows the pattern).
4. Register its configuration hash before any lockbox run.
