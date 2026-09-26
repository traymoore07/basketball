# Pre-registered experimental blueprint: index

**Status:** FROZEN (manifest hash `ccb065025fb3d4874b4e9fe12a6b7ead3ad4eb3cab6661214efb2c3a17bd9c40`, 25 files) before exposure to the real NBA/WNBA/NCAA dataset. See `registry/FREEZE.json` for file hashes and `registry/AMENDMENTS.md` for any later change.

**Mission:** build the laboratory that decides which parts of the proposed probabilistic-sequence forecasting architecture actually work, and write down the hypotheses, metrics, and expected outcomes *before* seeing results. No final prediction model is fitted here.

| # | Deliverable | Document | Machine-readable / code |
|---|---|---|---|
| 1 | Experimental blueprint | [01_experimental_blueprint.md](01_experimental_blueprint.md) | – |
| 2 | Hypothesis registry | [02_hypothesis_registry.md](02_hypothesis_registry.md) (rendered) | `registry/hypotheses.yaml` |
| 3 | Baseline ladder | [03_baseline_ladder.md](03_baseline_ladder.md) | `registry/baseline_ladder.yaml`, `src/hoopslab/models/` |
| 4 | Pre-registered expectations | [04_preregistered_expectations.md](04_preregistered_expectations.md) | `registry/expectations.yaml` |
| 5 | Data contract | [05_data_contract.md](05_data_contract.md) | `src/hoopslab/contract/data_contract.yaml` |
| 6 | Leakage checklist | [06_leakage_checklist.md](06_leakage_checklist.md) | `src/hoopslab/backtest/leakage.py`, `tests/test_asof_and_leakage.py` |
| 7 | Temporal backtesting specification | [07_temporal_backtest_spec.md](07_temporal_backtest_spec.md) | `src/hoopslab/backtest/runner.py`, `src/hoopslab/asof.py` |
| 8 | Metric specification | [08_metric_spec.md](08_metric_spec.md) | `src/hoopslab/metrics/` |
| 9 | Synthetic test suite | [09_synthetic_test_suite.md](09_synthetic_test_suite.md) | `src/hoopslab/synthetic/`, `src/hoopslab/experiments/synthetic_suite.py`, `results/synthetic_suite/` |
| 10 | Repository scaffolding | [12_repository_guide.md](12_repository_guide.md) | `src/hoopslab/` |
| 11 | Real-dataset checklist | [11_real_data_checklist.md](11_real_data_checklist.md) | – |
| – | Cross-league design (NBA/WNBA/NCAA) | [10_cross_league.md](10_cross_league.md) | – |

Background: [../research/nba_sequence_modeling.md](../research/nba_sequence_modeling.md) (first research report).
