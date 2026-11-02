# Lane X1: CONTEXT-SHADOW (external challenger executor)

*Lane card and executor instructions. Created 2026-11-02 (docs/lanes/00_lane_registry.md). A separate
external executor, not Research Claude. Status: CREATED; version plan fixed; executor skeleton in
`src/hoopslab/pregame/cli.py`; no challenger model exists yet; no prospective forecast made.*

## 1. Model version plan (`VERSIONS.json`, immutable once a version is FROZEN)

| version | role | changes | status |
|---|---|---|---|
| `L2-A-SHADOW-v0.1` | BASELINE, permanent control | none, ever | FROZEN 2026-10-30 (card in `shadow/`) |
| `SEASON-AWARE-SHADOW-v0.2` | challenger 1 | historical weighting only (R1 artefact) | PLANNED |
| `AVAILABILITY-SHADOW-v0.3` | challenger 2 | + tonight's availability / roster status (L1) | PLANNED |
| `CONTEXT-SHADOW-v0.4` | challenger 3 | + rotation / minutes distribution + opportunity redistribution (R2) | PLANNED |

Do not skip to v0.4: attribution needs each step. Every version forecasts the same eligible games at the
same T-60 cutoff from the same L1 snapshot; each writes its own immutable forecast package per game; no
version's prospective results alter another version during the evaluation period; old forecasts are never
replaced. Prospective evaluation answers separately: season awareness (v0.2 - v0.1), player availability
(v0.3 - v0.2), rotation / minutes (v0.4's minutes component), opportunity / usage redistribution (v0.4's
redistribution component; the card of v0.4 must make the two separable, e.g. by an ablation version
`CONTEXT-SHADOW-v0.4-minutes-only` registered in VERSIONS.json before v0.4 goes live).

## 2. The full challenger decomposes (v0.4)

```
CURRENT ROSTER -> AVAILABILITY -> START / BENCH / ROLE -> MINUTES DISTRIBUTION
  -> USAGE / FGA / OPPORTUNITY DISTRIBUTION -> PER-MINUTE PRODUCTION -> PTS / REB / AST / FG3M / TOV / PRA
```

Each stage is a frozen component with its own hash in the card, so a failing component can be seen. No
single opaque network. Bench players facing an enlarged role are forecast as projected minutes x shrunken
per-minute rate with the role uncertainty in the minutes distribution, never as their old total-game average.

## 3. Locations

```
repository (pinned commit)    src/hoopslab/pregame/ (contract, versions, cli), src/hoopslab/shadow/ (v0.1),
                              shadow_context/VERSIONS.json, shadow_context/cards/MODEL_CARD_<version>.json
CONTEXT_HOME (outside git)    live_inputs/<cutoff>/            L1 snapshots
                              live_results/<version>/          per-version lane: forecasts/, outcomes/, ledger/
                              live_results/comparisons/        cumulative tables across versions (executor-made)
```

## 4. Executor procedure (per forecast day, once a version is FROZEN)

```bash
set HOOPSLAB_LANE=live
set CONTEXT_HOME=C:\Users\Tray\Documents\Claude\context_lane     # outside the repository
python -m hoopslab.pregame.cli versions                            # which versions are forecastable (hashes verified)
python -m hoopslab.pregame.cli validate --snapshot %CONTEXT_HOME%\live_inputs\<stamp> --out <stamp>_validation.json
python -m hoopslab.pregame.cli forecast --version L2-A-SHADOW-v0.1 --snapshot %CONTEXT_HOME%\live_inputs\<stamp> ^
        --shadow-snapshot <dir with game.* and player_game.*> --game-id <game_id>
python -m hoopslab.pregame.cli forecast --version SEASON-AWARE-SHADOW-v0.2 ...   # refused until FROZEN
python -m hoopslab.pregame.cli score  --version <v> --game-id <g> --forecast-id <id> --box <csv> --result-time <ISO>
python -m hoopslab.pregame.cli report --version <v>                # cumulative table for that version
```

The guard refuses `forecast`, `score`, `report` and `verify` unless `HOOPSLAB_LANE=live` and `CONTEXT_HOME`
is outside the repository, so a research session cannot run them by accident. The frozen baseline is run
unchanged through its own CLI (`hoopslab.shadow.cli`) with only the ELIGIBLE table derived from the L1
snapshot; its S0 lane in `SHADOW_HOME` is untouched by this.

## 5. Firewall (docs/shadow/00 applies in full)

Research Claude may see: this code, the frozen model definitions, operational health (versions
forecastable, snapshots validated, games forecast, refusals, hash verification). Research Claude may not
see: prospective outcomes, individual forecast errors, rankings, cumulative scores, current-season
calibration. Tray controls any exposure (an `O-shadow` ledger row with an aggregate). Nothing learned here
changes a frozen version; a model that learns from prospective outcomes is a new version.

## 6. Freezing a challenger (hand-off checklist)

- [ ] the lane spec it rests on is frozen (R1 / R2 changelog row) and its report says the artefact is supported
- [ ] code at a pinned commit; card written to `shadow_context/cards/` with lineage, changes, inputs, per-file and combined implementation hash, environment, card sha256
- [ ] `VERSIONS.json` entry switched to FROZEN with `card_sha256`, `implementation_combined`, `pinned_commit`
- [ ] `python -m hoopslab.pregame.cli versions` shows it forecastable; `tests/test_lanes.py` passes
- [ ] POSTFREEZE_CHANGELOG row; the executor pulls the pinned commit and verifies the printed hash against the card
