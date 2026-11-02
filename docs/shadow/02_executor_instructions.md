# 02. Instructions for the external executor (L2-A-SHADOW-v0.1)

*Exact procedure for a process that is NOT Research Claude. Nothing produced by steps 5-8 is sent back
to the research lane (docs/shadow/00_firewall.md).*

## 0. One-time setup

```bash
git clone https://github.com/traymoore07/basketball shadow_exec
cd shadow_exec && git checkout <PINNED_COMMIT>        # the commit named in shadow/MODEL_CARD_L2-A-SHADOW-v0.1.json
python -m venv .venv && .venv/Scripts/activate
pip install -e .                                       # pins: pandas 3.0.6, numpy 2.5.3, scipy 1.18.1 (see environment block)
set SHADOW_HOME=C:\Users\Tray\Documents\Claude\shadow_lane   # anywhere outside the research repository
python -m hoopslab.shadow.card                          # must print the implementation hash in the model card
python -m pytest -q tests/test_shadow.py                # 4 passed
```

Verify the printed `implementation.combined` equals `implementation_hash.combined` in the model card.
If it differs, stop: the checkout is not the frozen model.

## 1. Obtain current data (every forecast day)

From any provider (official league feed, a stats site, a vendor): the executor's choice. Write, per
cutoff, a snapshot directory `SHADOW_HOME/snapshots/<YYYY-MM-DD>T<HHMM>Z/` with three files (CSV or
parquet; schema in docs/shadow/01 section 1):

- `game.*`: today's games (status `scheduled`) plus every game referenced by the history, with the
  time your data path first held each row as `knowledge_time`;
- `player_game.*`: all completed player-games in your history window (league-wide; at least the
  current and three previous seasons; the longer the closer to P1), de-duplicated per (game, player)
  by keeping the row with the most seconds, no NULL stats, `knowledge_time` = when the final line was
  available to you;
- `eligible.*`: for each game, the players you want forecasts for, with `availability_state` in
  {CONFIRMED_ACTIVE, EXPECTED_ACTIVE, QUESTIONABLE_UNKNOWN, OUT}, `status_source`, `status_time`.
  Use CONFIRMED_ACTIVE only for a public confirmation (official active list, confirmed starters) with
  a publication time at or before the cutoff; EXPECTED_ACTIVE for a healthy rostered player without
  a formal confirmation; QUESTIONABLE_UNKNOWN for questionable/doubtful/no information; OUT when ruled
  out. Never guess a probability.

Do not include box rows of games that are not final. Do not include anything you learned after the
cutoff: the builder drops rows by `knowledge_time` and refuses `status_time` after the cutoff, but the
state labels are yours.

## 2. Construct the pregame snapshot and 3. run the model and 4. lock

For each game, between T-60 and tip-off:

```bash
python -m hoopslab.shadow.cli forecast --snapshot SHADOW_HOME/snapshots/<stamp> --game-id <game_id> --lane %SHADOW_HOME%
```

This builds the snapshot at the game's T-60 cutoff (dropping and counting any later rows), runs
L2-A-SHADOW-v0.1, computes the derived quantities, canonicalises, hashes, and writes
`SHADOW_HOME/forecasts/<game>/<forecast_id>.json` + `.sha256` once. It prints the forecast_id, the
lock hash, the information cutoff and the counts. If it refuses (created after tip, status after
cutoff, contract violation), fix the snapshot and run again: a new file, never an edit.

To issue earlier than T-60 (travel days), add `--allow-early`; the package then records the creation
time as its information cutoff. Never re-run a locked game "to update": that is a new version
(`v2`, new timestamp, new hash), and v1 stays.

If a tip changes or a game is postponed/cancelled after the lock:

```bash
python -m hoopslab.shadow.cli void --forecast-id <id> --reason tip_changed|postponed|cancelled
```

then forecast the new tip as a fresh package when the time comes.

## 5. Wait for the game to finish

Nothing to do. Do not touch the locked file. Do not pre-score.

## 6. Attach outcomes and 7. score

When the box score is final, write it as `SHADOW_HOME/outcomes/<game_id>.csv` (PLAYER_GAME columns,
one row per player who dressed; `seconds` 0 for DNP; `knowledge_time` = when it became final for you)
and run:

```bash
python -m hoopslab.shadow.cli score --game-id <game_id> --forecast-id <forecast_id> --box SHADOW_HOME/outcomes/<game_id>.csv --result-time <ISO UTC>
```

The scorer verifies the lock hash, scores every issued forecast whose player appeared with the frozen
metrics, labels each unit by the availability state recorded before the game (STATUS_CONFIRMED is
the primary population), counts late scratches, and appends to `SHADOW_HOME/ledger/scores.jsonl`.
If the league corrects a box score later, score again with `--revision 2`; revision 1 stays.

## 8. Store results and report

Everything lives in `SHADOW_HOME` (snapshots, forecasts, outcomes, ledger). Weekly:

```bash
python -m hoopslab.shadow.cli verify    # every locked package still matches its hash
python -m hoopslab.shadow.cli report    # cumulative table by model version, target, population, month
```

Hand the cumulative table (and nothing else) to Tray. Tray decides what, if anything, Research
Claude is told, and logs it as an exposure row. Never paste snapshots, box scores, per-game packages
or ledger rows into a research session.

## 9. Adding a later model version

Pull the pinned commit that carries it, run its card and tests, and forecast the same games with both
versions into the same lane. Older versions are never re-run on games they already locked, never
deleted, and never scored differently. Comparisons use games both versions forecast.

## 10. Checklist per game

- [ ] snapshot written with knowledge_time on every row
- [ ] `forecast` run after T-60 (or `--allow-early` on purpose) and before tip
- [ ] forecast_id and lock hash recorded
- [ ] game final -> outcomes file -> `score` once (revision 1)
- [ ] postponement / tip change / cancellation after lock -> `void`, then a new package later
- [ ] nothing in SHADOW_HOME shared with Research Claude
