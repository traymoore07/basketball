"""Population exclusions applied by the P1 drivers (logged as data issue DI-1 in the P1 report).

DI-1: corpus v1.0.2 carries 17 nba_stats schedule rows whose home and/or away team is the
unknown-team placeholder '0' (team_id 'nba:nba_stats:0'); none has a box score. Fifteen are
preseason / all-star rows (never forecast in P1); two are 2005 'playoffs' rows
(0040401000, 0040401001, tips 2005-06-27/28) that produced an empty as-of roster and crashed
MinutesRate.predict on a zero-row request set (frozen runner.run_backtest has the same gap).
"""
import pandas as pd

UNKNOWN_TEAM = "nba:nba_stats:0"


def drop_unknown_team_games(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped). A game is dropped iff either team id is the unknown placeholder."""
    bad = (games["home_team_id"] == UNKNOWN_TEAM) | (games["away_team_id"] == UNKNOWN_TEAM)
    return games[~bad], games[bad]
