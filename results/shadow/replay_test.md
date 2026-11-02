# Shadow pipeline replay test (development-period dates only)

Tips: ['2010-01-15 12:00', '2015-03-10 12:00']; games: 19; all checks passed: True; provider refuses lockbox: True, holdout: True

| game | issued | P1 units | matched | max |pmf diff| vs P1 | identical | leaky same | truncated same hash | overwrite refused | v1 unchanged | tamper detected | scored rows | counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nba:nba_stats:0020900571 | 27 | 27 | 27 | 0.0e+00 | True | True | True | True | True | True | 168 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 24, 'NOT_APPEARED': 3, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900572 | 26 | 26 | 26 | 0.0e+00 | True | True | True | True | True | True | 133 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 19, 'NOT_APPEARED': 7, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900573 | 27 | 27 | 27 | 0.0e+00 | True | True | True | True | True | True | 140 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 20, 'NOT_APPEARED': 7, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900574 | 26 | 26 | 26 | 0.0e+00 | True | True | True | True | True | True | 140 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 20, 'NOT_APPEARED': 6, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900575 | 28 | 28 | 28 | 0.0e+00 | True | True | True | True | True | True | 133 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 19, 'NOT_APPEARED': 9, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900576 | 27 | 27 | 27 | 0.0e+00 | True | True | True | True | True | True | 168 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 24, 'NOT_APPEARED': 3, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900577 | 31 | 31 | 31 | 0.0e+00 | True | True | True | True | True | True | 154 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 22, 'NOT_APPEARED': 9, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900578 | 28 | 28 | 28 | 0.0e+00 | True | True | True | True | True | True | 119 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 17, 'NOT_APPEARED': 11, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900579 | 29 | 29 | 29 | 0.0e+00 | True | True | True | True | True | True | 133 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 19, 'NOT_APPEARED': 10, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900580 | 28 | 28 | 28 | 0.0e+00 | True | True | True | True | True | True | 126 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 18, 'NOT_APPEARED': 10, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900581 | 28 | 28 | 28 | 0.0e+00 | True | True | True | True | True | True | 119 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 17, 'NOT_APPEARED': 11, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900582 | 26 | 26 | 26 | 0.0e+00 | True | True | True | True | True | True | 161 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 23, 'NOT_APPEARED': 3, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0020900583 | 29 | 29 | 29 | 0.0e+00 | True | True | True | True | True | True | 133 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 19, 'NOT_APPEARED': 10, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400946 | 29 | 29 | 29 | 0.0e+00 | True | True | True | True | True | True | 182 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 26, 'NOT_APPEARED': 3, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400947 | 30 | 30 | 30 | 0.0e+00 | True | True | True | True | True | True | 168 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 24, 'NOT_APPEARED': 6, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400948 | 30 | 30 | 30 | 0.0e+00 | True | True | True | True | True | True | 168 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 24, 'NOT_APPEARED': 6, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400949 | 29 | 29 | 29 | 0.0e+00 | True | True | True | True | True | True | 161 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 23, 'NOT_APPEARED': 6, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400950 | 29 | 29 | 29 | 0.0e+00 | True | True | True | True | True | True | 147 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 21, 'NOT_APPEARED': 8, 'OUT': 0, 'OUT_but_appeared': 0} |
| nba:nba_stats:0021400951 | 28 | 28 | 28 | 0.0e+00 | True | True | True | True | True | True | 140 | {'STATUS_CONFIRMED': 0, 'STATUS_EXPECTED': 0, 'UNKNOWN': 20, 'NOT_APPEARED': 8, 'OUT': 0, 'OUT_but_appeared': 0} |
