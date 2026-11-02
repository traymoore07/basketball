# P1 ladder run 1 (2026-10-29 23:19 to 2026-10-30 ~04:30 local) -- SUPERSEDED, retained for audit

Driver experiments/p1_ladder/run_season.py as of commit af42312 (+ the DI-1 exclusion patch for
seasons 2005, 2015-2017) enumerated forecast points in fixed 7-day chunks, reading the schedule as
known at chunk start - 1 day. GAME v1 rows are published exactly 7 days before tip, so every game on
the 7th day of each chunk was not yet visible at the snapshot and was silently never forecast
(2005: 202 of 1,314 P1-phase games; 2016: 111 of 1,316; 2001: 152 of 1,260; offset 6 mod 7 in every
season checked). Classification: BUGFIX C11 in the P1 driver (engineering file; no model, metric,
split, stratum or decision rule touched). Fixed driver derives the chunk length from the measured lag
(chunk_days = lag_days - 1) and asserts full coverage at the end of every season.

Contents: per-season requests/forecasts/scores/strata/done/log/stdout for seasons 1998-2017 (20 seasons;
2018-2020 were stopped mid-run), the two summarize trials on 8 seasons (summary.json,
summary_tables.md = the PRELIMINARY 8-season summary; NOT a final result), cluster_sums/pit_summaries,
di1_verification.json (run-1 populations), preflight_run1.json. Leakage results from the same period
remain valid (they sample tips from the full schedule) and stay in results/p1_ladder/.
Nothing in this folder is used by the final P1 report except as audit evidence.
