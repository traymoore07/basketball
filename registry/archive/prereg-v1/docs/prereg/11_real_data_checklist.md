# 11. What I need from the real dataset when it arrives

This is a checklist for the handoff from the data-collection effort. It does not prescribe how that effort stores data. It lists what the adapter must be able to produce, and what I must be told. Items are ordered by how early they block work.

Legend:

- **[B]** blocks everything;
- **[L]** blocks a level (A–E);
- **[H]** blocks specific hypotheses;
- **[Q]** quality/provenance information that must be documented but does not block.

## 1. Provenance and knowledge time (read first)

- [ ] **[B]** For every source: how is `knowledge_time` determined? (Source timestamp, publication log, archive snapshot, or an imputation rule.) Which `knowledge_time_quality` does it map to?
- [ ] **[B]** Confirmation that `ingested_time` (scrape/backfill time) is stored separately and never substituted for knowledge time.
- [ ] **[B]** Versioning: are corrected box scores, updated injury statuses, reschedules, and roster changes kept as separate versions, or overwritten? If overwritten, which fields might be post-hoc?
- [ ] **[B]** Time zones: are all timestamps UTC, or is the local zone recorded per field?
- [ ] **[Q]** Known gaps, source switches (e.g. a change of play-by-play provider), and scraping failures, by league and season.

## 2. Level A: every league, every season available

- [ ] **[L-A]** GAME: game_id, league, season, phase, scheduled tip (UTC), home/away, neutral site, status incl. postponed/cancelled, and **schedule versions** if reschedules exist.
- [ ] **[L-A]** PLAYER_GAME for **every rostered player**, including DNP-CD and inactive, with status, seconds, and the full box (pts, fgm/fga, fg3m/fg3a, ftm/fta, oreb, dreb, ast, stl, blk, tov, pf, +/−).
- [ ] **[L-A]** TEAM_GAME incl. team rebounds and team turnovers.
- [ ] **[L-A]** ROSTER spells with transaction type and **announcement time** (not only the effective date): two-way, 10-day, hardship, G-League assignments, NCAA transfers.
- [ ] **[L-A]** PLAYER bio as versions (height/weight/position as listed at the time, if available), birth date, draft info.
- [ ] **[L-A]** RULESET per league and era, verified against official rulebooks: periods, period length, shot clock and offensive-rebound reset, three-point distance (arc and corner), foul limit, bonus rule, overtime length (`10_cross_league.md`).
- [ ] **[H: H3, H6, H9, L3+]** INJURY_REPORT **snapshots** (every published version) with publication time, status, and reason category. Coverage per league and season (NCAA coverage may be low; that is fine but must be stated).
- [ ] **[H: H11b]** PERSON_LINK across leagues (NCAA→NBA/WNBA, international), with the time at which each link could have been known.
- [ ] **[Q]** VENUE with time zone (and altitude, latitude/longitude if available).

## 3. Level B: play-by-play

- [ ] **[L-B]** EVENT: game_id, monotone sequence, period, clock remaining, elapsed seconds, event type from a documented vocabulary, actors (primary, secondary, blocker), made/missed, shot value, free-throw n-of-m, score after event, team events (team rebound / team turnover), offensive vs defensive rebound.
- [ ] **[L-B]** SHOT x/y with the coordinate convention documented (origin, units, orientation, which basket).
- [ ] **[Q]** Mapping from each source's event vocabulary to the canonical `event_type`, including known quirks: jump-ball violations, technical FTs, flagrant sequences, clear-path fouls, coach's challenges, replay reversals.
- [ ] **[Q]** Known PBP errors per source: out-of-order events, missing substitutions, duplicate events.

## 4. Level C: lineups and rotations

- [ ] **[L-C]** SUBSTITUTION with elapsed time and a timing-quality flag (exact / dead-ball batch / inferred), or LINEUP_STINT.
- [ ] **[L-C]** Period-start lineups (sources often omit them). Were they inferred? If yes, from which later events? (Leakage L-D1 for live use.)
- [ ] **[L-C]** STARTING_LINEUP announcements with announcement times (if available). Otherwise say so, and mode S/L forecasts will use realised starters labelled as such.
- [ ] **[L-C]** COACH_TENURE spells incl. interim coaches, with announcement times.
- [ ] **[L-C]** OFFICIAL_ASSIGNMENT (with assignment times if available).

## 5. Level D: tracking (if any)

- [ ] **[L-D]** Which leagues and seasons, provider, frame rate, coordinate system, and whether pose data exists.
- [ ] **[L-D]** Derived tables if raw frames are unavailable: matchups (partial possessions), closest defender and distance per shot, touches, dribbles, shot clock per event.
- [ ] **[Q]** Licence constraints on storage and derived-data publication.

## 6. Level E / optional

- [ ] **[L-E]** Anything team-internal (load, minute restrictions, intended rotations). Not expected.
- [ ] **[H: H14]** MARKET_SNAPSHOT: book, market type, line, prices, snapshot time (UTC). Only if snapshot times are honest. Never a single "closing" value without a timestamp.

## 7. Coverage report I will produce on arrival (P0), and what I need to be told if it looks wrong

| Check | Target |
|---|---|
| Contract validation (`build_store(strict=True)`) | zero violations at the claimed level |
| T4 knowledge-time audit | passes; unknown-quality share reported per table |
| Share of PLAYER_GAME rows with status ≠ played | > 0 in every season (else DNPs are missing: L-E1) |
| Injury-report coverage | per league-season share of games with ≥ 1 snapshot |
| Possessions per team-game parsed from PBP vs box-score estimate | within ±3 on 95% of games |
| Sum of player seconds per team-game | 14,400 × (1 + OT periods × 5/48), within ±60 s on 99% of games (NBA; league-specific) |
| PBP points vs box-score points | exact match on ≥ 99.5% of games |
| Lineup reconstruction: 5 players on court at all times | ≥ 99% of possessions (per league; NCAA reported separately) |
| ID stability | no player_id reused for two people; no person with two ids without a PERSON_LINK |

## 8. Sample for manual audit

- [ ] 20 randomly chosen games per league-season, with raw source files, so that I can hand-check parsing, lineups, and knowledge times before any modelling.

## 9. Delivery mechanics (whatever is convenient)

- [ ] Stable snapshot identifiers (a hash or version tag per delivery), so every result can cite the exact data snapshot.
- [ ] Either parquet/CSV plus a column mapping for `hoopslab.adapters.files.FileAdapter`, or access to whatever store the collection project uses. I will write the adapter.
