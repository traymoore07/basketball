"""E-STATE state variables per season (docs/p2/02_estate_spec.md section 3), as-of the segment's first event:
    def_bonus_l2m      defence has >= 1 team foul inside the last two minutes of the period, <= 120 s remain, not already in the bonus
    {def,off}_pf_max   max personal fouls (whole game, before the segment) among the five on-court players, capped at 6
    {def,off}_pf_trouble  players on court in foul trouble (fouls >= period + 1 in periods 1-4, >= 5 in OT), capped at 2
    {def,off}_pf_5     players on court with >= 5 personal fouls, capped at 2
    def_lineup_change  1 if the defence lineup differs from the defence's previous segment lineup (0 at its first)
    def_stint_len_s    seconds since the defence lineup last changed
Personal fouls = every foul event with a player id whose subtype is not a technical variant and not "Defense 3 Second".
Cross-check: the "(P#." tag of nba_stats foul descriptions must equal the running count computed here in >= 98% of
tagged personal fouls per season (written to results/p2_state/state_build_log.txt). Requires the E-HIST segment and
slow-variable parquets. Writes results/p2_state/state_{S}.parquet."""
import argparse
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "p2_hist"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.adapters.corpus_v102 import DEFAULT_DB  # noqa: E402
from segments import OUT as HIST_OUT, clock_s, elapsed  # noqa: E402
from slow_vars import TEAM_FOUL_SUBTYPES  # noqa: E402
from state_blocks import player_foul_features, seconds_left  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_state")
PTAG = re.compile(r"\(P(\d+)[.)]")


def is_personal(sub: str) -> bool:
    return ("Technical" not in sub) and (not sub.startswith("Defense 3 Second"))


def counts_before(groups: dict, keys, starts):
    """For each (key, start): number of sorted event times in groups[key] strictly before start."""
    res = np.zeros(len(starts), int)
    df = pd.DataFrame({"k": keys, "t": starts, "i": np.arange(len(starts))})
    for k, d in df.groupby("k", sort=False):
        arr = groups.get(k)
        if arr is not None:
            res[d["i"].to_numpy()] = np.searchsorted(arr, d["t"].to_numpy(), side="left")
    return res


def build_season(S: int, con, league="nba", source="nba_stats", prefix="nba", hist_dir=HIST_OUT, l2m_s=120.0):
    seg = pd.read_parquet(os.path.join(hist_dir, f"segments_{S}.parquet"))
    slow = pd.read_parquet(os.path.join(hist_dir, f"slow_{S}.parquet"))
    seg = seg.merge(slow[["game_id", "possession_seq", "def_in_bonus"]], on=["game_id", "possession_seq"], how="left")
    seg["def_in_bonus"] = seg["def_in_bonus"].fillna(0)
    seg = seg.sort_values(["game_id", "start_elapsed_s", "possession_seq"], kind="stable").reset_index(drop=True)
    ev = con.execute(f"""SELECT source_game_id, sequence, period, game_clock, team_id, player_id, event_subtype, description
                         FROM play_by_play WHERE league='{league}' AND source='{source}' AND season={S} AND event_family='foul'
                         ORDER BY source_game_id, sequence""").df()
    ev["clock_s"] = ev["game_clock"].map(clock_s)
    ev["elapsed"] = [elapsed(p, c) if not np.isnan(c) else np.nan for p, c in zip(ev["period"], ev["clock_s"])]
    ev = ev[ev["elapsed"].notna()].copy()
    ev["gid"] = f"{league}:{source}:" + ev["source_game_id"].astype(str)
    ev["team"] = f"{league}:{source}:" + ev["team_id"].astype("string").fillna("")
    ev["pid"] = f"{prefix}:" + ev["player_id"].astype("string").fillna("")
    ev["sub"] = ev["event_subtype"].astype("string").fillna("")
    personal = ev[ev["sub"].map(is_personal) & (ev["pid"] != f"{prefix}:")].copy()
    # cross-check against the running count in the description tag
    personal["running"] = personal.groupby(["gid", "pid"], sort=False).cumcount() + 1
    tags = personal["description"].astype("string").fillna("").str.extract(PTAG)[0]
    has = tags.notna()
    agree = float((tags[has].astype(int).to_numpy() == personal.loc[has, "running"].to_numpy()).mean()) if has.any() else np.nan
    # personal fouls before each segment for the ten on-court players
    pf_groups = {k: np.sort(d["elapsed"].to_numpy()) for k, d in personal.groupby(["gid", "pid"], sort=False)}
    out = pd.DataFrame({"game_id": seg["game_id"], "possession_seq": seg["possession_seq"]})
    n = len(seg)
    for side in ("def", "off"):
        L = np.stack(seg[f"{side}_lineup"].map(list).to_numpy())  # (n, 5)
        keys = list(zip(np.repeat(seg["game_id"].to_numpy(), 5), L.ravel()))
        starts = np.repeat(seg["start_elapsed_s"].to_numpy(float), 5)
        cnt = counts_before(pf_groups, keys, starts).reshape(n, 5)
        mx, trouble, five = player_foul_features(cnt, seg["period"].to_numpy())
        out[f"{side}_pf_max"], out[f"{side}_pf_trouble"], out[f"{side}_pf_5"] = mx, trouble, five
    # last-two-minutes bonus flag (team fouls, E-HIST subtypes)
    is_tf = ev["sub"].map(lambda s: any(s.startswith(k) for k in TEAM_FOUL_SUBTYPES))
    tf = ev[is_tf]
    tf_groups = {k: np.sort(d["elapsed"].to_numpy()) for k, d in tf.groupby(["gid", "team", "period"], sort=False)}
    left = seconds_left(seg)
    start = seg["start_elapsed_s"].to_numpy(float)
    win_start = start - np.clip(l2m_s - left, 0, None)  # = period_end - l2m_s when <= l2m_s remain
    keys = list(zip(seg["game_id"].to_numpy(), seg["def_team_id"].to_numpy(), seg["period"].to_numpy()))
    n_before = counts_before(tf_groups, keys, start)
    n_before_win = counts_before(tf_groups, keys, win_start)
    out["def_bonus_l2m"] = ((left <= l2m_s) & ((n_before - n_before_win) >= 1) & (seg["def_in_bonus"].to_numpy() == 0)).astype(int)
    # defence rotation
    g = seg.groupby(["game_id", "def_team_id"], sort=False)
    prev_lu = g["def_lineup"].shift(1)
    same = [None if p is None or isinstance(p, float) else tuple(p) == tuple(c) for p, c in zip(prev_lu, seg["def_lineup"])]
    out["def_lineup_change"] = [0 if s is None else int(not s) for s in same]
    stint_start = np.zeros(n)
    for (gid, team), d in g:
        last = d["start_elapsed_s"].iloc[0]
        prev = None
        for i, (lu, t0) in enumerate(zip(d["def_lineup"], d["start_elapsed_s"])):
            tl = tuple(lu)
            if prev is not None and tl != prev:
                last = t0
            stint_start[d.index[i]] = last
            prev = tl
    out["def_stint_len_s"] = seg["start_elapsed_s"].to_numpy() - stint_start
    info = {"season": S, "segments": int(n), "personal_fouls": int(len(personal)), "tagged": int(has.sum()), "tag_agreement": agree,
            "def_bonus_l2m_share": float(out["def_bonus_l2m"].mean()), "def_pf_trouble_share": float((out["def_pf_trouble"] > 0).mean()),
            "def_pf_5_share": float((out["def_pf_5"] > 0).mean()), "def_lineup_change_share": float(out["def_lineup_change"].mean())}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(DEFAULT_DB, read_only=True)
    log = open(os.path.join(OUT, "state_build_log.txt"), "a")
    for S in a.seasons:
        if S > 2020:
            raise PermissionError("<= 2020 only")
        t0 = time.time()
        out, info = build_season(S, con)
        out.to_parquet(os.path.join(OUT, f"state_{S}.parquet"), index=False)
        info["seconds"] = round(time.time() - t0, 1)
        info["gate_tag_agreement_ge_0.98"] = bool(info["tag_agreement"] >= 0.98)
        print(info, file=log, flush=True)
        print(info, flush=True)


if __name__ == "__main__":
    main()
