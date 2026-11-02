import sys, os, time
sys.path.insert(0, r"C:\Users\Tray\Documents\Claude\basketball\src")
import pandas as pd
from hoopslab.adapters.base import build_store
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter
from hoopslab.backtest.runner import make_requests
PHASES = {"regular", "postseason", "play_in", "in_season_tournament"}
seasons = [int(x) for x in sys.argv[1:]] or [2005]
t0 = time.time()
store, _ = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A"), strict=True)
print("store", round(time.time()-t0), flush=True)
g = store.full_table("GAME")
g1 = g[(g["record_version"] == 1) & g["phase"].isin(PHASES)].copy()
g1["season"] = g1["season_id"].str.split("-").str[1].astype(int)
R = store.full_table("ROSTER")
print("roster cols", list(R.columns), flush=True)
for S in seasons:
    gs = g1[g1["season"] == S].sort_values("scheduled_tip")
    bad = []
    for _, gm in gs.iterrows():
        at = gm["scheduled_tip"]
        vt = R["valid_to"]
        m = R["team_id"].isin([gm["home_team_id"], gm["away_team_id"]]) & (R["valid_from"] <= at) & (vt.isna() | (vt > at))
        n_full = int(m.sum())
        if n_full == 0:
            bad.append((gm["game_id"], str(at), gm["phase"], gm["home_team_id"], gm["away_team_id"], gm.get("status")))
    print(f"season {S}: games={len(gs)} empty_full_roster={len(bad)}", flush=True)
    for b in bad[:20]:
        print("  ", b, flush=True)
    # as-of check for the bad games and a couple of neighbours
    for b in bad[:5]:
        gid, at = b[0], pd.Timestamp(b[1])
        t = at - pd.Timedelta(minutes=60)
        view = store.view(t)
        gg = view.table("GAME")
        games = gg[gg["game_id"] == gid]
        print("   as-of rows for", gid, "games_visible", len(games), "status", games["status"].tolist(),
              "req", len(make_requests(view, games)) if len(games) else None, flush=True)
        # which teams have roster spells around that time
        for tm in (b[3], b[4]):
            rr = R[R["team_id"] == tm].sort_values("valid_from")
            near = rr[(rr["valid_from"] <= at + pd.Timedelta(days=60)) & (rr["valid_from"] >= at - pd.Timedelta(days=400))]
            print("     team", tm, "spells near:", near["valid_from"].min(), near["valid_from"].max(), "valid_to max", near["valid_to"].max(), "n", len(near), flush=True)
