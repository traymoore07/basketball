import time, pandas as pd, numpy as np
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter
ad = CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A")
tables = ad.load_tables()
r = tables["ROSTER"]; pg = tables["PLAYER_GAME"]; g = tables["GAME"].drop_duplicates("game_id", keep="last").set_index("game_id")
tm = "nba:nba_stats:1610612754"
for pid in ["nba:cp:nba_stats:202344", "nba:cp:nba_stats:203590", "nba:cp:nba_stats:1628500"]:
    print(pid); print(r[r.player_id == pid][["team_id","valid_from","valid_to","record_version","knowledge_time"]].to_string())
    b = pg[pg.player_id == pid].copy(); b["tip"] = b.game_id.map(g.scheduled_tip)
    print("box rows:", len(b), "teams:", b.team_id.value_counts().to_dict(), "first/last tip", b.tip.min(), b.tip.max())
# how many open (never-closed) spells per team overall vs players seen in the last 20 games as of 2019-01-15
t = pd.Timestamp("2019-01-15 11:00", tz="UTC")
vis = r[r.knowledge_time <= t].sort_values("knowledge_time").drop_duplicates(["player_id","team_id","valid_from"], keep="last")
open_ = vis[vis.valid_to.isna() | (vis.valid_to > t)]
print("open spells per team at t (summary):", open_.groupby("team_id").size().describe().to_dict())
# players whose spell is open at t but who have no box row for that team in the last 60 days
pg2 = pg.copy(); pg2["tip"] = pg2.game_id.map(g.scheduled_tip)
recent = pg2[(pg2.tip <= t) & (pg2.tip > t - pd.Timedelta(days=60))].groupby(["player_id","team_id"]).size()
stale = open_[~open_.set_index(["player_id","team_id"]).index.isin(recent.index)]
print("open-at-t spells with no box row in the last 60 days:", len(stale), "of", len(open_))
print(stale.head(10)[["player_id","team_id","valid_from","record_version","knowledge_time"]].to_string())
