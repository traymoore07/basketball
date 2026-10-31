import pandas as pd, numpy as np
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter, ROSTER_GAP_GAMES
ad = CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A")
tables = ad.load_tables()
r = tables["ROSTER"]; pg = tables["PLAYER_GAME"]; g = tables["GAME"].drop_duplicates("game_id", keep="last").set_index("game_id")
tip = g["scheduled_tip"]
obs = pg[["game_id","player_id","team_id"]].copy(); obs["tip"] = obs.game_id.map(tip)
team_games = obs.drop_duplicates(["team_id","game_id"])[["team_id","game_id","tip"]].sort_values(["team_id","tip"], kind="stable")
grp = team_games[team_games.team_id=="nba:nba_stats:1610612754"]
s = grp["tip"].dt.tz_convert("UTC").astype("int64").to_numpy()
print("sched dtype", s.dtype, "len", len(s), "first", s[:3], "sorted?", bool(np.all(np.diff(s)>=0)))
last = pd.Timestamp("2018-04-29 12:00", tz="UTC")
print("last.value", last.value)
i = np.searchsorted(s, last.value, side="right"); j = i + ROSTER_GAP_GAMES - 1
print("i", i, "j", j, "len", len(s), "gc", pd.Timestamp(int(s[j]), unit="ns", tz="UTC") if j < len(s) else None)
# how many version-2 rows are gap closes (valid_to not equal to the player's next spell start)?
r2 = r[r.record_version==2].sort_values(["player_id","valid_from"])
nxt = r[r.record_version==1].sort_values(["player_id","valid_from"]).groupby("player_id")["valid_from"].shift(-1)
r1 = r[r.record_version==1].sort_values(["player_id","valid_from"]).assign(next_from=nxt.values)
m = r2.merge(r1[["player_id","team_id","valid_from","next_from"]], on=["player_id","team_id","valid_from"])
print("version-2 rows", len(m), "transfer closes", int((m.valid_to==m.next_from).sum()), "gap closes", int((m.valid_to!=m.next_from).sum()))
print("final spells (no next):", int(r1.next_from.isna().sum()), "of which closed:", int(m[m.next_from.isna()].shape[0]))
