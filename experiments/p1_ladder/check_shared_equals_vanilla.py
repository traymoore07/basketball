"""Bit-identity check: shared-history P1 models == frozen baselines on sampled forecast days (S1)."""
import sys, time
import numpy as np, pandas as pd
from hoopslab.adapters.base import build_store
from hoopslab.adapters.corpus_v102 import CorpusV102Adapter
from hoopslab.backtest.runner import make_requests
from hoopslab.models import baselines as B
from hoopslab.targets import CORE_TARGETS
from models import ladder_models
store, _ = build_store(CorpusV102Adapter(league="nba", family="nba_stats", seasons=None, level="A"), strict=True)
vanilla = [B.Climatology(), B.PlayerMean(None), B.PlayerMean(5.0, name="L2_recent_hl5"), B.PlayerMean(10.0, name="L2_recent_hl10"),
           B.PlayerMean(20.0, name="L2_recent_hl20"), B.PlayerMean(40.0, name="L2_recent_hl40"),
           B.MinutesRate(shrink=False), B.MinutesRate(shrink=True, param_uncertainty=True), B.MinutesRate(shrink=True, param_uncertainty=False)]
shared = ladder_models()
worst = 0.0
for day in ["2001-03-10", "2009-12-01", "2016-04-05", "2020-08-20"]:
    t = pd.Timestamp(day + " 11:00", tz="UTC")
    v1, v2 = store.view(t), store.view(t)
    gg = v1.table("GAME"); games = gg[(gg.scheduled_tip == t + pd.Timedelta(minutes=60)) & (gg.status == "scheduled")]
    if not len(games): print(day, "no games"); continue
    req = make_requests(v1, games)
    t0 = time.time()
    for a, b in zip(vanilla, shared):
        a.fit(v1); fa = a.predict(v1, req, CORE_TARGETS)
        b.fit(v2); fb = b.predict(v2, req, CORE_TARGETS)
        assert a.name == b.name, (a.name, b.name)
        d = float(np.max(np.abs(fa.p_appear - fb.p_appear)))
        for tg in CORE_TARGETS: d = max(d, float(np.max(np.abs(fa.cond[tg] - fb.cond[tg]))))
        worst = max(worst, d)
    t1 = time.time()
    for b in shared: b.fit(store.view(t))
    print(day, "games", len(games), "req", len(req), "max |diff|", worst, f"(vanilla+shared {t1-t0:.1f}s)")
# shared-only timing for one day
t = pd.Timestamp("2016-04-05 11:00", tz="UTC"); v = store.view(t); gg = v.table("GAME")
games = gg[(gg.scheduled_tip == t + pd.Timedelta(minutes=60)) & (gg.status == "scheduled")]; req = make_requests(v, games)
t0 = time.time()
for b in shared: b.fit(v); b.predict(v, req, CORE_TARGETS)
print("shared models, one day, all 9:", round(time.time() - t0, 1), "s")
print("IDENTICAL" if worst == 0.0 else f"DIFFER max {worst}")
