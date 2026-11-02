"""Automated leakage detectors (docs/prereg/06_leakage_checklist.md).

Dynamic tests take a model FACTORY: a callable factory(store) -> fresh model. The factory
receives the (truncated / perturbed / relabelled) store and must build EVERYTHING the model
uses (caches, precomputed feature tables, fitted encoders) from that store. The tests are
end-to-end: a model that reads a side channel (a global, a precomputed file built from
the full dataset) would evade them. So side channels are forbidden (checklist L-X1),
and T5 scans for them statically.
  T1  truncation invariance     forecasts at t are identical when every record with
                                knowledge_time > t is physically deleted
  T2  future-perturbation       ... when outcome columns of not-yet-known records are shuffled
  T3  ID relabel invariance     ... when opaque IDs are consistently relabelled
Static / data audits:
  T4  knowledge-time audit      outcomes known before they could exist, ingestion-time
                                masquerading as knowledge time, pre-game use of late reports
  T5  access scan               model source files must not touch raw-store internals or
                                ingestion timestamps
  T6  population check          the evaluation population includes non-appearing players
  T7  feature recompute test    as-of features equal features recomputed on a truncated store
Runtime guards live elsewhere: AsOfView (no raw access) and LockboxGuard (runner.py).
"""

from __future__ import annotations

import inspect
import re
from typing import Callable

import numpy as np
import pandas as pd

from ..asof import AsOfStore, to_utc
from ..backtest.runner import make_requests


def _forecast_at(store: AsOfStore, factory: Callable, t, games: pd.DataFrame, targets):
    view = store.view(t)
    req = make_requests(view, games)
    m = factory(store)
    m.fit(view)
    return req, m.predict(view, req, targets)


def _max_diff(f1, f2, targets) -> float:
    d = float(np.max(np.abs(f1.p_appear - f2.p_appear))) if len(f1.p_appear) else 0.0
    for t in targets:
        d = max(d, float(np.max(np.abs(f1.cond[t] - f2.cond[t]))))
    return d


def _games_at(store: AsOfStore, t, lead):
    g = store.view(t).table("GAME")
    return g[(g["scheduled_tip"] == t + lead) & (g["status"] == "scheduled")]


def truncation_invariance(store: AsOfStore, factory: Callable, times, lead=pd.Timedelta(minutes=60),
                          targets=("PTS", "MIN"), tol: float = 1e-9) -> dict:
    """T1. The strongest general-purpose leak test: delete the future, compare outputs."""
    worst = 0.0
    for t in times:
        t = to_utc(t)
        games = _games_at(store, t, lead)
        if not len(games):
            continue
        _, f_full = _forecast_at(store, factory, t, games, list(targets))
        _, f_trunc = _forecast_at(store.truncated(t), factory, t, games, list(targets))
        worst = max(worst, _max_diff(f_full, f_trunc, targets))
    return {"test": "T1_truncation", "max_abs_diff": worst, "passed": worst <= tol}


def future_perturbation_invariance(store: AsOfStore, factory: Callable, times, lead=pd.Timedelta(minutes=60),
                                   targets=("PTS", "MIN"), tol: float = 1e-9, seed: int = 0) -> dict:
    """T2. Shuffle not-yet-known outcomes (incl. the target game's own box score)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in times:
        t = to_utc(t)
        games = _games_at(store, t, lead)
        if not len(games):
            continue
        _, f1 = _forecast_at(store, factory, t, games, list(targets))
        _, f2 = _forecast_at(store.perturbed_future(t, rng), factory, t, games, list(targets))
        worst = max(worst, _max_diff(f1, f2, targets))
    return {"test": "T2_future_perturbation", "max_abs_diff": worst, "passed": worst <= tol}


def relabel_ids(tables: dict[str, pd.DataFrame], seed: int = 0) -> tuple[dict, dict]:
    """Consistently relabel player IDs everywhere (incl. tuple lineup columns)."""
    rng = np.random.default_rng(seed)
    ids = set()
    for df in tables.values():
        for c in df.columns:
            if c.endswith("player_id") or c.endswith("_id") and c.startswith(("user", "assister", "rebounder",
                                                                               "stealer", "blocker", "fouler")):
                ids.update(df[c].dropna().unique())
    ids = sorted(ids)
    new = [f"X{n:07d}" for n in rng.choice(10 ** 7, size=len(ids), replace=False)]
    mp = dict(zip(ids, new))
    out = {}
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if c in ("off_lineup", "def_lineup"):
                df[c] = [tuple(mp.get(p, p) for p in lu) for lu in df[c]]
            elif (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])) and (c.endswith("player_id") or c in (
                    "user_player_id", "assister_id", "rebounder_id", "stealer_id", "blocker_id", "fouler_id")):
                df[c] = df[c].map(lambda x: mp.get(x, x))
        out[name] = df
    return out, mp


def id_relabel_invariance(tables: dict, factory: Callable, times, lead=pd.Timedelta(minutes=60),
                          targets=("PTS",), tol: float = 1e-9) -> dict:
    """T3. Models must not extract information from the content or ordering of opaque IDs."""
    s1 = AsOfStore(tables)
    t2, mp = relabel_ids(tables)
    s2 = AsOfStore(t2)
    worst = 0.0
    for t in times:
        t = to_utc(t)
        games = _games_at(s1, t, lead)
        if not len(games):
            continue
        r1, f1 = _forecast_at(s1, factory, t, games, list(targets))
        r2, f2 = _forecast_at(s2, factory, t, games, list(targets))
        k1 = f1.keys.assign(pid=f1.keys["player_id"].map(mp)).reset_index()
        k2 = f2.keys.reset_index()
        m = k1.merge(k2, left_on=["game_id", "pid"], right_on=["game_id", "player_id"])
        if len(m) != len(k1):
            return {"test": "T3_id_relabel", "passed": False, "detail": "request sets differ after relabel"}
        i1, i2 = m["index_x"].to_numpy(), m["index_y"].to_numpy()
        for tgt in targets:
            worst = max(worst, float(np.max(np.abs(f1.cond[tgt][i1] - f2.cond[tgt][i2]))))
        worst = max(worst, float(np.max(np.abs(f1.p_appear[i1] - f2.p_appear[i2]))))
    return {"test": "T3_id_relabel", "max_abs_diff": worst, "passed": worst <= tol}


def knowledge_time_audit(tables: dict[str, pd.DataFrame], min_game_s: float = 5400.0) -> dict:
    """T4. Data-level checks on knowledge times (does not need a model)."""
    issues = []
    games = tables.get("GAME")
    tips = None
    if games is not None:
        tips = games.drop_duplicates("game_id", keep="last").set_index("game_id")["scheduled_tip"]
    for name in ("PLAYER_GAME", "TEAM_GAME", "POSSESSION", "EVENT"):
        df = tables.get(name)
        if df is None or tips is None:
            continue
        tip = pd.to_datetime(df["game_id"].map(tips), utc=True)
        too_early = (pd.to_datetime(df["knowledge_time"], utc=True) < tip + pd.Timedelta(seconds=min_game_s))
        if too_early.any():
            issues.append(f"{name}: {int(too_early.sum())} outcome rows known < {min_game_s/60:.0f} min after tip")
    for name, df in tables.items():
        if "ingested_time" in df.columns and len(df):
            same = (df["ingested_time"] == df["knowledge_time"]).mean()
            if same > 0.99:
                issues.append(f"{name}: knowledge_time == ingested_time for {same:.0%} of rows")
        if "knowledge_time_quality" in df.columns:
            unk = (df["knowledge_time_quality"] == "unknown").mean()
            if unk > 0:
                issues.append(f"{name}: {unk:.1%} rows with unknown knowledge time (fallback rules apply)")
    ir = tables.get("INJURY_REPORT")
    if ir is not None and tips is not None and len(ir):
        after = pd.to_datetime(ir["knowledge_time"], utc=True) > pd.to_datetime(ir["game_id"].map(tips), utc=True)
        if after.any():
            issues.append(f"INJURY_REPORT: {int(after.sum())} rows published after tip (must not inform pre-game)")
    return {"test": "T4_knowledge_time_audit", "issues": issues, "passed": not any(
        "known <" in i or "ingested_time" in i for i in issues)}


FORBIDDEN = [r"full_table\(", r"_AsOfView__store", r"ingested_time", r"\._tables\b", r"truth\["]


def static_access_scan(modules) -> dict:
    """T5. Model code may not reach around the as-of view."""
    hits = []
    for mod in modules:
        src = inspect.getsource(mod)
        for pat in FORBIDDEN:
            for m in re.finditer(pat, src):
                line = src[: m.start()].count("\n") + 1
                hits.append(f"{mod.__name__}:{line}: {pat}")
    return {"test": "T5_static_access_scan", "hits": hits, "passed": not hits}


def population_check(requests: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """T6. The evaluation population must not be selected on the outcome."""
    m = requests.merge(outcomes[["game_id", "player_id", "appeared"]], on=["game_id", "player_id"], how="left")
    non_appear = float((~m["appeared"].fillna(False).astype(bool)).mean())
    return {"test": "T6_population", "share_non_appearing": non_appear, "passed": non_appear > 0.0}


def feature_recompute_test(store: AsOfStore, feature_fn: Callable, times, tol: float = 1e-12) -> dict:
    """T7. Any as-of feature function must give identical output on a truncated store."""
    worst = 0.0
    for t in times:
        t = to_utc(t)
        a = feature_fn(store.view(t))
        b = feature_fn(store.truncated(t).view(t))
        if isinstance(a, pd.DataFrame):
            a, b = a.reset_index(drop=True), b.reset_index(drop=True)
            if a.shape != b.shape:
                return {"test": "T7_feature_recompute", "passed": False, "detail": "shape mismatch"}
            num = a.select_dtypes("number").columns
            worst = max(worst, float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))) if len(num) else 0.0)
            if not a.drop(columns=num).equals(b.drop(columns=num)):
                return {"test": "T7_feature_recompute", "passed": False, "detail": "non-numeric mismatch"}
        else:
            worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return {"test": "T7_feature_recompute", "max_abs_diff": worst, "passed": worst <= tol}
