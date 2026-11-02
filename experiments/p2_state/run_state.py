"""E-STATE driver (docs/p2/02_estate_spec.md sections 4-5): nested observable states S0..S7, each with and
without the frozen within-game history (GL + L1 + L3 + L10 + EW), prequential by month on NBA nba_stats
segments, in the frozen multinomial logit (primary) and LightGBM with the E-HIST fixed parameters
(replication on identical events).

Per month, family mnl (23 fits): S_k, S_k_H (k = 0..7); S0_PC, S0_PC_H; S0_GL, S6_GL, S7_GL; controls
S6_permstate (the six state blocks permuted across events within the E-HIST state cells, train and test)
and S6_permhist (GL + history permuted within cells). Family gbm (24 fits): S_k, S_k_H, S0_PC, S0_PC_H and
leave-one-block-out from S6. Also bits_NULL: the training class frequencies (for the frozen H2 threshold).
Writes results/p2_state/bits/{family}_{month}.parquet (+ .json)."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "p2_hist"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoopslab.metrics.information import bits_per_event  # noqa: E402
from hoopslab.models.event import EventDesign, MultinomialLogit, event_classes  # noqa: E402
from run_hist import L2, OUT as HIST_OUT, TRAIN_DAYS, load_segments, state_cells  # noqa: E402
from run_hist import fit_predict as _fit_predict_hist  # noqa: E402
from state_blocks import BLOCK_ORDER, STATE_COLS, state_blocks  # noqa: E402

OUT = os.path.join(ROOT, "results", "p2_state")
HIST = ["GL", "L1", "L3", "L10", "EW"]
STATE_BLOCKS = BLOCK_ORDER[:-1]  # FB GAP TM SC PF ROT (PC is the seventh step)
MAXITER = 3000  # E-HIST used 1000; on the timing month S6/S6_H hit that cap and converged by 2000 (results/p2_state/convergence_check.json)


def fit_predict(family, Xtr, ytr, Xte):
    if family == "mnl":
        m = MultinomialLogit(l2=L2, maxiter=MAXITER).fit(Xtr, ytr)
        return m.predict_proba(Xte), bool(m.converged)
    return _fit_predict_hist(family, Xtr, ytr, Xte)


def chain():
    return {f"S{k}": ["S"] + BLOCK_ORDER[:k] for k in range(8)}


def sets_for(family):
    C = chain()
    sets = {}
    for k in range(8):
        sets[f"S{k}"] = C[f"S{k}"]
        sets[f"S{k}_H"] = C[f"S{k}"] + HIST
    sets["S0_PC"] = ["S", "PC"]
    sets["S0_PC_H"] = ["S", "PC"] + HIST
    if family == "mnl":
        for k in (0, 6, 7):
            sets[f"S{k}_GL"] = C[f"S{k}"] + ["GL"]
    else:
        for b in STATE_BLOCKS:
            sets[f"S6_no_{b}"] = ["S"] + [x for x in STATE_BLOCKS if x != b]
    return sets


LEAGUE = "nba"  # "wnba": portability run on results/p2_state/wnba (docs/p2/02 section 7)


def dirs():
    if LEAGUE == "nba":
        return HIST_OUT, OUT, OUT
    d = os.path.join(OUT, "wnba")
    return d, d, d


def season_of_month(m):
    y, mo = int(m[:4]), int(m[5:])
    return y + (1 if mo >= 10 else 0) if LEAGUE == "nba" else y


def load_all(seasons):
    seg_dir, state_dir, _ = dirs()
    if LEAGUE == "nba":
        P = load_segments(seasons)
    else:
        fr = [pd.read_parquet(os.path.join(seg_dir, f"segments_{S}.parquet")) for S in seasons if os.path.exists(os.path.join(seg_dir, f"segments_{S}.parquet"))]
        P = pd.concat(fr, ignore_index=True)
        P = P[P["season_type"].isin({"regular-season", "playoffs"})].copy()
        P["off_lineup"], P["def_lineup"] = P["off_lineup"].map(tuple), P["def_lineup"].map(tuple)
        P["game_date"] = pd.to_datetime(P["game_date"])
        P = P.sort_values(["game_date", "game_id", "possession_seq"], kind="stable").reset_index(drop=True)
    fr = []
    for S in seasons:
        f1, f2 = os.path.join(seg_dir, f"slow_{S}.parquet"), os.path.join(state_dir, f"state_{S}.parquet")
        if os.path.exists(f1) and os.path.exists(f2):
            fr.append(pd.read_parquet(f1).merge(pd.read_parquet(f2), on=["game_id", "possession_seq"], how="inner"))
    V = pd.concat(fr, ignore_index=True)
    P = P.merge(V, on=["game_id", "possession_seq"], how="left")
    missing = int(P["def_pf_max"].isna().sum())
    for c in STATE_COLS:
        P[c] = P[c].fillna(0 if c != "secs_since_prev" else -1)
    return P, missing


def permute_blocks(blocks, names, P, rng):
    cells = state_cells(P).to_numpy()
    perm = np.arange(len(P))
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        perm[idx] = rng.permutation(idx)
    out = dict(blocks)
    for b in names:
        out[b] = blocks[b][perm]
    return out


def months_from_gbm():
    return sorted(os.path.basename(f)[4:11] for f in glob.glob(os.path.join(HIST_OUT, "bits", "gbm_*.parquet")))


def run_month(P, m, family, rng):
    te = P[(P["month"] == m) & (P["season"] >= 2000)]
    if not len(te):
        return None
    m_start = te["game_date"].min().normalize()
    tr = P[(P["game_date"] < m_start) & (P["game_date"] >= m_start - pd.Timedelta(days=TRAIN_DAYS))]
    if len(tr) < 20000:
        return None
    t0 = time.time()
    d = EventDesign(tr)
    ytr, yte = event_classes(tr), event_classes(te)
    btr, bte = d.all_blocks(tr), d.all_blocks(te)
    btr.update(state_blocks(tr, ytr))
    bte.update(state_blocks(te, yte))
    sets = sets_for(family)
    out = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "possession_seq": te["possession_seq"].to_numpy(), "season": te["season"].to_numpy(),
                        "month": m, "y": yte, "period": te["period"].to_numpy(), "off_margin_start": te["off_margin_start"].to_numpy(),
                        "start_type": te["start_type"].to_numpy(), "def_in_bonus": te["def_in_bonus"].to_numpy()})
    freq = np.bincount(ytr, minlength=8) / len(ytr)
    out["bits_NULL"] = -np.log2(freq[yte])
    conv, secs = {}, {}
    for name, blocks in sets.items():
        t1 = time.time()
        p, c = fit_predict(family, d.select(btr, blocks), ytr, d.select(bte, blocks))
        out[f"bits_{name}"] = bits_per_event(p, yte)
        conv[name], secs[name] = c, round(time.time() - t1)
    if family == "mnl":
        s6 = sets["S6"]
        ptr, pte = permute_blocks(btr, STATE_BLOCKS, tr, rng), permute_blocks(bte, STATE_BLOCKS, te, rng)
        p, c = fit_predict(family, d.select(ptr, s6), ytr, d.select(pte, s6))
        out["bits_S6_permstate"] = bits_per_event(p, yte)
        conv["S6_permstate"] = c
        ptr, pte = permute_blocks(btr, HIST, tr, rng), permute_blocks(bte, HIST, te, rng)
        p, c = fit_predict(family, d.select(ptr, sets["S6_H"]), ytr, d.select(pte, sets["S6_H"]))
        out["bits_S6_permhist"] = bits_per_event(p, yte)
        conv["S6_permhist"] = c
    ncols = {name: int(d.select(btr, blocks).shape[1]) for name, blocks in sets.items()}
    meta = {"month": m, "family": family, "n_train": int(len(tr)), "n_test": int(len(te)), "train_from": str(tr["game_date"].min().date()),
            "train_to": str(tr["game_date"].max().date()), "n_players_vocab": len(d.players), "n_teams": len(d.teams), "columns": ncols,
            "converged": conv, "fit_seconds": secs, "seconds": round(time.time() - t0, 1)}
    return out, meta


def main():
    global LEAGUE
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mnl", "gbm"], default="mnl")
    ap.add_argument("--months", nargs="+", required=True, help="YYYY-MM test months (contiguous for memory)")
    ap.add_argument("--league", choices=["nba", "wnba"], default="nba")
    a = ap.parse_args()
    LEAGUE = a.league
    _, _, out_dir = dirs()
    bits_dir = os.path.join(out_dir, "bits")
    os.makedirs(bits_dir, exist_ok=True)
    seasons_needed = sorted({season_of_month(m) - k for m in a.months for k in range(4)} & set(range(1997, 2021)))
    P, missing = load_all(seasons_needed)
    rng = np.random.default_rng(12345)
    log = open(os.path.join(out_dir, f"run_{a.family}_{a.months[0]}_{a.months[-1]}.log"), "a")
    print(f"loaded seasons {seasons_needed}: {len(P)} segments, {missing} without state variables", file=log, flush=True)
    for m in a.months:
        f = os.path.join(bits_dir, f"{a.family}_{m}.parquet")
        if os.path.exists(f):
            continue
        r = run_month(P, m, a.family, rng)
        if r is None:
            print(m, "skipped", file=log, flush=True)
            continue
        out, meta = r
        out.to_parquet(f, index=False)
        json.dump(meta, open(os.path.join(bits_dir, f"{a.family}_{m}.json"), "w"), indent=1)
        msg = f"{m} {a.family} n_train={meta['n_train']} n_test={meta['n_test']} " + \
              " ".join(f"{k}={out[f'bits_{k}'].mean():.4f}" for k in ("S0", "S0_H", "S1", "S6", "S6_H", "S7", "S7_H")) + f" {meta['seconds']}s"
        print(msg, file=log, flush=True)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
