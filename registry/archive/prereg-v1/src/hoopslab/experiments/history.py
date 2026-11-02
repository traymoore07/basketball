"""E-HIST: how much does within-game history matter beyond the state? (H2/H3)

Nested feature sets, one model family, out-of-sample bits per event:
    A  S                         compact state only
    B  S + L1                    + previous possession
    C  S + L1 + L3               + previous 3
    D  S + L1 + L3 + L10         + previous 10
    E  D + EW                    + whole-game summaries (sequence-model stand-in)
    F  S + GL                    + game-level latent estimate only
    G  F + L1 + L3 + L10 + EW    history beyond the latent

Pre-registered contrasts:
    history_signal   = bits(A) - bits(E)   (does history carry usable information?)
    latent_signal    = bits(A) - bits(F)
    history_residual = bits(F) - bits(G)   (information in history NOT explained by the latent)

Diagnosis rule (MPE = 0.5 millibits per event):
    no history signal        : history_signal CI upper < MPE
    genuine history          : history_residual CI lower > 0 and estimate >= MPE
    apparent (latent) history: history_signal >= MPE, latent_signal >= MPE, history_residual CI upper < MPE
    otherwise                : inconclusive
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.information import bits_per_event, usable_information
from ..models.event import EventDesign, MultinomialLogit, event_classes

SETS = {
    "A": ["S"], "B": ["S", "L1"], "C": ["S", "L1", "L3"], "D": ["S", "L1", "L3", "L10"],
    "E": ["S", "L1", "L3", "L10", "EW"], "F": ["S", "GL"], "G": ["S", "GL", "L1", "L3", "L10", "EW"],
}
MPE_BITS = 0.0005


def run_history_ablation(P_train: pd.DataFrame, P_test: pd.DataFrame, sets: dict | None = None,
                         l2: float = 1e-4, B: int = 500, seed: int = 0) -> dict:
    sets = sets or SETS
    design = EventDesign(P_train)
    ytr, yte = event_classes(P_train), event_classes(P_test)
    btr, bte = design.all_blocks(P_train), design.all_blocks(P_test)
    bits = {}
    for name, blocks in sets.items():
        m = MultinomialLogit(l2=l2).fit(design.select(btr, blocks), ytr)
        bits[name] = bits_per_event(m.predict_proba(design.select(bte, blocks)), yte)
    cl = P_test["game_id"].to_numpy()
    rng = np.random.default_rng(seed)
    res = {"mean_bits": {k: float(v.mean()) for k, v in bits.items()}, "n_test": int(len(yte))}
    res["history_signal"] = usable_information(bits["A"], bits["E"], cl, B=B, rng=rng)
    res["latent_signal"] = usable_information(bits["A"], bits["F"], cl, B=B, rng=rng)
    res["history_residual"] = usable_information(bits["F"], bits["G"], cl, B=B, rng=rng)
    for k in ("B", "C", "D"):
        res[f"gain_{k}"] = usable_information(bits["A"], bits[k], cl, B=B, rng=rng)
    res["diagnosis"] = diagnose_history(res)
    res["_bits"] = bits
    return res


def diagnose_history(res: dict, mpe: float = MPE_BITS) -> str:
    hs, ls, hr = res["history_signal"], res["latent_signal"], res["history_residual"]
    if hs["ci_high"] < mpe:
        return "NO_HISTORY_SIGNAL"
    if hr["ci_low"] > 0 and hr["bits"] >= mpe:
        return "GENUINE_HISTORY"
    if hs["bits"] >= mpe and ls["bits"] >= mpe and hr["ci_high"] < mpe:
        return "APPARENT_HISTORY_EXPLAINED_BY_LATENT"
    return "INCONCLUSIVE"
