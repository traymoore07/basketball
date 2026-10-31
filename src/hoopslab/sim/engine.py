"""Vectorised possession-level basketball engine.

The same engine serves two roles:
  1. the GROUND-TRUTH generator of synthetic worlds (rows = real synthetic games);
  2. the reference STRUCTURAL SIMULATOR (rows = Monte Carlo replicates of one game,
     driven by parameters estimated as-of the forecast time).

That the truth and the reference simulator share structure is deliberate: the
synthetic suite verifies instruments, not models. Misspecification is introduced
explicitly (momentum the simulator does not model, omitted game latents, wrong pace,
disabled blowout rule) so the instruments can be tested for detecting it.

The model is simplified on purpose:
  * slot rotation (each of 5 slots has a starter and a bench player; the starter
    plays the first and last `frac/2` of each quarter; overtime: starters);
  * blowout rule (bench plays late when the margin is large);
  * possession user chosen by softmax over on-court usage weights;
  * a terminal event per segment: turnover / 2-shot FT trip / 2PA / 3PA;
  * offensive rebounds continue the possession;
  * assists, steals, blocks, shooting fouls;
  * start-type effects (transition after steals, put-backs after offensive rebounds);
  * optional per-game latents and a true "momentum" effect (history dependence).

Foul-outs, timeouts, end-of-game intentional fouling and period-start possession
rules are not modelled. They are listed as known gaps in the synthetic spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATS = ["SEC", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
         "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
SI = {s: i for i, s in enumerate(STATS)}

START_TYPES = ["after_make", "after_dreb", "after_live_turnover", "after_oreb"]
EVENT_CODES = ["TO", "FT", "2PA", "3PA"]

REG_LEN = 2880.0
QTR = 720.0
OT_LEN = 300.0


@dataclass
class GameBatch:
    """Parameters for S rows (games or replicates). Team axis: 0 = home, 1 = away.

    Player arrays have shape (S, 2, R), indexed by roster position.
    """
    usage: np.ndarray
    to_rate: np.ndarray
    ft_rate: np.ndarray
    share3: np.ndarray
    logit2: np.ndarray
    logit3: np.ndarray
    pft: np.ndarray
    orb_w: np.ndarray
    drb_w: np.ndarray
    ast_w: np.ndarray
    stl_w: np.ndarray
    blk_w: np.ndarray
    starter_idx: np.ndarray      # (S, 2, 5) roster positions of slot starters
    bench_idx: np.ndarray        # (S, 2, 5)
    starter_frac: np.ndarray     # (S, 2, 5) share of regulation slot time for the starter
    def_eff: np.ndarray          # (S, 2) logit shift applied to OPPONENT shots
    shoot_latent: np.ndarray     # (S, 2) per-game offensive shooting logit shift
    usage_mult: np.ndarray       # (S, 2, R) per-game usage multipliers ("form")
    pace_mult: np.ndarray        # (S,)   per-game duration multiplier
    home_adv: float = 0.08
    momentum: float = 0.0        # logit bonus if this team scored on its previous possession
    transition_bonus: tuple = (0.0, 0.10, 0.45, 0.25)  # by START_TYPES
    assist_p: tuple = (0.55, 0.85)                    # P(assisted | made 2 / made 3)
    steal_share: float = 0.55
    block_p2: float = 0.10
    blowout_margin: float | None = 20.0
    blowout_time: float = 2520.0
    duration_ranges: dict = field(default_factory=lambda: {
        0: (8.0, 22.0), 1: (7.0, 20.0), 2: (3.0, 10.0), 3: (3.0, 12.0)})

    @property
    def S(self) -> int:
        return self.usage.shape[0]


@dataclass
class SimResult:
    box: np.ndarray                    # (S, 2, R, n_stats)
    score: np.ndarray                  # (S, 2)
    periods: np.ndarray                # (S,)
    lead_changes: np.ndarray           # (S,)
    possessions: np.ndarray            # (S, 2) offensive possessions (excl. OREB continuations)
    segments: np.ndarray               # (S, 2)
    events: pd.DataFrame | None = None


def _pick(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a column index per row with probability proportional to `weights` (n, k)."""
    c = np.cumsum(weights, axis=1)
    r = rng.random(len(weights)) * c[:, -1]
    return np.minimum((c < r[:, None]).sum(axis=1), weights.shape[1] - 1)


def _period_end(t: np.ndarray) -> np.ndarray:
    reg = (np.floor(t / QTR + 1e-9) + 1) * QTR
    ot = REG_LEN + (np.floor((t - REG_LEN) / OT_LEN + 1e-9) + 1) * OT_LEN
    return np.where(t < REG_LEN - 1e-9, reg, ot)


def simulate(b: GameBatch, rng: np.random.Generator, log_events: bool = False,
             max_steps: int = 2000) -> SimResult:
    S = b.S
    R = b.usage.shape[2]
    NS = len(STATS)
    box = np.zeros((S, 2, R, NS))
    flat_box = box.reshape(-1)  # view: scatter-adds write through to `box`
    t = np.zeros(S)
    score = np.zeros((S, 2))
    off = (rng.random(S) < 0.5).astype(int)
    start = np.zeros(S, dtype=int)
    prev_scored = np.zeros((S, 2))
    last_sign = np.zeros(S)
    lead_changes = np.zeros(S)
    poss = np.zeros((S, 2))
    segs = np.zeros((S, 2))
    active = np.ones(S, dtype=bool)
    usage_eff = b.usage * b.usage_mult
    logs = []
    ar5 = np.arange(5)

    for _ in range(max_steps):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        n = len(idx)
        ti = t[idx]
        o = off[idx]
        d = 1 - o
        st = start[idx]

        # ---------------- lineups ----------------
        tq = np.mod(ti, QTR)
        frac = b.starter_frac[idx]                                   # (n,2,5)
        half = frac * QTR / 2
        s_on = (tq[:, None, None] < half) | (tq[:, None, None] > QTR - half)
        s_on |= (ti >= REG_LEN - 1e-9)[:, None, None]               # overtime: starters
        if b.blowout_margin is not None:
            blow = (ti >= b.blowout_time) & (ti < REG_LEN) & \
                   (np.abs(score[idx, 0] - score[idx, 1]) >= b.blowout_margin)
            s_on &= ~blow[:, None, None]
        on = np.where(s_on, b.starter_idx[idx], b.bench_idx[idx])    # (n,2,5)
        rows = np.arange(n)
        off_line = on[rows, o]                                        # (n,5)
        def_line = on[rows, d]

        def P(arr, team, line):  # gather player params for a lineup -> (n,5)
            return arr[idx[:, None], team[:, None], line]

        # ---------------- user and terminal event ----------------
        j = _pick(P(usage_eff, o, off_line), rng)
        user = off_line[rows, j]

        def U(arr):
            return arr[idx, o, user]

        to_r = U(b.to_rate) * np.where(st == 2, 0.8, 1.0)
        r = rng.random(n)
        is_to = r < to_r
        is_ft = (~is_to) & (r < to_r + U(b.ft_rate))
        is_shot = ~(is_to | is_ft)
        is3 = is_shot & (rng.random(n) < U(b.share3))
        tb = np.asarray(b.transition_bonus)[st]
        lg = np.where(is3, U(b.logit3), U(b.logit2)) + b.def_eff[idx, d] + b.shoot_latent[idx, o] \
            + b.home_adv * (o == 0) + tb + b.momentum * prev_scored[idx, o]
        made = is_shot & (rng.random(n) < 1.0 / (1.0 + np.exp(-lg)))
        pft = U(b.pft)
        ftm = is_ft * ((rng.random(n) < pft).astype(int) + (rng.random(n) < pft).astype(int))
        pts = made * (2 + is3) + ftm

        # ---------------- secondary actors ----------------
        missed = is_shot & ~made
        blk = missed & ~is3 & (rng.random(n) < b.block_p2)
        A = P(b.orb_w, o, off_line).sum(1)
        D = P(b.drb_w, d, def_line).sum(1)
        orb = missed & (rng.random(n) < A / (A + D))
        drb = missed & ~orb
        reb_off = off_line[rows, _pick(P(b.orb_w, o, off_line), rng)]
        reb_def = def_line[rows, _pick(P(b.drb_w, d, def_line), rng)]
        ap = np.where(is3, b.assist_p[1], b.assist_p[0])
        assisted = made & (rng.random(n) < ap)
        aw = P(b.ast_w, o, off_line).copy()
        aw[rows, j] = 0.0
        assister = off_line[rows, _pick(aw + 1e-12, rng)]
        steal = is_to & (rng.random(n) < b.steal_share)
        stealer = def_line[rows, _pick(P(b.stl_w, d, def_line), rng)]
        blocker = def_line[rows, _pick(P(b.blk_w, d, def_line), rng)]
        fouler = def_line[rows, rng.integers(0, 5, n)]

        # ---------------- clock ----------------
        lo_hi = np.array([b.duration_ranges[k] for k in range(4)])
        lo, hi = lo_hi[st, 0], lo_hi[st, 1]
        dur = (lo + (hi - lo) * rng.random(n)) * b.pace_mult[idx]
        dur = np.minimum(dur, _period_end(ti) - ti)

        # ---------------- box score accumulation (one scatter-add per step) ----------------
        fi, fv = [], []

        def add(team, player, stat, val, mask=None):
            val = np.broadcast_to(np.asarray(val, float), (n,))
            if mask is not None:
                sel = np.nonzero(mask)[0]
                if len(sel) == 0:
                    return
                rr, tt, pp, vv = idx[sel], team[sel], player[sel], val[sel]
            else:
                rr, tt, pp, vv = idx, team, player, val
            fi.append(((rr * 2 + tt) * R + pp) * NS + SI[stat])
            fv.append(vv)

        for k in range(5):
            add(o, off_line[:, k], "SEC", dur)
            add(d, def_line[:, k], "SEC", dur)
        add(o, user, "PTS", pts)
        add(o, user, "FGA", 1.0, is_shot)
        add(o, user, "FGM", 1.0, made)
        add(o, user, "FG3A", 1.0, is3)
        add(o, user, "FG3M", 1.0, made & is3)
        add(o, user, "FTA", 2.0, is_ft)
        add(o, user, "FTM", ftm, is_ft)
        add(o, user, "TOV", 1.0, is_to)
        add(o, reb_off, "OREB", 1.0, orb)
        add(d, reb_def, "DREB", 1.0, drb)
        add(o, assister, "AST", 1.0, assisted)
        add(d, stealer, "STL", 1.0, steal)
        add(d, blocker, "BLK", 1.0, blk)
        add(d, fouler, "PF", 1.0, is_ft)
        np.add.at(flat_box, np.concatenate(fi), np.concatenate(fv))

        if log_events:
            logs.append(pd.DataFrame({
                "row": idx, "t": ti, "dur": dur, "off": o, "start": st, "user": user,
                "event": np.select([is_to, is_ft, is3], [0, 1, 3], 2), "made": made.astype(int),
                "ftm": ftm, "pts": pts, "orb": orb.astype(int), "drb": drb.astype(int),
                "steal": steal.astype(int), "assisted": assisted.astype(int),
                "stealer": np.where(steal, stealer, -1), "blocked": blk.astype(int),
                "blocker": np.where(blk, blocker, -1), "fouler": np.where(is_ft, fouler, -1),
                "assister": np.where(assisted, assister, -1),
                "rebounder": np.where(orb, reb_off, np.where(drb, reb_def, -1)),
                "margin": score[idx, o] - score[idx, d], "prev_scored": prev_scored[idx, o],
                **{f"ol{k}": off_line[:, k] for k in range(5)},
                **{f"dl{k}": def_line[:, k] for k in range(5)},
            }))

        # ---------------- state update ----------------
        segs[idx, o] += 1
        poss[idx, o] += (st != 3)
        score[idx, o] += pts
        t[idx] = ti + dur
        # previous-possession memory only updates when the possession ends
        # (points only occur on a possession's final segment)
        ends = ~orb
        prev_scored[idx[ends], o[ends]] = (pts[ends] > 0)
        new_start = np.select([orb, steal, drb], [3, 2, 1], 0)
        start[idx] = new_start
        off[idx] = np.where(orb, o, d)
        m = score[idx, 0] - score[idx, 1]
        sgn = np.sign(m)
        lc = (sgn != 0) & (last_sign[idx] != 0) & (sgn != last_sign[idx])
        lead_changes[idx] += lc
        last_sign[idx] = np.where(sgn != 0, sgn, last_sign[idx])
        tn = t[idx]
        at_boundary = np.isclose(tn, _period_end(tn - 1e-6))
        finished = (tn >= REG_LEN - 1e-6) & at_boundary & (m != 0)
        active[idx[finished]] = False

    periods = np.where(t <= REG_LEN + 1e-6, 4, 4 + np.ceil((t - REG_LEN - 1e-6) / OT_LEN)).astype(int)
    ev = pd.concat(logs, ignore_index=True) if log_events and logs else None
    return SimResult(box=box, score=score, periods=periods, lead_changes=lead_changes,
                     possessions=poss, segments=segs, events=ev)
