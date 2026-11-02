"""
Toy numerical demonstrations supporting docs/research/nba_sequence_modeling.md.

These are NOT fits to real NBA data. Parameters are rough, round-number
approximations chosen to be in a realistic range. The point of each demo is a
qualitative/structural claim that holds across a wide range of parameters:

  A. Per-possession outcomes are high-entropy; team quality carries very few
     bits per possession, while possession *context* carries more.
  B. Individually calibrated per-event probabilities can still produce badly
     under-dispersed game totals if a game-level latent effect is omitted,
     and per-event log loss barely notices.
  C. Monte Carlo sample size is almost never the bottleneck; parameter
     uncertainty is.
  D. Plug-in conditional entropy with many conditioning cells manufactures
     "predictability" out of pure noise; held-out cross-entropy does not.
  E. A game-level latent shooting effect produces an apparent "hot hand"
     (P(make|make) > P(make|miss)) with zero causal dependence between shots,
     and the naive within-game estimator is biased the other way
     (Miller & Sanjurjo 2018).

Run:  python3 experiments/toy_demos.py
"""

import numpy as np

rng = np.random.default_rng(20260926)


def H(p):
    """Shannon entropy in bits of a pmf (ignores zeros)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tilt(base, values, target_mean):
    """Exponentially tilt pmf `base` over `values` so its mean equals target."""
    lo, hi = -5.0, 5.0
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        w = base * np.exp(lam * values)
        w /= w.sum()
        if (w * values).sum() < target_mean:
            lo = lam
        else:
            hi = lam
    return w


def demo_a():
    print("=" * 72)
    print("A. How many bits are in a possession, and how many does 'team' carry?")
    vals = np.array([0, 1, 2, 3, 4], dtype=float)
    # Rough league-wide pmf of points scored on a possession (~1.13 ppp).
    base = np.array([0.50, 0.03, 0.31, 0.15, 0.01])
    base /= base.sum()
    mean = (base * vals).sum()
    print(f"  league pmf {np.round(base, 3)}, mean {mean:.3f} ppp, "
          f"H = {H(base):.3f} bits/possession")

    # Team offensive quality: SD of team ORtg ~ 3.5 pts/100 -> 0.035 ppp.
    # Matchup (off - def) spread roughly sqrt(2) wider.
    for label, sd in [("offense only (sd 0.035 ppp)", 0.035),
                      ("off+def matchup (sd 0.05 ppp)", 0.05)]:
        shifts = rng.normal(0, sd, 20000)
        cond_H = np.mean([H(tilt(base, vals, mean + s)) for s in shifts[:2000]])
        # marginal over teams
        marg = np.mean([tilt(base, vals, mean + s) for s in shifts[:2000]], axis=0)
        mi = H(marg) - cond_H
        print(f"  I(outcome; {label:32s}) = {mi*1000:.2f} millibits/possession")

    # Possession start context: transition-ish starts (steal/live-ball TO) are
    # far more efficient than half-court starts. Rough mix/means.
    ctx = {"after made basket / dead ball": (0.55, 1.05),
           "after defensive rebound": (0.33, 1.12),
           "after live-ball turnover": (0.12, 1.30)}
    pmfs, ws = [], []
    for name, (w, m) in ctx.items():
        pmfs.append(tilt(base, vals, m))
        ws.append(w)
    ws = np.array(ws) / sum(ws)
    marg = (ws[:, None] * np.array(pmfs)).sum(0)
    mi = H(marg) - sum(w * H(p) for w, p in zip(ws, pmfs))
    print(f"  I(outcome; possession start type)   = {mi*1000:.2f} millibits/possession")
    print("  -> nearly all of the ~1.6 bits are irreducible pre-possession noise;")
    print("     model comparisons live in the 3rd-4th decimal place of bits/possession.")


def simulate_player_games(n_games, game_effects, rng):
    """Simulate a scorer's points. Returns (points, per-shot records)."""
    # Minutes ~ N(34, 4) clipped; team possessions/min ~ 100/48.
    mins = np.clip(rng.normal(34, 4, n_games), 10, 46)
    poss = rng.poisson(mins * 100 / 48)
    usage = 0.28
    # Shot mix among used possessions: 3PA, 2PA, FT trip (2 FTs)
    mix = np.array([0.35, 0.50, 0.15])
    logit_p = np.array([np.log(0.37 / 0.63), np.log(0.53 / 0.47), np.log(0.80 / 0.20)])
    if game_effects:
        u_eff = usage * np.exp(rng.normal(0, 0.15, n_games))
        s_eff = rng.normal(0, 0.20, n_games)  # game-level shooting logit shift
    else:
        u_eff = np.full(n_games, usage)
        s_eff = np.zeros(n_games)
    pts = np.zeros(n_games)
    shots = []  # (game, kind, made, true_p)
    for g in range(n_games):
        n_use = rng.binomial(poss[g], min(u_eff[g], 0.95))
        kinds = rng.choice(3, size=n_use, p=mix)
        for k in kinds:
            p = 1 / (1 + np.exp(-(logit_p[k] + s_eff[g])))
            if k == 2:
                makes = rng.binomial(2, p)
                pts[g] += makes
                shots.append((g, k, makes, p, 2))
            else:
                m = rng.random() < p
                pts[g] += (3 if k == 0 else 2) * m
                shots.append((g, k, int(m), p, 1))
    return pts, shots


def demo_b():
    print("=" * 72)
    print("B. Calibrated events, mis-calibrated totals (omitted game-level latent)")
    n = 20000
    truth_pts, truth_shots = simulate_player_games(n, True, rng)

    # Model A: same structure, no game effects, but per-event probabilities
    # set to the *marginal* make rates so each event is perfectly calibrated.
    kinds = np.array([s[1] for s in truth_shots])
    made = np.array([s[2] for s in truth_shots])
    trials = np.array([s[4] for s in truth_shots])
    true_p = np.array([s[3] for s in truth_shots])
    marg_p = np.array([made[kinds == k].sum() / trials[kinds == k].sum() for k in range(3)])

    # Per-attempt log loss: model A (marginal p) vs oracle (knows game effect).
    def ll(p, y, t):
        return -(y * np.log2(p) + (t - y) * np.log2(1 - p)) / t

    llA = ll(marg_p[kinds], made, trials).mean()
    llO = ll(true_p, made, trials).mean()
    print(f"  per-attempt log loss: model A {llA:.4f} bits, oracle {llO:.4f} bits, "
          f"gap {1000*(llA-llO):.2f} millibits  (upper bound on what B can gain)")

    # Model A's implied distribution of totals: simulate with no game effects.
    modelA_pts, _ = simulate_player_games(n, False, rng)
    print(f"  truth: mean {truth_pts.mean():.2f}, sd {truth_pts.std():.2f}")
    print(f"  model A: mean {modelA_pts.mean():.2f}, sd {modelA_pts.std():.2f}")
    lo, hi = np.quantile(modelA_pts, [0.10, 0.90])
    cover = np.mean((truth_pts >= lo) & (truth_pts <= hi))
    p30_t = np.mean(truth_pts >= 30)
    p30_a = np.mean(modelA_pts >= 30)
    p35_t = np.mean(truth_pts >= 35)
    p35_a = np.mean(modelA_pts >= 35)
    print(f"  model A nominal 80% interval [{lo:.0f}, {hi:.0f}] covers {100*cover:.1f}% of true games")
    print(f"  P(pts>=30): truth {p30_t:.3f} vs model A {p30_a:.3f};  "
          f"P(pts>=35): truth {p35_t:.3f} vs model A {p35_a:.3f}")


def demo_c():
    print("=" * 72)
    print("C. Monte Carlo error vs parameter uncertainty")
    for N in [1000, 10000, 100000]:
        print(f"  N={N:>6d} sims: SE of a probability near 0.3 = {np.sqrt(0.3*0.7/N):.4f}")
    # Parameter uncertainty: player's true mean points uncertain by sd ~1.5.
    # Approx game sd 7.5. P(>=30) under mean 25 +/- 1.5.
    from scipy.stats import norm
    for mu in [23.5, 25.0, 26.5]:
        print(f"  if true mean = {mu:4.1f}: P(pts>=29.5) = {1-norm.cdf(29.5, mu, 7.5):.3f}")
    print("  -> a 1-sd parameter error moves the tail probability ~7 points;")
    print("     10k sims already has MC error ~0.005. Sample parameters, not just games.")


def demo_d():
    print("=" * 72)
    print("D. Spurious entropy reduction from sparse conditioning")
    k_actions = 4
    true_H = np.log2(k_actions)
    n = 2000
    for n_cells in [1, 10, 100, 500]:
        a = rng.integers(0, k_actions, n)
        c = rng.integers(0, n_cells, n)
        # plug-in H(A|C) on the same data
        hc = 0.0
        for cell in range(n_cells):
            m = c == cell
            if m.sum() == 0:
                continue
            counts = np.bincount(a[m], minlength=k_actions)
            hc += m.mean() * H(counts / counts.sum())
        # held-out cross-entropy with add-1/2 smoothing, 2-fold
        idx = rng.permutation(n)
        tr, te = idx[: n // 2], idx[n // 2:]
        ce = 0.0
        for i in te:
            m = c[tr] == c[i]
            counts = np.bincount(a[tr][m], minlength=k_actions) + 0.5
            ce += -np.log2(counts[a[i]] / counts.sum())
        ce /= len(te)
        print(f"  {n_cells:4d} context cells: true H={true_H:.2f}, "
              f"plug-in H(A|C)={hc:.2f} (fake gain {true_H-hc:.2f} bits), "
              f"held-out CE={ce:.2f}")


def demo_e():
    print("=" * 72)
    print("E. Apparent hot hand from a game-level latent; Miller-Sanjurjo bias")
    n_games, shots_per_game = 20000, 20
    for label, sd in [("iid shots, p=0.5", 0.0), ("game-level latent sd=0.3 logit", 0.3)]:
        eff = rng.normal(0, sd, n_games)
        p = 1 / (1 + np.exp(-eff))
        x = rng.random((n_games, shots_per_game)) < p[:, None]
        prev, nxt = x[:, :-1], x[:, 1:]
        pooled = nxt[prev].mean() - nxt[~prev].mean()
        # naive per-game difference then averaged (the GVT-style estimator)
        diffs = []
        for g in range(n_games):
            pm, nm = prev[g], nxt[g]
            if pm.sum() > 0 and (~pm).sum() > 0:
                diffs.append(nm[pm].mean() - nm[~pm].mean())
        print(f"  {label:32s}: pooled P(make|make)-P(make|miss) = {pooled:+.4f}; "
              f"per-game average = {np.mean(diffs):+.4f}")
    print("  -> pooled estimates confound between-game heterogeneity with momentum;")
    print("     per-game averages are biased negative even for iid shots.")


if __name__ == "__main__":
    demo_a()
    demo_b()
    demo_c()
    demo_d()
    demo_e()
