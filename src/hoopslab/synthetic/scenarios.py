"""Named synthetic scenarios with known ground truth (docs/prereg/09_synthetic_test_suite.md).

Each scenario isolates one mechanism. The expected diagnosis is fixed here, before the
instruments are run. Evaluation happens in the LAST season (index n_seasons - 1).
Earlier seasons are history.
"""

from __future__ import annotations

from .world import ScenarioConfig

EVAL_SEASON = 2


def scenario(name: str, seed: int = 0) -> ScenarioConfig:
    s = EVAL_SEASON
    table = {
        # A: history truly does not matter (Markov in the compact state; no latents)
        "A_no_history": ScenarioConfig(name="A_no_history", seed=seed),
        # B: genuine history dependence the compact state does not contain
        "B_true_history": ScenarioConfig(name="B_true_history", seed=seed, momentum=0.45),
        # C: a hidden game-level shooting variable creates apparent momentum
        "C_latent_momentum": ScenarioConfig(name="C_latent_momentum", seed=seed, game_latent_sd=0.55),
        # D: sudden role changes (usage x2.5) for several bench players mid-season
        "D_role_change": ScenarioConfig(name="D_role_change", seed=seed, role_changes=[
            {"season": s, "day": 8, "team": t, "pos": p, "usage_mult": 2.5}
            for t, p in [(0, 6), (1, 7), (2, 5), (3, 8), (4, 6), (5, 7)]]),
        # E: star teammates disappear for long stretches
        "E_star_absence": ScenarioConfig(name="E_star_absence", seed=seed, star_absences=[
            {"season": s, "day": 6, "team": t, "n_games": 14} for t in (0, 1, 2, 3)]),
        # F: trades change the surrounding lineup AND the traded players' roles
        #    (a starter moves to a deeper team's bench; a bench player becomes a starter)
        "F_trade": ScenarioConfig(name="F_trade", seed=seed, trades=[
            {"season": s, "day": 4, "team_a": a, "pos_a": 1, "team_b": b, "pos_b": 6}
            for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]]),
        # G: parameter uncertainty matters: half of every roster is new each season
        "G_param_uncertainty": ScenarioConfig(name="G_param_uncertainty", seed=seed, rookie_frac=0.5),
        # H: event probabilities right, aggregate tails wrong unless a game latent is modelled
        "H_tail_latent": ScenarioConfig(name="H_tail_latent", seed=seed, game_latent_sd=0.40),
    }
    return table[name]


EXPECTED = {
    "A_no_history": {"history": "NO_HISTORY_SIGNAL"},
    "B_true_history": {"history": "GENUINE_HISTORY"},
    "C_latent_momentum": {"history": "APPARENT_HISTORY_EXPLAINED_BY_LATENT"},
    "D_role_change": {"stratified": "ADVANTAGE_LOCALISED_IN_STRATUM", "changepoint": "DETECTED"},
    "E_star_absence": {"stratified": "ADVANTAGE_LOCALISED_IN_STRATUM"},
    "F_trade": {"stratified": "ADVANTAGE_LOCALISED_IN_STRATUM"},
    "G_param_uncertainty": {"param_uncertainty": "PLUGIN_OVERCONFIDENT_IN_LOW_SAMPLE_STRATUM"},
    "H_tail_latent": {"tails": "EVENT_OK_AGGREGATE_UNDERDISPERSED_WITHOUT_LATENT"},
}
