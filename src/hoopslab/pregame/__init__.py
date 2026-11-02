"""Lanes L1 (PREGAME-CONTEXT-LIVE) and X1 (CONTEXT-SHADOW): the provider-agnostic pregame-context input
contract with its validator, the one-way mapping to the frozen L2-A-SHADOW-v0.1 ELIGIBLE table, the
minutes-distribution output contract, the model-version registry and the executor CLI skeleton.

Research Claude may create and test this package on fixtures; it may not operate it on current-season
data (docs/lanes/00_lane_registry.md). The CLI refuses to forecast or report unless HOOPSLAB_LANE=live and
CONTEXT_HOME lies outside the repository.
"""
