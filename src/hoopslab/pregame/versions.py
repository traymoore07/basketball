"""X1 CONTEXT-SHADOW model-version registry (shadow_context/VERSIONS.json).

A version is forecastable only when its status is FROZEN, its model card exists, the card's bytes hash to the
registered card_sha256, and (for versions whose mathematics live in this repository) the implementation
hash recomputed from the files equals the card. PLANNED versions are refused. Nothing here changes a card.
"""
from __future__ import annotations

import hashlib
import json
import os

PLAN_PATH = os.path.join("shadow_context", "VERSIONS.json")


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_plan(root: str | None = None) -> dict:
    root = root or repo_root()
    return json.load(open(os.path.join(root, PLAN_PATH), encoding="utf-8"))


def get_version(model_version: str, root: str | None = None) -> dict:
    for v in load_plan(root)["versions"]:
        if v["model_version"] == model_version:
            return v
    raise KeyError(f"unknown model version {model_version}")


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def card_sha256(card: dict) -> str:
    """The card's self-hash as hoopslab.shadow.card writes it: canonical JSON of every field but card_sha256."""
    return hashlib.sha256(json.dumps({k: v for k, v in card.items() if k != "card_sha256"}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def verify_version(model_version: str, root: str | None = None) -> tuple[bool, list[str]]:
    """(forecastable, problems). Frozen cards must match byte for byte; planned versions are never forecastable."""
    root = root or repo_root()
    v = get_version(model_version, root)
    problems = []
    if v["status"] != "FROZEN":
        problems.append(f"{model_version} is {v['status']}, not FROZEN: no forecast may be issued")
        return False, problems
    card = os.path.join(root, v["card"])
    if not os.path.exists(card):
        problems.append(f"card missing: {v['card']}")
        return False, problems
    c = json.load(open(card, encoding="utf-8"))
    if card_sha256(c) != v["card_sha256"] or c.get("card_sha256") != v["card_sha256"]:
        problems.append("card content does not match the registered card_sha256")
    if c.get("model_version") != model_version:
        problems.append("card names a different model_version")
    if v.get("implementation_combined"):
        if c.get("implementation_hash", {}).get("combined") != v["implementation_combined"]:
            problems.append("card implementation hash differs from the registry")
        if model_version == "L2-A-SHADOW-v0.1":
            from ..shadow.model import model_implementation_hash
            if model_implementation_hash()["combined"] != v["implementation_combined"]:
                problems.append("frozen baseline code changed: recomputed implementation hash differs")
    return not problems, problems


def forecastable_versions(root: str | None = None) -> list[str]:
    return [v["model_version"] for v in load_plan(root)["versions"] if verify_version(v["model_version"], root)[0]]
