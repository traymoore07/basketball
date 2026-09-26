"""Pre-registration registry: load, validate, render, freeze, verify.

    python -m hoopslab.registry validate
    python -m hoopslab.registry render      # regenerate docs/prereg/02-04 tables from YAML
    python -m hoopslab.registry freeze      # write registry/FREEZE.json (hashes of frozen files)
    python -m hoopslab.registry verify      # fail if any frozen file changed without an amendment

Amendments: an edit to a frozen file after the freeze must be recorded in
registry/AMENDMENTS.md with date, reason, and whether real data had been seen
(POST-HOC amendments are allowed but labelled as such in every report). After
recording it, run `freeze --amend "<reason>"` to append a new manifest entry.
The original manifest is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "registry"
FROZEN_FILES = [
    # canonical registry
    "registry/hypotheses.yaml",
    "registry/baseline_ladder.yaml",
    "registry/expectations.yaml",
    "src/hoopslab/contract/data_contract.yaml",
    # blueprint documents
    "docs/prereg/01_experimental_blueprint.md",
    "docs/prereg/02_hypothesis_registry.md",
    "docs/prereg/03_baseline_ladder.md",
    "docs/prereg/04_preregistered_expectations.md",
    "docs/prereg/05_data_contract.md",
    "docs/prereg/06_leakage_checklist.md",
    "docs/prereg/07_temporal_backtest_spec.md",
    "docs/prereg/08_metric_spec.md",
    "docs/prereg/09_synthetic_test_suite.md",
    "docs/prereg/10_cross_league.md",
    "docs/prereg/11_real_data_checklist.md",
    # code that encodes pre-registered decisions (metrics, decision rule, targets, strata,
    # tolerances, diagnosis rules, expected synthetic diagnoses)
    "src/hoopslab/metrics/scoring.py",
    "src/hoopslab/metrics/compare.py",
    "src/hoopslab/metrics/information.py",
    "src/hoopslab/targets.py",
    "src/hoopslab/backtest/strata.py",
    "src/hoopslab/backtest/leakage.py",
    "src/hoopslab/diagnostics/realism.py",
    "src/hoopslab/diagnostics/changepoint.py",
    "src/hoopslab/experiments/history.py",
    "src/hoopslab/synthetic/scenarios.py",
]
REQUIRED_H = ["id", "family", "title", "claim"]
CONFIRMATORY_REQUIRED = ["null_hypothesis", "alternative", "models", "primary_metric", "mpe", "falsified_if"]


def load(name: str) -> dict:
    return yaml.safe_load((REG / name).read_text())


def validate() -> list[str]:
    errs = []
    H = load("hypotheses.yaml")
    ladder = load("baseline_ladder.yaml")
    ids = set()
    rung_ids = {r["id"] for r in ladder["rungs"]} | {o["id"] for o in ladder["oracles"]}
    for h in H["hypotheses"]:
        for k in REQUIRED_H:
            if k not in h:
                errs.append(f"{h.get('id')}: missing {k}")
        if h.get("family") in ("primary", "secondary"):
            for k in CONFIRMATORY_REQUIRED:
                if k not in h:
                    errs.append(f"{h['id']}: confirmatory hypothesis missing {k}")
        if h["id"] in ids:
            errs.append(f"duplicate id {h['id']}")
        ids.add(h["id"])
        m = h.get("models", {})
        for role in ("reference", "candidate"):
            v = str(m.get(role, ""))
            if v in {"L" + str(i) for i in range(20)} and v not in rung_ids:
                errs.append(f"{h['id']}: unknown rung {v}")
    E = load("expectations.yaml")
    exp_ids = [e["id"].split("-")[0] for e in E["hypotheses"]]
    for h in H["hypotheses"]:
        if h.get("family") == "primary" and not any(x == h["id"] or x.startswith(h["id"]) for x in exp_ids):
            errs.append(f"{h['id']}: primary hypothesis has no pre-registered expectation")
    for e in E["hypotheses"] + E["ladder_gaps"]:
        if "strength" not in e:
            errs.append(f"expectation {e['id']} missing strength")
        elif e["strength"] not in ("strong", "weak", "speculation"):
            errs.append(f"expectation {e['id']} bad strength {e['strength']}")
    return errs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    return {f: _sha(ROOT / f) for f in FROZEN_FILES if (ROOT / f).exists()}


def manifest_hash(m: dict | None = None) -> str:
    m = m or manifest()
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def freeze(amend: str | None = None) -> dict:
    path = REG / "FREEZE.json"
    history = json.loads(path.read_text()) if path.exists() else {"entries": []}
    if history["entries"] and amend is None:
        raise SystemExit("already frozen; use --amend with a reason recorded in registry/AMENDMENTS.md")
    m = manifest()
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": m,
             "manifest_hash": manifest_hash(m), "amendment": amend,
             "real_data_seen": False if amend is None else "see AMENDMENTS.md"}
    history["entries"].append(entry)
    path.write_text(json.dumps(history, indent=2) + "\n")
    return entry


def verify() -> tuple[bool, list[str]]:
    path = REG / "FREEZE.json"
    if not path.exists():
        return False, ["registry/FREEZE.json missing"]
    last = json.loads(path.read_text())["entries"][-1]
    cur = manifest()
    diffs = [f for f in set(cur) | set(last["files"]) if cur.get(f) != last["files"].get(f)]
    return not diffs, sorted(diffs)


def render() -> str:
    H = load("hypotheses.yaml")
    lines = ["# Hypothesis registry (rendered)", "",
             "*Generated from `registry/hypotheses.yaml` by `python -m hoopslab.registry render`. "
             "The YAML is canonical; do not edit this file by hand.*", "",
             "| ID | Family | Title | Reference → Candidate | Primary metric | MPE | Falsified if |",
             "|---|---|---|---|---|---|---|"]
    for h in H["hypotheses"]:
        m = h.get("models", {})
        rc = f"{m.get('reference', '')} → {m.get('candidate', '')}" if m else ""
        mpe = h.get("mpe", {})
        mp = f"{mpe.get('value')} {mpe.get('units', '')}" if mpe else ""
        pm = h.get("primary_metric", "")
        pm = ", ".join(pm) if isinstance(pm, list) else pm
        cells = [h["id"], h["family"], h["title"], rc, pm, mp, str(h.get("falsified_if", ""))]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines += ["", "## Full entries", ""]
    for h in H["hypotheses"]:
        lines.append(f"### {h['id']}: {h['title']}")
        lines.append("")
        for k, v in h.items():
            if k in ("id", "title"):
                continue
            if isinstance(v, (dict, list)):
                v = "`" + json.dumps(v, ensure_ascii=False) + "`"
            lines.append(f"- **{k}**: {str(v).strip()}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "validate"
    if cmd == "validate":
        e = validate()
        print("OK" if not e else "\n".join(e))
        return 0 if not e else 1
    if cmd == "render":
        out = ROOT / "docs/prereg/02_hypothesis_registry.md"
        out.write_text(render())
        print(f"wrote {out}")
        return 0
    if cmd == "freeze":
        amend = argv[2] if len(argv) > 2 and argv[1] == "--amend" else None
        e = freeze(amend)
        print(json.dumps({"manifest_hash": e["manifest_hash"], "files": len(e["files"])}, indent=2))
        return 0
    if cmd == "verify":
        ok, diffs = verify()
        print("FROZEN FILES UNCHANGED" if ok else "CHANGED SINCE FREEZE:\n" + "\n".join(diffs))
        return 0 if ok else 1
    raise SystemExit(f"unknown command {cmd}")
