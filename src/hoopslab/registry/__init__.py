"""Pre-registration registry: load, validate, render, freeze, verify, amend.

    python -m hoopslab.registry validate
    python -m hoopslab.registry render                 # regenerate docs/prereg/02 from YAML
    python -m hoopslab.registry verify                 # all integrity checks (tiers 1 and 2, archive, logs)
    python -m hoopslab.registry begin-amendment A1     # snapshot frozen files BEFORE editing them
    python -m hoopslab.registry freeze --amend A1      # append a FREEZE entry for a logged amendment

Three layers of protection (docs/onboarding/00_onboarding_protocol.md):
  tier 1  files in registry/FREEZE.json. Change only via a logged amendment; entry 0 and
          registry/archive/prereg-v1/ (the original version) are immutable.
  tier 2  all other files committed at the freeze commit, hashed from its git objects in
          registry/FREEZE_SUPPLEMENT.json. A change needs a row in registry/POSTFREEZE_CHANGELOG.md;
          decision-bearing files only as bugfix / implementation-per-spec / amendment:A<n>.
  logs    AMENDMENTS.md, DATA_CONTRACT_MISMATCHES.md, EXPOSURE_LOG.md (append-only by convention).
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


ARCHIVE = REG / "archive"
SUPPLEMENT = REG / "FREEZE_SUPPLEMENT.json"
CHANGELOG = REG / "POSTFREEZE_CHANGELOG.md"
AMEND_LOG = REG / "AMENDMENTS.md"
AMEND_CLASSES = {"DATA-DRIVEN NECESSITY", "ERROR CORRECTION", "RESEARCH PREFERENCE"}
REPORT_LABELS = {"PRE-OUTCOME", "POST-HOC"}
CHANGE_CLASSES = {"engineering", "governance", "bugfix", "implementation-per-spec"}
DECISION_OK = {"bugfix", "implementation-per-spec"}


def _entries() -> list[dict]:
    return json.loads((REG / "FREEZE.json").read_text())["entries"]


def _table_rows(path: Path, id_prefix: str) -> list[list[str]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0].startswith(id_prefix) and cells[0][1:].isdigit():
            rows.append(cells)
    return rows


def amendment_rows() -> dict[str, list[str]]:
    return {r[0]: r for r in _table_rows(AMEND_LOG, "A")}


def amendment_complete(aid: str) -> tuple[bool, str]:
    r = amendment_rows().get(aid)
    if r is None:
        return False, f"{aid}: no row in AMENDMENTS.md"
    if len(r) < 10 or any(c in ("", "-", "–") for c in r[:10]):
        return False, f"{aid}: row incomplete (10 fields required)"
    if r[5] not in AMEND_CLASSES:
        return False, f"{aid}: class must be one of {sorted(AMEND_CLASSES)}"
    if r[9] not in REPORT_LABELS:
        return False, f"{aid}: report label must be PRE-OUTCOME or POST-HOC"
    return True, "ok"


def changelog_rows() -> list[list[str]]:
    return _table_rows(CHANGELOG, "C")


def freeze(amend: str | None = None) -> dict:
    path = REG / "FREEZE.json"
    history = json.loads(path.read_text()) if path.exists() else {"entries": []}
    if history["entries"] and amend is None:
        raise SystemExit("already frozen; amendments: begin-amendment A<n>, log it, then freeze --amend A<n>")
    if amend is not None:
        ok, msg = amendment_complete(amend)
        if not ok:
            raise SystemExit(f"refusing to freeze: {msg}")
        if not (ARCHIVE / f"pre-{amend}").exists():
            raise SystemExit(f"refusing to freeze: run begin-amendment {amend} before editing frozen files")
        files = sorted(set(history["entries"][-1]["files"]) | set(FROZEN_FILES))
    else:
        files = FROZEN_FILES
    m = {f: _sha(ROOT / f) for f in files if (ROOT / f).exists()}
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": m,
             "manifest_hash": manifest_hash(m), "amendment": amend,
             "real_data_seen": False if amend is None else "see registry/EXPOSURE_LOG.md"}
    history["entries"].append(entry)
    path.write_text(json.dumps(history, indent=2) + "\n")
    return entry


def begin_amendment(aid: str) -> Path:
    """Snapshot the current frozen files (which must be unmodified) before an amendment edits them."""
    last = _entries()[-1]["files"]
    changed = [f for f, h in last.items() if _sha(ROOT / f) != h]
    if changed:
        raise SystemExit(f"frozen files already modified before snapshot: {changed}")
    dest = ARCHIVE / f"pre-{aid}"
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    for f in last:
        out = dest / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((ROOT / f).read_bytes())
    return dest


def verify() -> tuple[bool, list[str]]:
    """Tier-1 check only: files in the latest FREEZE entry are byte-identical on disk."""
    path = REG / "FREEZE.json"
    if not path.exists():
        return False, ["registry/FREEZE.json missing"]
    last = _entries()[-1]["files"]
    diffs = [f for f, h in last.items() if not (ROOT / f).exists() or _sha(ROOT / f) != h]
    return not diffs, sorted(diffs)


def verify_all() -> tuple[bool, list[str]]:
    """Every integrity check. Returns (ok, list of problems)."""
    problems = []
    entries = _entries()
    # 1. every FREEZE entry's manifest hash reproduces (no entry was edited)
    for i, e in enumerate(entries):
        if manifest_hash(e["files"]) != e["manifest_hash"]:
            problems.append(f"FREEZE entry {i} was edited (manifest hash does not reproduce)")
    if entries and entries[0].get("amendment") is not None:
        problems.append("FREEZE entry 0 is not the original freeze")
    # 2. the original version is preserved verbatim
    for f, h in entries[0]["files"].items():
        a = ARCHIVE / "prereg-v1" / f
        if not a.exists() or _sha(a) != h:
            problems.append(f"archive/prereg-v1 missing or altered: {f}")
    # 3. tier 1: current files equal the latest entry
    ok1, d1 = verify()
    problems += [f"tier-1 file changed without amendment: {f}" for f in d1]
    # 4. each amendment entry has a complete log row and a pre-amendment archive
    for e in entries[1:]:
        aid = e.get("amendment")
        ok, msg = amendment_complete(str(aid))
        if not ok:
            problems.append(msg)
        if not (ARCHIVE / f"pre-{aid}").exists():
            problems.append(f"missing archive/pre-{aid}")
    # 5. tier 2: changes must be logged; decision-bearing files need an allowed class
    if SUPPLEMENT.exists():
        sup = json.loads(SUPPLEMENT.read_text())
        rows = changelog_rows()
        by_file: dict[str, list[str]] = {}
        for r in rows:
            if len(r) >= 4:
                by_file.setdefault(r[2], []).append(r[3])
        for r in rows:
            if len(r) >= 4 and r[3] not in CHANGE_CLASSES and not r[3].startswith("amendment:A"):
                problems.append(f"changelog {r[0]}: invalid class '{r[3]}'")
            if len(r) >= 4 and r[3].startswith("amendment:"):
                aid = r[3].split(":", 1)[1]
                if aid not in amendment_rows():
                    problems.append(f"changelog {r[0]} cites unknown amendment {aid}")
        decision = set(sup.get("decision_bearing", []))
        for f, h in sup["files"].items():
            p = ROOT / f
            if p.exists() and _sha(p) == h:
                continue
            classes = by_file.get(f)
            if not classes:
                problems.append(f"tier-2 file changed without changelog entry: {f}")
            elif f in decision and not all(c in DECISION_OK or c.startswith("amendment:A") for c in classes):
                problems.append(f"decision-bearing file {f} changed with class {classes} "
                                "(allowed: bugfix, implementation-per-spec, amendment:A<n>)")
    else:
        problems.append("registry/FREEZE_SUPPLEMENT.json missing")
    return not problems, problems


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
        print(json.dumps({"manifest_hash": e["manifest_hash"], "files": len(e["files"]), "amendment": amend}, indent=2))
        return 0
    if cmd == "begin-amendment":
        print(f"snapshot written to {begin_amendment(argv[1])}")
        return 0
    if cmd == "verify":
        ok, problems = verify_all()
        print("PRE-REGISTRATION INTACT (tier 1, tier 2, archive, logs)" if ok else
              "INTEGRITY PROBLEMS:\n" + "\n".join(problems))
        return 0 if ok else 1
    raise SystemExit(f"unknown command {cmd}")
