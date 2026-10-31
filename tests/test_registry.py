"""The pre-registration must stay valid, frozen, and its original version preserved."""
import json

from hoopslab import registry


def test_registry_validates():
    assert registry.validate() == []


def test_tier1_frozen_files_unchanged_or_amended():
    ok, diffs = registry.verify()
    assert ok, f"frozen files changed without an amendment entry: {diffs}"


def test_full_integrity_tier2_archive_logs():
    ok, problems = registry.verify_all()
    assert ok, "\n".join(problems)


def test_original_freeze_entry_is_the_pre_data_freeze():
    e0 = json.loads((registry.REG / "FREEZE.json").read_text())["entries"][0]
    assert e0["amendment"] is None and e0["real_data_seen"] is False
    assert e0["manifest_hash"] == "ccb065025fb3d4874b4e9fe12a6b7ead3ad4eb3cab6661214efb2c3a17bd9c40"


def test_amendment_requires_complete_log_row(tmp_path, monkeypatch):
    log = tmp_path / "AMENDMENTS.md"
    log.write_text("| A1 | 2026-10-01 | x | y | z | RESEARCH PREFERENCE | – | H2 | none | PRE-OUTCOME |\n")
    monkeypatch.setattr(registry, "AMEND_LOG", log)
    ok, msg = registry.amendment_complete("A1")
    assert not ok and "incomplete" in msg          # '–' in the mismatch column is not a valid entry
    log.write_text("| A1 | 2026-10-01 | x | y | z | PREFERENCE | M1 | H2 | none | PRE-OUTCOME |\n")
    assert not registry.amendment_complete("A1")[0]  # invalid class
    log.write_text("| A1 | 2026-10-01 | x | y | z | DATA-DRIVEN NECESSITY | M1 | H2 | none | PRE-OUTCOME |\n")
    assert registry.amendment_complete("A1")[0]
    assert not registry.amendment_complete("A2")[0]


def test_unlogged_tier2_change_is_detected(tmp_path, monkeypatch):
    sup = json.loads(registry.SUPPLEMENT.read_text())
    f = "src/hoopslab/backtest/runner.py"
    sup["files"][f] = "0" * 64                     # pretend the file differs from its pre-data hash
    p = tmp_path / "sup.json"
    p.write_text(json.dumps(sup))
    monkeypatch.setattr(registry, "SUPPLEMENT", p)
    ok, problems = registry.verify_all()
    assert not ok and any(f in x for x in problems)
    cl = tmp_path / "cl.md"
    real = registry.CHANGELOG.read_text()
    cl.write_text(real + f"| C99 | 2026-10-01 | {f} | engineering | faster | S1 |\n")
    monkeypatch.setattr(registry, "CHANGELOG", cl)
    ok, problems = registry.verify_all()
    assert not ok and any("decision-bearing" in x for x in problems)   # engineering not allowed here
    cl.write_text(real + f"| C99 | 2026-10-01 | {f} | bugfix | lead time ignored spec | S1 |\n")
    ok, problems = registry.verify_all()
    assert ok, problems
