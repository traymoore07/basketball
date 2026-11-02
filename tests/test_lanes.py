"""Lane governance tests (L1 contract, X1 version registry, live-lane guard). Fixture data only: no corpus,
no network, no 2023+ data, no CONTEXT_HOME / SHADOW_HOME content."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from hoopslab.pregame import cli as CLI
from hoopslab.pregame import contract as C
from hoopslab.pregame import versions as V
from hoopslab.shadow.interface import AvailabilityState

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX = os.path.join(ROOT, "live_inputs", "example_snapshot")


def test_fixture_snapshot_passes_the_contract():
    snap = C.load_and_validate(FIX)
    assert snap.ok, snap.problems
    assert snap.counts["availability"] == 9 and snap.counts["roster"] == 9
    assert len(snap.snapshot_id) == 64


def test_rows_after_the_cutoff_are_refused_not_dropped():
    manifest, tables = C.read_snapshot_dir(FIX)
    av = tables["availability"].copy()
    av.loc[0, "published_at"] = "2010-01-31T23:30:00Z"   # after the 23:00 cutoff
    tables["availability"] = av
    snap = C.validate(manifest, tables)
    assert not snap.ok and any("after the cutoff" in p for p in snap.problems)
    assert len(snap.tables["availability"]) == 9      # nothing dropped silently


def test_unknown_is_never_active_and_official_cannot_be_unknown_or_inferred():
    snap = C.load_and_validate(FIX)
    el = C.to_eligible_v01(snap)
    p6 = el.set_index("player_id").loc["P6", "availability_state"]
    assert p6 == AvailabilityState.QUESTIONABLE_UNKNOWN.value
    assert all(v != AvailabilityState.CONFIRMED_ACTIVE and v != AvailabilityState.EXPECTED_ACTIVE
               for (a, _), v in C.V01_MAP.items() if a == C.Availability.UNKNOWN)
    manifest, tables = C.read_snapshot_dir(FIX)
    av = tables["availability"].copy()
    av.loc[av["player_id"] == "P6", "official"] = True
    bad = C.validate(manifest, {**tables, "availability": av})
    assert any("official row cannot be UNKNOWN" in p for p in bad.problems)
    av = tables["availability"].copy()
    av.loc[av["player_id"] == "P2", "source_kind"] = "inferred"
    av.loc[av["player_id"] == "P2", "official"] = True
    bad = C.validate(manifest, {**tables, "availability": av})
    assert any("inferred state cannot be official" in p for p in bad.problems)


def test_mapping_to_the_frozen_four_states():
    el = C.to_eligible_v01(C.load_and_validate(FIX)).set_index("player_id")["availability_state"]
    assert el["P1"] == "CONFIRMED_ACTIVE"      # ACTIVE + official
    assert el["P2"] == "EXPECTED_ACTIVE"       # ACTIVE, beat writer
    assert el["P3"] == "EXPECTED_ACTIVE"       # PROBABLE
    assert el["P4"] == "QUESTIONABLE_UNKNOWN"  # QUESTIONABLE
    assert el["P8"] == "QUESTIONABLE_UNKNOWN"  # DOUBTFUL
    assert el["P5"] == "OUT"
    assert set(el.unique()) <= {s.value for s in AvailabilityState}


def test_team_absences_must_be_out_or_doubtful_with_known_roles():
    manifest, tables = C.read_snapshot_dir(FIX)
    ta = tables["team_absences"].copy()
    ta.loc[0, "availability"] = "ACTIVE"
    assert any("only OUT or DOUBTFUL" in p for p in C.validate(manifest, {**tables, "team_absences": ta}).problems)
    ta = tables["team_absences"].copy()
    ta.loc[0, "role_removed"] = "SUPERSTAR"
    assert any("role taxonomy" in p for p in C.validate(manifest, {**tables, "team_absences": ta}).problems)


def test_minutes_distribution_contract():
    d = C.minutes_distribution_from_pmf(np.r_[np.zeros(20), np.ones(21), np.zeros(30)])
    assert d.validate() == [] and 20 <= d.p10 <= d.p50 <= d.p90 <= 40 and abs(d.mean - 30) < 1e-9
    bad = C.MinutesDistribution(mean=30, p10=25, p25=20, p50=30, p75=35, p90=40)
    assert "quantiles not monotone" in bad.validate()
    assert "median" in d.as_dict() and len(d.as_dict()["pmf"]) == 71


def test_version_registry_verifies_the_frozen_baseline_and_refuses_planned():
    ok, probs = V.verify_version("L2-A-SHADOW-v0.1")
    assert ok, probs
    card = json.load(open(os.path.join(ROOT, "shadow", "MODEL_CARD_L2-A-SHADOW-v0.1.json"), encoding="utf-8"))
    assert card["card_sha256"] == V.get_version("L2-A-SHADOW-v0.1")["card_sha256"]
    for v in ("SEASON-AWARE-SHADOW-v0.2", "AVAILABILITY-SHADOW-v0.3", "CONTEXT-SHADOW-v0.4"):
        ok, probs = V.verify_version(v)
        assert not ok and "PLANNED" in probs[0]
    assert V.forecastable_versions() == ["L2-A-SHADOW-v0.1"]


def test_research_lane_cannot_forecast_score_or_report(monkeypatch, tmp_path):
    monkeypatch.delenv("HOOPSLAB_LANE", raising=False)
    monkeypatch.setenv("CONTEXT_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="not 'live'"):
        CLI.main(["forecast", "--version", "L2-A-SHADOW-v0.1", "--snapshot", FIX, "--shadow-snapshot", str(tmp_path), "--game-id", "G:FIXTURE:1"])
    with pytest.raises(SystemExit, match="not 'live'"):
        CLI.main(["report", "--version", "L2-A-SHADOW-v0.1"])
    monkeypatch.setenv("HOOPSLAB_LANE", "live")
    monkeypatch.setenv("CONTEXT_HOME", os.path.join(ROOT, "results"))      # inside the repository: refused
    with pytest.raises(SystemExit, match="inside the research repository"):
        CLI.main(["report", "--version", "L2-A-SHADOW-v0.1"])
    monkeypatch.setenv("CONTEXT_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="PLANNED"):
        CLI.main(["forecast", "--version", "SEASON-AWARE-SHADOW-v0.2", "--snapshot", FIX, "--shadow-snapshot", str(tmp_path), "--game-id", "G:FIXTURE:1"])


def test_research_lane_reads_only_the_fixture_and_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.delenv("HOOPSLAB_LANE", raising=False)
    other = tmp_path / "snap"
    other.mkdir()
    with pytest.raises(SystemExit, match="only the fixture snapshot"):
        CLI.main(["validate", "--snapshot", str(other)])
    with pytest.raises(SystemExit, match="only the fixture snapshot"):
        CLI.main(["eligible", "--snapshot", str(other), "--out", str(tmp_path / "e.csv")])
    out = tmp_path / "eligible.csv"
    assert CLI.main(["eligible", "--snapshot", FIX, "--out", str(out)]) == 0
    with pytest.raises(SystemExit, match="never overwritten"):
        CLI.main(["eligible", "--snapshot", FIX, "--out", str(out)])


def test_live_forecast_refuses_shadow_home_and_existing_derived_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HOOPSLAB_LANE", "live")
    monkeypatch.setenv("CONTEXT_HOME", str(tmp_path / "ctx"))
    monkeypatch.setenv("SHADOW_HOME", str(tmp_path / "shadow_lane"))
    inside = tmp_path / "shadow_lane" / "snapshots" / "x"
    inside.mkdir(parents=True)
    with pytest.raises(SystemExit, match="under SHADOW_HOME"):
        CLI.main(["forecast", "--version", "L2-A-SHADOW-v0.1", "--snapshot", FIX, "--shadow-snapshot", str(inside), "--game-id", "G:FIXTURE:1"])
    ss = tmp_path / "ss"
    ss.mkdir()
    (ss / "eligible.csv").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        CLI.main(["forecast", "--version", "L2-A-SHADOW-v0.1", "--snapshot", FIX, "--shadow-snapshot", str(ss), "--game-id", "G:FIXTURE:1"])
    monkeypatch.setenv("CONTEXT_HOME", os.path.join(os.path.expanduser("~"), "Documents", "Claude", "context_lane"))
    with pytest.raises(SystemExit, match="research working folder"):
        CLI.main(["report", "--version", "L2-A-SHADOW-v0.1"])
    assert "Documents" not in CLI.DEFAULT_CONTEXT_HOME.replace(os.path.expanduser("~"), "")


def test_version_plan_names_the_live_rules():
    plan = V.load_plan()
    assert any("betting lines" in r for r in plan["rules"])
    v = {x["model_version"]: x for x in plan["versions"]}
    assert "live_decision_rule" in v["SEASON-AWARE-SHADOW-v0.2"] and "0.01" in v["SEASON-AWARE-SHADOW-v0.2"]["live_decision_rule"]["minimum_effect"]
    assert "NO ARTEFACT" in v["AVAILABILITY-SHADOW-v0.3"]["lineage"] or "otherwise L2-A-SHADOW-v0.1" in v["AVAILABILITY-SHADOW-v0.3"]["lineage"]


def test_cli_validate_and_eligible_on_the_fixture(tmp_path, capsys):
    assert CLI.main(["validate", "--snapshot", FIX]) == 0
    out = tmp_path / "eligible.csv"
    assert CLI.main(["eligible", "--snapshot", FIX, "--out", str(out)]) == 0
    el = pd.read_csv(out)
    assert list(el.columns) == ["game_id", "player_id", "team_id", "availability_state", "status_source", "status_time"]
    assert len(el) == 9
