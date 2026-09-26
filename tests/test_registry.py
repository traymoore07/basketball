"""The pre-registration must stay valid and frozen."""
from hoopslab import registry


def test_registry_validates():
    assert registry.validate() == []


def test_frozen_files_unchanged_or_amended():
    ok, diffs = registry.verify()
    assert ok, f"frozen files changed without an amendment entry: {diffs}"
