import json
from pathlib import Path

import pytest

from src.data_layer.snapshots import publish_snapshot


def version(path, good=True):
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"complete": good, "companies": 1}))
    (path / "research_checks.json").write_text(json.dumps({"qa_checks": {"passed": 1, "early_date_abstains": True, "unknown_company_abstains": True}}))
    (path / "recent_update.json").write_text("{}")
    (path / "filings.sqlite").write_text("test index")
    return path


def test_failed_refresh_cannot_replace_current_snapshot(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    (current / "review").write_text("existing")
    with pytest.raises(ValueError, match="publication"):
        publish_snapshot(version(tmp_path / "bad", False), current)
    assert (current / "review").read_text() == "existing"


def test_versions_publish_atomically_and_preserve_previous_data(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    (current / "review").write_text("existing")
    first = version(tmp_path / "first")
    publication = publish_snapshot(first, current)
    assert current.is_symlink() and current.resolve() == first
    assert (Path(publication["legacy_preserved"]) / "review").read_text() == "existing"
    second = version(tmp_path / "second")
    publish_snapshot(second, current)
    assert current.resolve() == second and first.exists()


def test_concurrent_publisher_lock_is_respected(tmp_path):
    lock = tmp_path / ".research_publish.lock"
    lock.write_text("another publisher")
    with pytest.raises(FileExistsError):
        publish_snapshot(version(tmp_path / "ready"), tmp_path / "current")
    assert lock.read_text() == "another publisher"
