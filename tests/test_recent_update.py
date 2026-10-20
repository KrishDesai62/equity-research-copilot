import json
import zipfile

import pandas as pd

from src.data_layer.xbrl_bulk import parse_zip_to_dataframe
from src.validation.update_recent import last_completed_quarter, preserve_snapshot
from src.validation.download import fetch


def test_completed_quarter_uses_explicit_asof_and_handles_year_boundary():
    assert last_completed_quarter("2026-10-16") == (2026, 3)
    assert last_completed_quarter("2026-01-01") == (2025, 4)
    assert last_completed_quarter("2026-09-30") == (2026, 2)


def test_recent_numeric_data_excludes_segments_custom_tags_and_other_companies(tmp_path):
    path = tmp_path / "quarter.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub.txt", "adsh\tcik\tname\tfiled\tperiod\tfy\tfp\tform\n"
            "a\t1\tA\t20260801\t20260630\t2026\tQ2\t10-Q\n"
            "b\t2\tB\t20260801\t20260630\t2026\tQ2\t10-Q\n")
        archive.writestr("num.txt", "adsh\ttag\tversion\tvalue\tddate\tqtrs\tuom\tcoreg\tsegments\n"
            "a\tAssets\tus-gaap/2026\t100\t20260630\t0\tUSD\t\t\n"
            "a\tAssets\tus-gaap/2026\t20\t20260630\t0\tUSD\t\tBusinessSegment=Widgets\n"
            "a\tAssets\tcompany/2026\t500\t20260630\t0\tUSD\t\t\n"
            "b\tAssets\tus-gaap/2026\t200\t20260630\t0\tUSD\t\t\n")
    frame = parse_zip_to_dataframe(path, ciks=["1"])
    assert frame.value.tolist() == [100]
    assert frame.cik.tolist() == ["0000000001"]
    assert frame.filing_date.iloc[0] == pd.Timestamp("2026-08-01")


def test_previous_research_snapshot_and_review_work_are_preserved(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"observation_end": "2024-12-31"}))
    (root / "review.csv").write_text("human review")
    saved = preserve_snapshot(root)
    from pathlib import Path
    assert (Path(saved) / "review.csv").read_text() == "human review"
    (root / "review.csv").write_text("new review")
    assert preserve_snapshot(root) == saved
    assert (Path(saved) / "review.csv").read_text() == "human review"


def test_transient_public_source_timeout_retries_without_publishing_partial_cache(tmp_path, monkeypatch):
    import requests
    calls = []
    class Response:
        content = b"complete source"
        def raise_for_status(self):
            pass
    def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("temporary interruption")
        return Response()
    monkeypatch.setattr("src.validation.download.requests.get", get)
    monkeypatch.setattr("src.validation.download.limiter.wait", lambda: None)
    monkeypatch.setattr("src.validation.download.time.sleep", lambda _: None)
    path = tmp_path / "download"
    result = fetch("https://example.org/source", path)
    assert len(calls) == 2
    assert path.read_bytes() == b"complete source"
    assert result["bytes"] == len(b"complete source")
