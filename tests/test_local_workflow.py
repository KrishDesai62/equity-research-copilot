import json
import os
from pathlib import Path

import pytest

from src.aws.s3_cache import S3Cache
from src.validation.local_workflow import local_only, run, saved_evidence


def test_local_mode_never_initializes_s3_and_restores_environment(monkeypatch):
    monkeypatch.setenv("AWS_S3_BUCKET", "configured-bucket")
    monkeypatch.setenv("EQUITY_LOCAL_ONLY", "previous-value")
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("AWS must not initialize"))
    with pytest.raises(RuntimeError):
        with local_only():
            assert S3Cache.from_env() is None
            raise RuntimeError("interrupted run")
    assert os.environ["EQUITY_LOCAL_ONLY"] == "previous-value"


def test_saved_results_require_matching_artifact(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"candidate_artifact_sha256": "model-a", "test_metrics": {"recall": .7}}))
    with pytest.raises(ValueError, match="does not match"):
        saved_evidence(path, "model-b")
    assert saved_evidence(path, "model-a")["status"] == "saved_evidence_only"


def test_local_only_flag_loaded_from_dotenv_also_disables_s3(monkeypatch):
    monkeypatch.delenv("EQUITY_LOCAL_ONLY", raising=False)
    monkeypatch.setenv("AWS_S3_BUCKET", "configured-bucket")
    monkeypatch.setattr("src.aws.s3_cache.load_dotenv", lambda: monkeypatch.setenv("EQUITY_LOCAL_ONLY", "true"))
    monkeypatch.setattr("boto3.client", lambda *a, **kw: pytest.fail("Must honor .env local mode"))
    assert S3Cache.from_env() is None


def test_cached_local_run_needs_no_network_and_retains_release_blockers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({}))  # Production config also uses the default start date.
    root = tmp_path / "data/processed/research"
    root.mkdir(parents=True)
    for name in ("summary.json", "financial_features.csv", "filings.sqlite"):
        (root / name).touch()
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("Default run must not download"))
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("Must not access AWS"))
    def readiness(as_of):
        assert S3Cache.from_env() is None
        return {"data_issues": [], "model_release": {"eligible": False, "failures": ["calibration missing"]},
            "research_snapshot_age_days": 1, "unfinished_product_components": ["recommendation validation"]}
    monkeypatch.setattr("src.validation.production_readiness.run", readiness)
    result = run(as_of="2020-01-01", config=str(config))
    assert result["status"] == "completed_research_only"
    assert not result["production_ready"] and not result["holdout_rescored"]
    assert result["steps"]["validation"]["model_release"]["failures"] == ["calibration missing"]
    assert (Path(result["run_directory"]) / "report.json").exists()


def test_local_run_reports_missing_data_and_validates_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"start": "2010-01-01"}))
    result = run(as_of="2020-01-01", config=str(config))
    assert result["status"] == "needs_local_data"
    assert len(result["missing_files"]) == 3
    with pytest.raises(ValueError, match="both"):
        run(as_of="2020-01-01", config=str(config), ticker="AAPL")
    with pytest.raises(ValueError, match="future"):
        run(as_of="2099-01-01", config=str(config))
