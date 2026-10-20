from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from src.agents_full.risk_agent import RiskAgentTool
from src.ml.artifacts import load_model, save_model
from src.ml.release import assess, validate_release
from src.ml.train_distress import train_logistic_baseline


POLICY = json.loads(Path("config/release_policy.json").read_text())


def good_evidence():
    return {"test_metrics": {"precision": .8, "recall": .8}, "test_companies": 100,
        "independent_positive_events": 30, "independent_test": True,
        "representative_population": True, "validated_calibration": True,
        "preprocessing_fit_on": "training rows only"}


def model_file(tmp_path):
    data = pd.DataFrame({"x": [-3., -2., -1., -.5, .5, 1., 2., 3.], "label": [0, 0, 0, 0, 1, 1, 1, 1]})
    model = train_logistic_baseline(data, ["x"], "label")
    path = tmp_path / "distress.joblib"
    save_model(model, ["x"], path, trained_through="2020-01-01", target="target_bankruptcy_next_1yr")
    return path


def release_file(path):
    evidence = good_evidence()
    release = {"status": "approved", "calibrated": True, "threshold": .5,
        "released_on": "2020-02-01", "evidence": evidence,
        "artifact_sha256": sha256(path.read_bytes()).hexdigest(), "assessment": assess(evidence, POLICY)}
    path.with_suffix(".release.json").write_text(json.dumps(release))
    return release


def features():
    return pd.DataFrame({"ticker": ["TEST"], "cik": ["0000000001"], "filing_date": ["2020-03-01"],
        "latest_input_filing_date": ["2020-03-01"],
        "accession_number": ["0000000001-20-000001"], "source_url": ["https://www.sec.gov/source"], "x": [2.]})


def test_model_with_zero_recall_cannot_pass_release_gate():
    evidence = good_evidence()
    evidence["test_metrics"]["recall"] = 0
    report = assess(evidence, POLICY)
    assert not report["eligible"]
    assert any("recall" in failure for failure in report["failures"])


def test_artifact_checksum_is_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"untrusted content")
    monkeypatch.setattr("src.ml.artifacts.joblib.load", lambda _: pytest.fail("Deserialization must not happen"))
    with pytest.raises(ValueError, match="integrity"):
        load_model(path, expected_sha256="wrong digest")


def test_production_requires_release_and_rechecks_its_evidence(tmp_path):
    path = model_file(tmp_path)
    agent = RiskAgentTool(tmp_path)
    result = agent.get_risk_profile("TEST", "2020-04-01", features_df=features())
    assert result["distress_probability"] is None
    assert result["unavailable_reasons"]["distress"] == "validated_release_required"
    release = release_file(path)
    assert agent.get_risk_profile("TEST", "2020-04-01", features_df=features())["distress_probability"] > .5
    release["evidence"]["test_metrics"]["recall"] = 0
    path.with_suffix(".release.json").write_text(json.dumps(release))
    with pytest.raises(ValueError, match="validation"):
        validate_release(path.with_suffix(".release.json"), path)


def test_inference_abstains_on_stale_missing_or_uncited_features(tmp_path):
    path = model_file(tmp_path)
    release_file(path)
    agent = RiskAgentTool(tmp_path)
    stale = agent.get_risk_profile("TEST", "2021-01-01", features_df=features())
    assert stale["unavailable_reasons"]["distress"] == "stale_financial_features"
    missing = agent.get_risk_profile("TEST", "2020-04-01", features_df=features().assign(x=float("nan")))
    assert missing["unavailable_reasons"]["distress"] == "insufficient_observed_features"
    uncited = agent.get_risk_profile("TEST", "2020-04-01", features_df=features().drop(columns="source_url"))
    assert uncited["unavailable_reasons"]["distress"] == "feature_provenance_required"


def test_inference_rejects_duplicate_or_future_inputs(tmp_path):
    agent = RiskAgentTool(tmp_path)
    with pytest.raises(ValueError, match="Ambiguous"):
        agent.get_risk_profile("TEST", "2020-04-01", features_df=pd.concat([features(), features()]))
    with pytest.raises(ValueError, match="published after"):
        agent.get_risk_profile("TEST", "2020-04-01", features_df=features().assign(latest_input_filing_date="2021-01-01"))


def test_invalid_dates_and_empty_provenance_cannot_bypass_production_checks(tmp_path):
    path = model_file(tmp_path)
    release = release_file(path)
    release["released_on"] = "NaT"
    path.with_suffix(".release.json").write_text(json.dumps(release))
    with pytest.raises(ValueError, match="released_on"):
        validate_release(path.with_suffix(".release.json"), path)
    release_file(path)
    agent = RiskAgentTool(tmp_path)
    result = agent.get_risk_profile("TEST", "2020-04-01", features_df=features().assign(source_url="  "))
    assert result["distress_probability"] is None
    with pytest.raises(ValueError, match="dates must be present"):
        agent.get_risk_profile("TEST", "NaT", features_df=features())


def test_feature_audit_detects_empty_data_and_invalid_publication_dates():
    from src.validation.dataset_audit import audit_features
    frame = features().assign(latest_input_filing_date="not a date")
    assert any("dates" in issue for issue in audit_features(frame, "2020-04-01"))
    assert any("empty" in issue for issue in audit_features(frame.iloc[:0], "2020-04-01"))


def test_deserializer_receives_verified_bytes_even_if_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    original = b"trusted serialized model bytes"
    path.write_bytes(original)
    def deserialize(stream):
        path.write_bytes(b"different artifact published concurrently")
        assert stream.read() == original
        return {"model": object(), "feature_cols": ["x"], "target": "y", "trained_through": "2020-01-01"}
    monkeypatch.setattr("src.ml.artifacts.joblib.load", deserialize)
    assert load_model(path, expected_sha256=sha256(original).hexdigest())["target"] == "y"
