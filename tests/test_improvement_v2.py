import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.robust_models import InterceptCalibrator, QuantileClipper, train_candidate
from src.validation.improvement_data import sample_vintage
from src.validation.improve_v2 import make_parts, validate_stages


def test_outlier_limits_are_fitted_on_training_only():
    clip = QuantileClipper(.1, .9).fit([[1.], [2.], [3.]])
    bounds = clip.upper_.copy()
    result = clip.transform([[1e20]])
    np.testing.assert_array_equal(bounds, clip.upper_)
    assert result[0, 0] == bounds[0]


def test_intercept_calibration_handles_zero_events_without_asserting_zero_risk():
    class Model:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])
    X = np.array([[.1], [.2], [.4], [.8]])
    calibration = InterceptCalibrator(Model()).fit(X, [0, 0, 0, 0])
    scores = calibration.predict_proba(X)[:, 1]
    assert scores.mean() == pytest.approx(1 / 6)
    assert np.all(scores > 0) and np.all(scores < 1)
    assert np.all(np.diff(scores) > 0)
    with pytest.raises(ValueError):
        InterceptCalibrator(Model(), pseudocount=np.nan).fit(X, [0, 0, 0, 0])


def test_vintage_sampling_does_not_use_outcomes_and_retains_entry_date():
    frame = pd.DataFrame({"cik": [1, 2, 3], "form": ["10-K"] * 3,
        "sic": [3500] * 3, "filed": [20140301, 20140302, 20140303], "target": [0, 0, 0]})
    first = sample_vintage(frame, "seed", 2014, 2, {"0000000002"})
    second = sample_vintage(frame.sample(frac=1), "seed", 2014, 2, {"0000000002"})
    assert first.cik.tolist() == second.cik.tolist()
    assert "0000000002" not in first.cik.tolist()
    assert first.filed.min().year == 2014


def test_stage_validation_purges_company_and_outcome_leakage():
    parts = {}
    for number, (name, year) in enumerate((("training", 2014), ("calibration", 2016), ("selection", 2018), ("final", 2020))):
        parts[name] = pd.DataFrame({"cik": [str(number).zfill(10)], "filing_date": [f"{year}-03-01"],
            "latest_input_filing_date": [f"{year}-03-01"], "label_available_date_1yr": [f"{year+1}-03-01"],
            "target_distressed_1yr": [0]})
    validate_stages(parts, include_final=True)
    overlap = copy.deepcopy(parts)
    overlap["final"]["cik"] = parts["training"].cik.values
    with pytest.raises(ValueError, match="overlap"):
        validate_stages(overlap, include_final=True)
    parts["calibration"]["label_available_date_1yr"] = "2019-01-01"
    with pytest.raises(ValueError, match="unavailable"):
        validate_stages(parts, include_final=True)


def test_natural_anchors_are_chosen_before_label_eligibility():
    config = json.loads(Path("config/improvement_v2.json").read_text())
    rows = []
    for company in range(100):
        for i, filed in enumerate(("2016-03-01", "2016-04-01")):
            rows.append({"cik": str(company).zfill(10), "filing_date": filed, "accession_number": str(i),
                "fiscal_period_end": "2015-12-31", "valid_for_training": bool(i),
                "historical_coverage_eligible": bool(i), "target_distressed_1yr": 0,
                "label_available_date_1yr": "2017-04-01"})
    natural = pd.DataFrame(rows)
    original = natural.iloc[:0].copy()
    parts = make_parts(natural, original, config)
    assert parts["calibration"].empty  # Never jump to the eligible second anchor.


@pytest.mark.parametrize("name", ["logistic_regularized", "xgb_regularized", "neural_network"])
def test_research_variants_fit_and_return_finite_probabilities(name):
    frame = pd.DataFrame({"cik": ["1", "1", "2", "2", "3", "3", "4", "4"],
        "x": [-4., -3., -2., -1., 1., 2., 3., 4.], "y": [0, 0, 0, 0, 1, 1, 1, 1]})
    model = train_candidate(frame, ["x"], "y", name)
    probabilities = model.predict_proba(frame[["x"]])
    assert probabilities.shape == (8, 2) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_better_than_old_model_does_not_imply_validated_when_constant_is_better():
    from src.validation.improve_v2 import assess_result
    result = {"probability_metrics": {"v2": {"brier_loss": .02, "log_loss": .06},
        "previous_candidate": {"brier_loss": .06, "log_loss": .7},
        "constant_calibration_rate": {"brier_loss": .001, "log_loss": .02}},
        "final_cohort": {"positive_companies": 0}, "representative_population": False, "validated_calibration": False}
    development = {"candidate": {"threshold": None}, "partitions": {"calibration": {"positive_companies": 0}}}
    assessment = assess_result(result, development)
    assert assessment["brier_improved_vs_previous"]
    assert not assessment["production_ready"]
    assert any("constant-rate" in reason for reason in assessment["failures"])


def test_completed_v2_evaluation_cannot_generate_predictions_again(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.validation import improve_v2 as experiment
    parts = {}
    for i, (name, year) in enumerate((("training", 2014), ("calibration", 2016), ("selection", 2018), ("final", 2020))):
        parts[name] = pd.DataFrame({"cik": [str(i).zfill(10)], "filing_date": [f"{year}-03-01"],
            "latest_input_filing_date": [f"{year}-03-01"], "label_available_date_1yr": [f"{year+1}-03-01"],
            "target_distressed_1yr": [0]})
    parts["final"].to_csv(tmp_path / "final_features.csv", index=False)
    config = tmp_path / "config.json"
    config.write_text("{}")
    choice = {"config_sha256": experiment.digest(config), "data_sha256": "data-hash", "training_source_hashes": {},
        "final_features_sha256": experiment.digest(tmp_path / "final_features.csv"),
        "artifact_sha256": "v2", "baseline_sha256": "prior", "development_companies": [str(i).zfill(10) for i in range(3)]}
    (tmp_path / "candidate_selection.json").write_text(json.dumps(choice))
    marker = tmp_path / "evaluation_started.json"
    marker.write_text("consumed")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "CONFIG", config)
    monkeypatch.setattr(experiment, "read_parts", lambda: ({}, {"features_sha256": "data-hash"}, {}, parts))
    model = SimpleNamespace(predict_proba=lambda *a: pytest.fail("No rescore permitted"))
    monkeypatch.setattr(experiment, "load_model", lambda *a, **kw: {"model": model})
    with pytest.raises(FileExistsError):
        experiment.evaluate_once()
    assert marker.read_text() == "consumed" and not (tmp_path / ".experiment.lock").exists()
