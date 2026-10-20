"""Sampling, calendar labels, weighted scoring and cross-company controls."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.ml.design_calibration import WeightedSigmoid, choose_threshold, metrics
from src.ml.robust_models import train_candidate
from src.validation.cohort_v3 import fold_for, validate_parts
from src.validation.cohort_v3_data import make_snapshots, sampling_record


def test_company_design_weights_do_not_depend_on_snapshot_outcome():
    config = {"seed": "frozen", "background_probability": .1}
    case = sampling_record("0000000001", True, config)
    background = sampling_record("0000000001", False, config)
    assert case["selected"] and case["design_weight"] == 1
    assert background["design_weight"] == 10
    assert case["role"] == background["role"]
    # A future case company's earlier negative snapshots must also carry weight 1.
    assert case["inclusion_probability"] * case["design_weight"] == 1
    assert sampling_record("0000000001", True, config) == case


def test_calendar_snapshot_preserves_public_feature_date_and_rejects_future():
    history = pd.DataFrame([
        {"filing_date": "2018-11-01", "filing_type": "10-Q", "accession_number": "old", "Assets": 1e9},
        {"filing_date": "2018-12-31", "filing_type": "10-Q", "accession_number": "same-day", "Assets": 2e9},
        {"filing_date": "2019-01-02", "filing_type": "10-Q", "accession_number": "future", "Assets": 3e9}])
    company = {"cik": "1", "role": "final", "inclusion_probability": .1, "design_weight": 10, "sampling_stratum": "background"}
    config = {"final_anchors": ["2018-12-31"], "max_feature_age_days": 180}
    annual = pd.DataFrame([{"filed": "2018-03-01", "sic": 2000}])
    rows, excluded = make_snapshots(history, annual, company, config)
    assert not excluded
    assert rows[0]["source_accession_number"] == "old"
    assert rows[0]["filing_date"] == "2018-12-31"
    assert rows[0]["feature_filing_date"] == "2018-11-01"
    assert rows[0]["design_weight"] == 10
    config["max_feature_age_days"] = 30
    assert make_snapshots(history, annual, company, config)[1][0]["reason"] == "stale_financial_inputs_abstain"


def test_design_weighted_metrics_match_expanded_population():
    y, p, w = np.array([1, 0, 0]), np.array([.8, .7, .1]), np.array([1, 10, 10])
    actual = metrics(y, p, w, .5)
    expanded = metrics(np.repeat(y, w), np.repeat(p, w), np.ones(w.sum()), .5)
    assert actual["precision"] == pytest.approx(1 / 11)
    assert actual["recall"] == 1
    for key in ("brier", "log_loss", "average_precision", "roc_auc", "accuracy"):
        assert actual[key] == pytest.approx(expanded[key])
    assert choose_threshold(y, p, w, .5, .5) == .8
    assert choose_threshold([0, 0], [.1, .2], [1, 1]) is None
    with pytest.raises(ValueError):
        metrics(y, p, [1, 0, 1])


def test_sigmoid_uses_design_weights_and_requires_real_outcomes():
    base = DummyClassifier().fit([[0], [1]], [0, 1])
    p, y = np.array([.2, .2, .8, .8]), np.array([0, 1, 0, 1])
    a = WeightedSigmoid(base).fit_scores(p, y, [10, 1, 10, 1])
    b = WeightedSigmoid(base).fit_scores(p, y, [1, 10, 1, 10])
    assert a.predict_scores(p).mean() < b.predict_scores(p).mean()
    with pytest.raises(ValueError, match="both outcomes"):
        WeightedSigmoid(base).fit_scores([.1, .2], [0, 0], [1, 1])
    assert fold_for("123", "fixed", 3) == fold_for("123", "fixed", 3)


def parts():
    result = {}
    for role, cik, year in [("training", "1", 2012), ("calibration", "2", 2016), ("final", "3", 2020)]:
        result[role] = pd.DataFrame([{"cik": cik, "unit_id": f"{cik}@{year}", "prediction_date": f"{year}-12-31",
            "feature_filing_date": f"{year}-11-01", "latest_input_filing_date": f"{year}-11-01",
            "label_available_date_1yr": f"{year+1}-12-31", "source_url": "https://www.sec.gov/Archives/example",
            "target_bankruptcy_next_1yr": 0, "design_weight": 10., "inclusion_probability": .1}])
    return result


def test_stage_validation_rejects_company_leakage_and_censored_negatives():
    stages = parts()
    validate_parts(stages)
    stages["final"].loc[0, "cik"] = "1"
    with pytest.raises(ValueError, match="Company leakage"):
        validate_parts(stages)
    stages = parts()
    stages["final"].loc[0, "label_available_date_1yr"] = "2021-06-01"
    with pytest.raises(ValueError, match="full calendar-year"):
        validate_parts(stages)
    stages = parts()
    stages["final"].loc[0, "latest_input_filing_date"] = "2021-01-01"
    with pytest.raises(ValueError, match="point-in-time"):
        validate_parts(stages)


def test_training_rejects_invalid_design_weights():
    data = pd.DataFrame({"cik": ["1", "2", "3", "4"], "x": [.1, .2, .8, .9], "y": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="Training weights"):
        train_candidate(data, ["x"], "y", "logistic_regularized", sample_weight=[1, -1, 1, 1])


def test_final_evaluation_is_consumed_once_and_never_auto_promotes(tmp_path, monkeypatch):
    import src.validation.cohort_v3 as experiment
    import src.validation.cohort_v3_audit as auditing
    from contextlib import nullcontext
    from pathlib import Path
    (tmp_path / 'candidate_selection.json').write_text('{}')
    (tmp_path / 'development_report.json').write_text('{"calibration_positive_company_gate": true}')
    frame = pd.DataFrame({"cik": ['a', 'b'], "prediction_date": ['2020-12-31', '2020-12-31'], "target_bankruptcy_next_1yr": [0, 1], "design_weight": [10., 1.]})
    calls = []
    class Model:
        def predict_proba(self, X):
            calls.append(True)
            return np.array([[.9, .1], [.1, .9]])
    frozen = {"artifact_sha256": "hash", "dataset_sha256": "data", "threshold": .5, "constant_baseline": .1}
    config = {"min_final_companies": 1, "min_final_positive_companies": 1, "min_precision": .5, "min_recall": .5}
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'lock', nullcontext)
    monkeypatch.setattr(experiment, 'verify_selection', lambda: frozen)
    monkeypatch.setattr(experiment, 'read_parts', lambda: (config, {}, {'final': frame}))
    monkeypatch.setattr(experiment, 'inputs', lambda f: np.zeros((len(f), 1)))
    monkeypatch.setattr(experiment, 'load_model', lambda *a, **k: {'feature_cols': experiment.FEATURES, 'target': experiment.TARGET, 'trained_through': '2018-12-30', 'model': Model()})
    monkeypatch.setattr(auditing, 'audit', lambda: {})
    report = experiment.evaluate_once()
    assert not report['production_ready']
    assert report['gates']['precision_floor']
    assert not report['gates']['current_population_source_completeness_reviewed']
    with pytest.raises(FileExistsError):
        experiment.evaluate_once()
    assert len(calls) == 1
    assert not experiment.summary()['production_ready']
    (tmp_path / 'final_evaluation.json').write_text('{"production_ready": true}')
    with pytest.raises(ValueError, match='Saved final evidence changed'):
        experiment.summary()


def test_reliability_bins_include_zero_one_and_respect_design_weights():
    from src.ml.design_calibration import reliability
    rows = reliability([0, 1, 0, 1], [0., .01, .01, 1.], [10, 1, 10, 1])
    assert sum(row['rows'] for row in rows) == 4
    assert rows[1]['weighted_observed_rate'] == pytest.approx(1 / 11)
    assert rows[-1]['weighted_mean_probability'] == 1


def test_source_audit_rejects_silently_healthy_unknowns(tmp_path, monkeypatch):
    import json
    import src.validation.cohort_v3_audit as auditing
    from src.validation.cohort_v3_data import digest
    config, plan = tmp_path / 'config.json', tmp_path / 'plan.json'
    config.write_text('{}')
    plan.write_text('{}')
    pd.DataFrame({'cik': ['1'], 'unit_id': ['1@2020'], 'eligible': [False],
                  'target_bankruptcy_next_1yr': [0]}).to_csv(tmp_path / 'dataset.csv', index=False)
    (tmp_path / 'dataset_manifest.json').write_text(json.dumps({'config_sha256': digest(config),
        'plan_sha256': digest(plan), 'dataset_sha256': digest(tmp_path / 'dataset.csv')}))
    (tmp_path / 'acquisition_report.json').write_text('{"errors": []}')
    monkeypatch.setattr(auditing, 'ROOT', tmp_path)
    monkeypatch.setattr(auditing, 'CONFIG', config)
    monkeypatch.setattr(auditing, 'PLAN', plan)
    with pytest.raises(ValueError, match='unknown outcome used as a label'):
        auditing.audit(verify_raw=False)
