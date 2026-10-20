import pandas as pd
import pytest

from src.validation.independent_candidate import validate_test


def inputs():
    frame = pd.DataFrame({"cik": ["0000000001", "0000000002"],
        "filing_date": ["2020-02-01", "2020-03-01"],
        "latest_input_filing_date": ["2020-02-01", "2020-03-01"],
        "label_available_date_1yr": ["2021-02-01", "2021-03-01"],
        "fiscal_period_end": ["2019-12-31", "2019-12-31"],
        "target_distressed_1yr": [1, 0]})
    manifest = {"companies": [{"cik": "0000000001"}, {"cik": "0000000002"}],
        "excluded_companies": ["0000000003"], "test_start": "2020-01-01",
        "test_end": "2021-12-31", "observation_end": "2022-12-31"}
    return frame, manifest


def test_new_final_requires_new_companies_and_purged_dates():
    frame, plan = inputs()
    validate_test(frame, plan, "2020-01-01")
    with pytest.raises(ValueError, match="leakage"):
        validate_test(frame.assign(cik=["0000000003", "0000000002"]), plan, "2020-01-01")
    with pytest.raises(ValueError, match="Future"):
        validate_test(frame, plan, "2020-04-01")
    with pytest.raises(ValueError, match="Future"):
        validate_test(frame.assign(latest_input_filing_date="2022-01-01"), plan, "2020-01-01")


def test_new_final_rejects_censored_outcomes_duplicates_and_single_class():
    frame, plan = inputs()
    with pytest.raises(ValueError, match="outside"):
        validate_test(frame.assign(label_available_date_1yr="2023-01-01"), plan, "2020-01-01")
    with pytest.raises(ValueError, match="duplicate"):
        validate_test(pd.concat([frame, frame]), plan, "2020-01-01")
    with pytest.raises(ValueError, match="both classes"):
        validate_test(frame.assign(target_distressed_1yr=0), plan, "2020-01-01")
    short = frame.copy()
    short.loc[1, "label_available_date_1yr"] = "2020-04-01"
    with pytest.raises(ValueError, match="complete one-year"):
        validate_test(short, plan, "2020-01-01")


def test_consumed_evaluation_marker_blocks_predictions(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import numpy as np
    from src.validation import independent_candidate as experiment
    frame, plan = inputs()
    frame.to_csv(tmp_path / "test_features.csv", index=False)
    plan["artifact_sha256"] = "trusted digest"
    manifest_path = tmp_path / "plan.json"
    manifest_path.write_text(json.dumps(plan))
    sealed = {"plan_sha256": experiment.digest(manifest_path),
              "features_sha256": experiment.digest(tmp_path / "test_features.csv")}
    (tmp_path / "sealed_manifest.json").write_text(json.dumps(sealed))
    (tmp_path / "evaluation_started.json").write_text("already consumed")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "PLAN", manifest_path)
    monkeypatch.setattr(experiment, "commit", lambda: plan)
    model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda _: pytest.fail("Must not rescore"))
    monkeypatch.setattr(experiment, "load_model", lambda *args, **kwargs: {
        "model": model, "trained_through": "2020-01-01", "feature_cols": ["x"]})
    with pytest.raises(FileExistsError):
        experiment.evaluate_once()
