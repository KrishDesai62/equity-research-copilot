import copy

import numpy as np
import pandas as pd
import pytest

from src.ml.calibration import check_partitions, fit_candidate, select_threshold
from src.ml.train_distress import train_logistic_baseline
from src.validation.calibration_development import partition


def parts():
    result = {}
    for i, name in enumerate(("training", "calibration", "threshold_tuning")):
        year = 2010 + i * 2
        result[name] = pd.DataFrame({"cik": [str(100 * i + j).zfill(10) for j in range(8)],
            "filing_date": [f"{year}-01-01"] * 8,
            "label_available_date_1yr": [f"{year+1}-01-01"] * 8,
            "x": [-4., -3., -2., -1., 1., 2., 3., 4.],
            "target": [0, 0, 0, 0, 1, 1, 1, 1]})
    return result


def test_threshold_selection_avoids_default_all_negative_and_never_claims_final_accuracy():
    selection = select_threshold([0, 0, 1, 1], [.01, .02, .1, .2])
    assert selection["threshold"] == .1
    assert selection["selection_metrics"]["recall"] == 1
    assert "not independent" in selection["note"]
    blocked = select_threshold([1, 1, 0, 0], [.01, .02, .1, .2], min_precision=.9)
    assert blocked["threshold"] is None


def test_calibration_and_threshold_partitions_cannot_leak_companies_or_outcomes():
    data = parts()
    check_partitions(data, ["x"], "target")
    check_partitions(dict(reversed(list(data.items()))), ["x"], "target")
    overlap = copy.deepcopy(data)
    overlap["calibration"].loc[0, "cik"] = overlap["training"].iloc[0].cik
    with pytest.raises(ValueError, match="overlap"):
        check_partitions(overlap, ["x"], "target")
    data["training"]["label_available_date_1yr"] = "2013-01-01"
    with pytest.raises(ValueError, match="not known"):
        check_partitions(data, ["x"], "target")


def test_frozen_calibration_does_not_refit_base_preprocessing():
    data = parts()
    models = []
    def trainer(frame, features, target):
        model = train_logistic_baseline(frame, features, target)
        models.append((model, model.named_steps["classifier"].coef_.copy()))
        return model
    model, report = fit_candidate(data, ["x"], "target", trainer)
    assert report["calibration_fitted"] and not report["validated_calibration"]
    assert not report["production_ready"]
    np.testing.assert_array_equal(models[0][0].named_steps["classifier"].coef_, models[0][1])
    assert np.isfinite(model.predict_proba(data["threshold_tuning"][["x"]])).all()


def test_partition_is_deterministic_and_purges_unobserved_outcomes():
    import json
    from pathlib import Path
    policy = json.loads(Path("config/calibration_development.json").read_text())
    data = pd.concat(parts().values(), ignore_index=True)
    data["label_available_date_1yr"] = "2099-01-01"
    assert all(frame.empty for frame in partition(data, policy).values())


def test_company_assignment_is_stable_and_separate_from_row_order():
    import json
    from pathlib import Path
    policy = json.loads(Path("config/calibration_development.json").read_text())
    rows = [{"cik": str(company).zfill(10), "filing_date": f"{year}-01-01",
             "label_available_date_1yr": f"{year+1}-01-01"}
            for company in range(100) for year in (2013, 2016, 2018)]
    data = pd.DataFrame(rows)
    first, shuffled = partition(data, policy), partition(data.sample(frac=1, random_state=42), policy)
    sets = []
    for name, frame in first.items():
        assert set(frame.cik) == set(shuffled[name].cik)
        assert not any(set(frame.cik) & existing for existing in sets)
        sets.append(set(frame.cik))


def test_concurrent_candidate_run_cannot_remove_owner_lock(tmp_path):
    from src.validation.calibration_development import run
    lock = tmp_path / ".candidate_run.lock"
    lock.write_text("other run")
    with pytest.raises(FileExistsError):
        run(str(tmp_path))
    assert lock.read_text() == "other run"
