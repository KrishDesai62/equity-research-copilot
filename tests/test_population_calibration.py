import numpy as np
import pandas as pd
import pytest

from src.validation.population_calibration import calibration_metrics, select_sample


def test_population_sampling_ignores_outcomes_and_row_order():
    universe = pd.DataFrame({"cik": list(range(100)), "form": ["10-K"] * 100,
        "sic": [3500] * 100, "outcome": [0] * 100})
    first = select_sample(universe, "fixed-seed", 20)
    changed = universe.sample(frac=1, random_state=42).assign(outcome=1)
    assert first.cik.tolist() == select_sample(changed, "fixed-seed", 20).cik.tolist()


def test_population_frame_excludes_financial_and_nonannual_filings():
    universe = pd.DataFrame({"cik": [1, 2, 3, 4], "form": ["10-K", "10-Q", "10-K", "10-K"],
        "sic": [3500, 3500, 6000, None]})
    assert select_sample(universe, "seed", 10).cik.tolist() == ["0000000001"]


def test_calibration_accepts_zero_events_without_inventing_accuracy():
    result = calibration_metrics([0, 0, 0], [.1, .2, .3])
    assert result["observed_events"] == 0
    assert result["mean_score"] == pytest.approx(.2)
    assert result["brier_loss"] == pytest.approx((.01 + .04 + .09) / 3)
    assert np.isfinite(result["log_loss"])
    assert sum(b["companies"] for b in result["bins"]) == 3


@pytest.mark.parametrize("labels,scores", [([], []), ([.5, 1], [.2, .3]), ([0, 1], [.2, np.nan]), ([0], [1.1])])
def test_invalid_calibration_data_is_rejected(labels, scores):
    with pytest.raises(ValueError):
        calibration_metrics(labels, scores)


def test_impossible_fiscal_dates_are_quarantined_before_history_features():
    from src.ml.financial_history import extract_history
    values = [{"accn": "good", "end": "2019-12-31", "filed": "2020-03-01", "form": "10-K", "val": 100},
              {"accn": "bad", "end": "2030-12-31", "filed": "2020-03-02", "form": "10-K", "val": 999}]
    payload = {"cik": 1, "facts": {"us-gaap": {"Assets": {"units": {"USD": values}}}}}
    frame, provenance, issues = extract_history(payload, "TEST")
    assert frame.accession_number.tolist() == ["good"]
    assert all(cell["accn"] != "bad" for cell in provenance)
    assert any(issue["accession"] == "bad" and "quarantined" in issue["reason"] for issue in issues)
