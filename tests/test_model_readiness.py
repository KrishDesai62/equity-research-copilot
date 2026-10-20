import numpy as np
import pandas as pd
import pytest

from src.ml.financial_history import extract_history
from src.validation.model_readiness import reviewed_labels


def test_unreconciled_input_cannot_contaminate_later_quarter_features():
    periods = [("a", "2020-03-31", "2020-05-01", 10),
               ("bad", "2020-06-30", "2020-08-01", 999),
               ("c", "2020-09-30", "2020-11-01", 60)]
    assets, revenues = [], []
    for accession, end, filed, value in periods:
        base = {"accn": accession, "end": end, "filed": filed, "form": "10-Q"}
        assets.append({**base, "val": 100})
        revenues.append({**base, "start": "2020-01-01", "val": value})
    payload = {"cik": 1, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": assets}}, "Revenues": {"units": {"USD": revenues}}}}}
    metadata = [{"cik": "0000000001", "accession_number": a, "filing_date": filed,
                 "filing_type": "10-Q/A" if a == "bad" else "10-Q"}
                for a, _, filed, _ in periods]
    original, _, _ = extract_history(payload, "TEST")
    corrected, cells, issues = extract_history(payload, "TEST", public_filings=metadata)
    assert original.Revenues_quarter.iloc[-1] == -939
    assert np.isnan(corrected.Revenues_quarter.iloc[-1])
    assert "bad" not in corrected.accession_number.tolist()
    assert all(c["accn"] != "bad" for c in cells)
    assert any(i["accession"] == "bad" for i in issues)


def coverage(reviewed=False):
    return pd.DataFrame([{"cik": "0000000001", "event_coverage_reviewed": reviewed,
        "coverage_start": "2020-01-01", "observed_through": "2024-12-31",
        "reviewer": "Analyst", "coverage_source_url": "https://example.org/verified-registry"}])


def test_unreviewed_followup_does_not_become_negative():
    frame = pd.DataFrame({"cik": ["0000000001"], "filing_date": ["2021-01-01"], "removed_date": [pd.NaT]})
    unknown = reviewed_labels(frame, coverage(), [], "2024-12-31")
    assert unknown.target_distressed_1yr.isna().all()
    reviewed = reviewed_labels(frame, coverage(True), [], "2024-12-31")
    assert reviewed.target_distressed_1yr.iloc[0] == 0
    assert reviewed.label_available_date_1yr.iloc[0] == pd.Timestamp("2022-01-01")


def test_review_requires_evidence_and_does_not_override_censoring():
    frame = pd.DataFrame({"cik": ["0000000001"], "filing_date": ["2024-06-01"], "removed_date": [pd.NaT]})
    incomplete = coverage(True)
    incomplete.loc[0, "reviewer"] = None
    with pytest.raises(ValueError, match="reviewer"):
        reviewed_labels(frame, incomplete, [], "2024-12-31")
    mature = reviewed_labels(frame, coverage(True), [], "2024-12-31")
    assert mature.target_distressed_1yr.isna().all()


def test_public_date_conflict_is_removed_before_history_calculation():
    fact = {"accn": "a", "end": "2020-12-31", "filed": "2021-03-02", "form": "10-K", "val": 100}
    payload = {"cik": 1, "facts": {"us-gaap": {"Assets": {"units": {"USD": [fact]}}}}}
    with pytest.raises(ValueError, match="No original"):
        extract_history(payload, "TEST", public_filings=[{"cik": "0000000001", "accession_number": "a",
            "filing_date": "2021-03-01", "filing_type": "10-K"}])
