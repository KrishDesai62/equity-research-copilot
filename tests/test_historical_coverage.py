from hashlib import sha256
import json

import pandas as pd
import pytest

from src.validation.historical_coverage import apply_historical_coverage
from src.ml.train_distress import train_logistic_baseline


def inputs(assets=1_000_000_000):
    frame = pd.DataFrame({"cik": ["0000000001", "0000000001"], "filing_type": ["10-K", "10-Q"],
        "filing_date": ["2020-03-01", "2020-05-01"], "fiscal_period_end": ["2019-12-31", "2020-03-31"],
        "accession_number": ["annual", "quarter"], "Assets": [assets, assets],
        "target_distressed_1yr": pd.Series([pd.NA, pd.NA], dtype="Int64"),
        "label_available_date_1yr": [pd.NaT, pd.NaT], "bankruptcy_label_conflict": [False, False]})
    cases = pd.DataFrame({"cik": ["0000000002"], "filing_date": ["2021-01-01"]})
    return frame, cases, frame.copy()


def test_negative_requires_registry_identity_and_annual_size_evidence():
    frame, cases, filings = inputs()
    reviewed, evidence = apply_historical_coverage(frame, cases, filings)
    assert reviewed.target_distressed_1yr.tolist() == [0, 0]
    assert reviewed.label_available_date_1yr.iloc[1] == pd.Timestamp("2021-05-01")
    assert len(evidence) == 2
    small, cases, filings = inputs(100_000_000)
    assert apply_historical_coverage(small, cases, filings)[0].target_distressed_1yr.isna().all()
    cases.loc[0, "cik"] = None
    with pytest.raises(ValueError, match="identities"):
        apply_historical_coverage(frame, cases, filings)


def test_missing_annual_does_not_authorize_a_healthy_label():
    frame, cases, filings = inputs()
    extra = filings.iloc[[0]].copy()
    extra["accession_number"] = "unextracted_annual"
    extra["filing_date"] = "2021-01-01"
    filings = pd.concat([filings, extra], ignore_index=True)
    assert apply_historical_coverage(frame, cases, filings)[0].target_distressed_1yr.isna().all()


def test_event_conflicts_and_censored_horizons_never_become_negative():
    frame, cases, filings = inputs()
    frame.loc[0, "bankruptcy_label_conflict"] = True
    frame.loc[1, "filing_date"] = "2022-06-01"
    assert apply_historical_coverage(frame, cases, filings)[0].target_distressed_1yr.isna().all()
    frame, cases, filings = inputs()
    cases.loc[0, "cik"] = "0000000001"
    assert apply_historical_coverage(frame, cases, filings)[0].target_distressed_1yr.isna().all()


def test_preprocessing_is_fitted_only_on_training_rows():
    training = pd.DataFrame({"ratio": [1., 2., 3., 4., 5., 6., 7., None],
                            "target": [0, 0, 0, 0, 1, 1, 1, 1]})
    model = train_logistic_baseline(training, ["ratio"], "target")
    assert model.named_steps["imputer"].statistics_[0] == 4
    mean = model.named_steps["scaler"].mean_.copy()
    model.predict_proba(pd.DataFrame({"ratio": [1e9]}))
    assert model.named_steps["scaler"].mean_ == mean


def test_confirmed_positive_needs_followup_to_event_not_post_bankruptcy_annuals():
    frame, cases, filings = inputs()
    cases.loc[0, "cik"] = "0000000001"
    cases.loc[0, "filing_date"] = "2020-08-01"
    frame["target_distressed_1yr"] = 1
    frame["label_available_date_1yr"] = pd.Timestamp("2020-08-01")
    later = filings.iloc[[0]].assign(accession_number="post_bankruptcy_annual", filing_date="2021-01-01")
    reviewed, evidence = apply_historical_coverage(frame, cases, pd.concat([filings, later], ignore_index=True))
    assert reviewed.historical_coverage_eligible.all()
    assert evidence.review_end.eq("2020-08-01").all()
    assert not evidence.annual_accessions.str.contains("post_bankruptcy_annual").any()


def test_positive_cannot_skip_missing_annual_before_the_event():
    frame, cases, filings = inputs()
    cases.loc[0, "cik"] = "0000000001"
    cases.loc[0, "filing_date"] = "2020-08-01"
    frame["target_distressed_1yr"] = 1
    frame["label_available_date_1yr"] = pd.Timestamp("2020-08-01")
    missing = filings.iloc[[0]].assign(accession_number="missing_before_case", filing_date="2020-07-01")
    reviewed, _ = apply_historical_coverage(frame, cases, pd.concat([filings, missing], ignore_index=True))
    assert not reviewed.historical_coverage_eligible.any()


def test_source_observation_end_cannot_be_extended_by_a_later_registry_snapshot():
    frame, cases, filings = inputs()
    reviewed, _ = apply_historical_coverage(frame, cases, filings, observation_end="2020-12-31")
    assert reviewed.target_distressed_1yr.isna().all()


def test_saved_csv_date_columns_accept_reviewed_label_availability():
    frame, cases, filings = inputs()
    frame["label_available_date_1yr"] = pd.Series([None, None], dtype="str")
    reviewed, _ = apply_historical_coverage(frame, cases, filings)
    assert reviewed.label_available_date_1yr.iloc[0] == pd.Timestamp("2021-03-01")
