import io

import numpy as np
import pandas as pd
import pytest

from src.data_layer.bankruptcy_labels import build_distress_labels
from src.ml.evaluate import company_walk_forward_splits
from src.ml.features_distress import compute_altman_components
from src.ml.train_distress import parse_arff_to_dataframe, run_walk_forward_evaluation
from src.validation.audit_filings import InlineFacts
from src.validation.sec_pilot import extract_annual_rows, label_rows


def panel():
    dates = pd.date_range("2018-01-01", periods=12, freq="QS")
    return pd.DataFrame([{"cik": str(company), "filing_date": date,
                          "available": date + pd.Timedelta(days=100), "x": company % 2,
                          "y": company % 2} for date in dates for company in range(6)])


def test_recurring_panel_does_not_erase_all_training_companies():
    frame = panel().sample(frac=1, random_state=4).reset_index(drop=True)
    splits = list(company_walk_forward_splits(frame, label_available_col="available"))
    assert splits
    positions = []
    for _, train_idx, test_idx in splits:
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        assert not train.empty
        assert set(train.cik).isdisjoint(test.cik)
        assert train.filing_date.max() < test.filing_date.min()
        assert train.available.max() < test.filing_date.min()
        positions.extend(test_idx)
    assert len(positions) == len(set(positions))


def test_recurring_panel_evaluates_and_reports_skipped_folds():
    report = run_walk_forward_evaluation(panel(), ["x"], "y", label_available_col="available")
    assert report["pooled_metrics"]["pr_auc"] > .9
    assert report["total_folds_evaluated"] > 0


def test_no_valid_training_window_raises_instead_of_returning_empty_report():
    frame = panel()
    frame["available"] = pd.Timestamp("2040-01-01")
    with pytest.raises(ValueError, match="No usable"):
        run_walk_forward_evaluation(frame, ["x"], "y", label_available_col="available")


def test_split_rejects_outcome_available_before_feature():
    frame = panel()
    frame.loc[0, "available"] = pd.Timestamp("2000-01-01")
    with pytest.raises(ValueError, match="cannot precede"):
        list(company_walk_forward_splits(frame, label_available_col="available"))


def base_universe():
    return pd.DataFrame([{"cik": "1", "filing_date": "2020-02-29", "removed_date": None,
                           "coverage_start": "2010-01-01", "observed_through": "2022-01-01",
                           "event_coverage_reviewed": True}])


def events(date=None):
    return pd.DataFrame([{"cik": "1", "filing_date": date, "source_url": "https://example.org/event"}]) if date else pd.DataFrame(columns=["cik", "filing_date", "source_url"])


def test_calendar_year_negative_requires_full_observed_horizon():
    df = base_universe()
    labels = build_distress_labels(df, events(), horizon_years=(1,))
    assert labels.target_distressed_1yr.tolist() == [0]
    assert labels.label_available_date_1yr.iloc[0] == pd.Timestamp("2021-02-28")
    df["observed_through"] = "2021-02-27"
    assert build_distress_labels(df, events(), horizon_years=(1,)).empty


def test_absence_of_bankruptcy_record_does_not_prove_negative():
    df = base_universe().drop(columns="event_coverage_reviewed")
    assert build_distress_labels(df, events(), horizon_years=(1,)).empty


def test_positive_event_and_post_event_exclusion():
    df = base_universe()
    labels = build_distress_labels(df, events("2020-05-01"), horizon_years=(1,))
    assert labels.target_distressed_1yr.tolist() == [1]
    assert labels.label_available_date_1yr.iloc[0] == pd.Timestamp("2020-05-01")
    df["filing_date"] = "2020-06-01"
    assert build_distress_labels(df, events("2020-05-01"), horizon_years=(1,)).empty


def test_fiscal_period_end_is_not_accepted_as_label_anchor():
    with pytest.raises(ValueError, match="filing_date"):
        build_distress_labels(base_universe().rename(columns={"filing_date": "date"}), events())


def test_modern_revenue_tag_and_nan_fallback():
    result = compute_altman_components({"Assets": 100, "Revenues": np.nan,
        "RevenueFromContractWithCustomerExcludingAssessedTax": 80})
    assert result["sales_to_ta"] == .8


def test_sec_extraction_does_not_use_later_restatements_or_comparative_period():
    acc = "0000000001-20-000001"
    def fact(value, end="2019-12-31", **kwargs):
        return dict(accn=acc, filed="2020-03-01", form="10-K", end=end, val=value, **kwargs)
    payload = {"cik": 1, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [fact(100), fact(80, end="2018-12-31")]}},
        "Revenues": {"units": {"USD": [fact(50, start="2019-01-01"),
                {**fact(999, start="2019-01-01"), "accn": "later-amendment", "form": "10-K/A", "filed": "2020-04-01"}]}}}}}
    frame, provenance, issues = extract_annual_rows(payload, "TEST")
    assert frame.Assets.iloc[0] == 100
    assert frame.Revenues.iloc[0] == 50
    assert frame.sales_to_ta.iloc[0] == .5
    assert all(cell["accn"] == acc for cell in provenance)


def test_pilot_provisional_negatives_not_eligible_for_production():
    df = pd.DataFrame({"filing_date": ["2020-01-01", "2024-06-01"]})
    result = label_rows(df, {"event_date": None}, "2024-12-31")
    assert not result.valid_for_training.any()
    assert result.target_distressed_1yr.iloc[0] == 0
    assert pd.isna(result.target_distressed_1yr.iloc[1])


def test_inline_audit_reads_native_numeric_and_scaled_negative_values():
    parser = InlineFacts()
    parser.feed('<xbrli:context id="c"><xbrli:period><xbrli:instant>2020-01-01</xbrli:instant></xbrli:period></xbrli:context>'
                '<ix:nonFraction name="us-gaap:Assets" contextRef="c" unitRef="usd" scale="6" sign="-">214</ix:nonFraction>')
    assert parser.contexts["c"]["instant"] == "2020-01-01"
    assert parser.facts[0]["value"] == -214000000


def test_arff_parser_converts_nominal_binary_target():
    content = b'@relation test\n@attribute x numeric\n@attribute class {0,1}\n@data\n1,0\n?,1\n'
    frame = parse_arff_to_dataframe(io.BytesIO(content))
    assert frame["class"].tolist() == [0, 1]
    assert pd.isna(frame.x.iloc[1])


def test_entirely_missing_features_cannot_be_imputed_into_fake_inputs():
    from src.ml.neural_network import validate_training_data
    frame = pd.DataFrame({"market_cap_ratio": [np.nan, np.nan], "target": [0, 1]})
    with pytest.raises(ValueError, match="no observed"):
        validate_training_data(frame, ["market_cap_ratio"], "target")


def test_unreviewed_pilot_cannot_create_production_artifact(tmp_path, monkeypatch):
    from src.ml.train import main
    frame = pd.DataFrame({"cik": ["1", "2"], "filing_date": ["2020-01-01"]*2,
        "available": ["2021-01-01"]*2, "label_review_status": ["coverage_review_required"]*2,
        "x": [0, 1], "target": [0, 1]})
    source = tmp_path / "pilot.csv"
    frame.to_csv(source, index=False)
    monkeypatch.setattr("sys.argv", ["train", str(source), "--track", "distress", "--target", "target",
        "--features", "x", "--label-available-col", "available", "--cutoff", "2022-01-01",
        "--output-dir", str(tmp_path / "models")])
    with pytest.raises(ValueError, match="Unreviewed"):
        main()
    assert not (tmp_path / "models/distress.joblib").exists()


def test_event_after_observation_end_cannot_become_positive():
    df = base_universe()
    df["observed_through"] = "2020-03-01"
    assert build_distress_labels(df, events("2020-05-01"), horizon_years=(1,)).empty
