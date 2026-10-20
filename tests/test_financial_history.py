from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.ml.financial_history import FLOW_TAGS, add_history_features, extract_history
from src.validation.improve import commit_split, choose_threshold, company_bootstrap_delta


def history_row(end, filed, start, revenue, cashflow, *, form="10-Q", cik="1"):
    row = {"cik": cik, "ticker": "TEST", "filing_date": filed, "fiscal_period_end": end,
           "filing_type": form, "accession_number": filed,
           "Assets": 100, "AssetsCurrent": 40, "LiabilitiesCurrent": 20,
           "Liabilities": 60, "RetainedEarningsAccumulatedDeficit": 10, "Cash": 5}
    for field in FLOW_TAGS:
        row[field] = revenue if field == "Revenues" else cashflow
        row[field + "_start"] = start
    return row


def fiscal_year():
    return [history_row("2020-03-31", "2020-05-01", "2020-01-01", 10, 1),
            history_row("2020-06-30", "2020-08-01", "2020-01-01", 30, 3),
            history_row("2020-09-30", "2020-11-01", "2020-01-01", 60, 6),
            history_row("2020-12-31", "2021-03-01", "2020-01-01", 100, 10, form="10-K")]


def test_ytd_cash_flows_are_subtracted_before_trailing_year_sum():
    rows = fiscal_year() + [history_row("2021-03-31", "2021-05-01", "2021-01-01", 20, 2)]
    frame = add_history_features(pd.DataFrame(rows))
    assert frame.OperatingCashFlow_quarter.tolist() == [1, 2, 3, 4, 2]
    assert frame.OperatingCashFlow_ttm.iloc[-1] == 11
    assert frame.Revenues_ttm.iloc[-1] == 110
    assert frame.cashflow_to_assets.iloc[-1] == .11


def test_missing_quarter_prevents_fake_ttm():
    rows = fiscal_year()
    frame = add_history_features(pd.DataFrame([rows[0], rows[2], rows[3],
        history_row("2021-03-31", "2021-05-01", "2021-01-01", 20, 2)]))
    assert pd.isna(frame.OperatingCashFlow_ttm.iloc[-1])


def test_first_annual_flow_is_not_treated_as_one_quarter():
    frame = add_history_features(pd.DataFrame([fiscal_year()[-1]]))
    assert frame.Revenues_ttm.iloc[0] == 100
    assert pd.isna(frame.Revenues_quarter.iloc[0])


def test_short_reorganization_stub_is_not_a_full_quarter_or_annual_ttm():
    rows = fiscal_year()[:3] + [history_row("2020-12-31", "2021-03-01", "2020-03-01", 80, 8, form="10-K")]
    frame = add_history_features(pd.DataFrame(rows))
    assert pd.isna(frame.Revenues_ttm.iloc[-1])
    stub = history_row("2021-06-30", "2021-08-01", "2021-05-01", 20, 2)
    result = add_history_features(pd.DataFrame(fiscal_year() + [stub]))
    assert pd.isna(result.Revenues_quarter.iloc[-1])
    assert pd.isna(result.Revenues_ttm.iloc[-1])


def test_future_filings_do_not_change_historical_features():
    old = add_history_features(pd.DataFrame(fiscal_year()))
    later = history_row("2021-03-31", "2021-05-01", "2021-01-01", 9999, 9999)
    new = add_history_features(pd.DataFrame([later] + fiscal_year()))
    pd.testing.assert_frame_equal(old, new.iloc[:4].reset_index(drop=True))
    assert new.latest_input_filing_date.le(new.filing_date).all()


def test_histories_do_not_mix_companies():
    one = fiscal_year()
    other = history_row("2021-03-31", "2021-05-01", "2021-01-01", 20, 2, cik="2")
    frame = add_history_features(pd.DataFrame(one + [other]))
    assert pd.isna(frame.loc[frame.cik.eq("2"), "Revenues_ttm"].iloc[0])


def test_proxy_fallback_is_explicit_not_mislabeled_operating_income():
    row = fiscal_year()[-1]
    row["OperatingIncomeLoss"] = np.nan
    row["PretaxIncome"] = 5
    row["InterestExpense"] = 2
    frame = add_history_features(pd.DataFrame([row]))
    assert pd.isna(frame.ebit_to_ta.iloc[0])
    assert frame.earnings_proxy_to_assets.iloc[0] == .07
    assert frame.earnings_proxy_method.iloc[0] == "pretax_plus_interest"


def test_year_over_year_features_require_a_previous_year_not_previous_quarter():
    rows = fiscal_year() + [history_row("2021-03-31", "2021-05-01", "2021-01-01", 20, 2),
                           history_row("2021-12-31", "2022-03-01", "2021-01-01", 120, 12, form="10-K")]
    frame = add_history_features(pd.DataFrame(rows))
    assert frame.revenue_growth_yoy.iloc[-1] == pytest.approx(.2)


def test_extraction_ignores_amendment_values_and_comparative_assets():
    def fact(value, **kwargs):
        return {"accn": "a", "end": "2020-03-31", "filed": "2020-05-01", "form": "10-Q", "val": value, **kwargs}
    payload = {"cik": 1, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [fact(100), fact(80, end="2019-12-31")]}},
        "Revenues": {"units": {"USD": [fact(20, start="2020-01-01"),
                fact(9999, start="2020-01-01", accn="amendment", form="10-Q/A", filed="2020-06-01")]}}}}}
    frame, cells, _ = extract_history(payload, "TEST")
    assert len(frame) == 1
    assert frame.Revenues.iloc[0] == 20
    assert frame.Assets.iloc[0] == 100
    assert all(cell["filed"] == "2020-05-01" for cell in cells)


def test_committed_holdout_does_not_depend_on_labels_and_cannot_silently_change(tmp_path):
    config = {"companies": [{"cik": str(i), "event_date": None} for i in range(12)],
              "holdout_policy": {"seed": 42, "fraction": .25}}
    path = tmp_path / "split.json"
    first = commit_split(config, path)
    altered = deepcopy(config)
    for company in altered["companies"]:
        company["event_date"] = "2020-01-01"
    assert commit_split(altered, path) == first
    altered["holdout_policy"]["seed"] = 43
    with pytest.raises(ValueError, match="Committed holdout differs"):
        commit_split(altered, path)


def test_threshold_selection_uses_supplied_development_predictions():
    result = choose_threshold([0, 0, 1, 1], [.1, .6, .7, .9], minimum_precision=.8)
    assert result["development_precision"] == 1
    assert result["development_recall"] == 1
    assert result["threshold"] == .7


def test_paired_company_bootstrap_identical_models_have_zero_difference():
    frame = pd.DataFrame({"cik": ["1", "1", "2", "2", "3", "3"],
                          "target_distressed_1yr": [0, 0, 0, 1, 0, 1]})
    scores = np.array([.1, .2, .3, .8, .2, .9])
    result = company_bootstrap_delta(frame, scores, scores, repetitions=30)
    assert result["valid_resamples"] > 0
    assert result["pr_auc_difference_95pct_interval"] == [0, 0]


def test_completed_holdout_cannot_be_silently_retuned(tmp_path):
    from src.validation.improve import experiment
    (tmp_path / "experiment.json").write_text("{}")
    with pytest.raises(ValueError, match="already been evaluated"):
        experiment(pd.DataFrame(), {}, {}, tmp_path)


def test_independent_ttm_identity_flags_corrupted_feature():
    from src.validation.flow_audit import audit_trailing_flows
    frame = add_history_features(pd.DataFrame(fiscal_year() + [
        history_row("2021-03-31", "2021-05-01", "2021-01-01", 20, 2)]))
    result = audit_trailing_flows(frame)
    assert result["check_count"] > 0
    assert result["mismatches"] == 0
    frame.loc[frame.index[-1], "OperatingCashFlow_ttm"] = 999
    assert audit_trailing_flows(frame)["mismatches"] == 1
