import pandas as pd
from src.ml.features_distress import build_feature_table, add_rag_sentiment_feature


def test_feature_table_uses_filing_date_not_period_end():
    xbrl = pd.DataFrame([dict(cik="1", ticker="A", fiscal_quarter="2020Q1",
                             filing_date="2020-05-01", Assets=100)])
    labels = pd.DataFrame([dict(cik="1", fiscal_quarter="2020Q1", event_date="2020-04-01")])
    assert build_feature_table(xbrl, labels).empty


def test_no_future_filing_leaks_into_past_quarter_feature():
    def retrieve(**kwargs):
        assert kwargs["date_range"][1] == "2020-05-01"
        return [{"filing_date": "2020-05-02", "text": "bankruptcy"},
                {"filing_date": "2020-04-01", "text": "steady growth"}]
    row = add_rag_sentiment_feature(dict(ticker="A", filing_date="2020-05-01"), retrieve)
    assert row["risk_sentiment_score"] == 0
