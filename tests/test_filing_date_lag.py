"""
Run this before trusting any ML feature table. Catches the most common
leakage bug: joining on fiscal_period_end instead of filing_date.
"""
import pytest


def test_feature_table_uses_filing_date_not_period_end():
    """
    TODO: build a small synthetic xbrl_df with a known filing_date
    later than fiscal_period_end (this is realistic -- filings lag
    period end by 40-90 days). Run it through
    ml.features_distress.build_feature_table and assert that any join
    to labels or retrieved text used filing_date, not period_end -- e.g.
    by checking no chunk with a filing_date after the row's filing_date
    was retrieved for it.
    """
    raise NotImplementedError


def test_no_future_filing_leaks_into_past_quarter_feature():
    """
    TODO: end-to-end check -- for a given (cik, fiscal_quarter) row,
    confirm every piece of data used to build its features (XBRL values,
    RAG-retrieved sentiment, insider trading aggregates) has a
    filing_date <= that quarter's filing_date.
    """
    raise NotImplementedError
