"""
MVP smoke test for the retrieval backbone.
"""
import pytest


def test_retrieve_filters_by_ticker():
    """
    TODO: after ingesting at least 2 different tickers' filings, confirm
    retrieve(query, ticker="X") never returns chunks for ticker "Y".
    """
    raise NotImplementedError


def test_retrieve_returns_expected_metadata_shape():
    """
    TODO: confirm each result dict has the keys retrieval.py's docstring
    promises (text, ticker, filing_type, section, filing_date,
    similarity_score) -- agents built later depend on this exact shape.
    """
    raise NotImplementedError
