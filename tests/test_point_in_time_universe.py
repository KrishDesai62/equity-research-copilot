"""
Run this before trusting any Track A results. If it fails, your
distress classifier has no positive labels to learn from — see README
fix #1.
"""
import pytest


def test_universe_includes_historically_delisted_companies():
    """
    TODO: load your MVP universe (config/tickers_sp500.yaml or the
    output of point_in_time_universe.write_mvp_universe) and assert it
    contains at least one company that is NOT in today's S&P 500 because
    it was delisted/went bankrupt. If this fails, you've built a
    survivorship-biased universe — go back to point_in_time_universe.py.
    """
    raise NotImplementedError


def test_universe_as_of_date_excludes_future_information():
    """
    TODO: call universe_as_of() for some historical date T and assert
    no company that was added to the index AFTER T appears in the
    result.
    """
    raise NotImplementedError
