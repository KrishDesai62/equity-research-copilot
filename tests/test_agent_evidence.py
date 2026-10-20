from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest

from src.agents_full.bull_agent import BullAgent
from src.agents_full.bear_agent import BearAgent
from src.agents_full.evidence import filing_evidence
from src.agents_full.real_time_search_agent import NewsMacroAgent


@pytest.mark.parametrize("agent,method", [(BullAgent, "analyze_bull_case"), (BearAgent, "analyze_bear_case")])
def test_agents_abstain_without_evidence_and_accept_missing_scores(agent, method, monkeypatch):
    monkeypatch.setattr("src.agents_full.evidence.retrieve", lambda *a, **kw: [])
    result = getattr(agent(), method)("TEST", "0000000001", as_of_date="2020-01-01")
    assert result["evidence_status"] == "no_cited_evidence"
    assert result["recommendation"] == "ABSTAIN"
    assert not result["citations"]
    assert "compelling upside" not in result["thesis_summary"]
    assert "severe vulnerability" not in result["thesis_summary"]


def test_evidence_rejects_future_wrong_company_and_uncited_excerpts(monkeypatch):
    good = {"text": "source passage", "filing_date": "2019-01-01", "metadata": {
        "source_url": "https://www.sec.gov/source", "accession_number": "accession", "cik": "0000000001"}}
    rows = [good, {**good, "filing_date": "2021-01-01"},
            {**good, "metadata": {**good["metadata"], "cik": "0000000002"}},
            {**good, "metadata": {}}]
    def retrieve(*args, **kwargs):
        assert kwargs["date_range"][1] == "2020-01-01"
        return rows
    monkeypatch.setattr("src.agents_full.evidence.retrieve", retrieve)
    result = filing_evidence("growth", "TEST", "1", 5, "2020-01-01")
    assert len(result) == 1 and result[0]["source_url"] == good["metadata"]["source_url"]


def test_news_keeps_dated_citations_and_drops_future_or_undated_snippets(monkeypatch):
    class Search:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def news(self, *args, **kwargs):
            return [{"title": "News", "body": "Reported fact", "url": "https://example.org/news", "date": "2020-01-01T12:00:00Z"},
                {"title": "Future", "url": "https://example.org/future", "date": "2099-01-01T00:00:00Z"},
                {"title": "Undated", "url": "https://example.org/undated"}]
    monkeypatch.setitem(sys.modules, "duckduckgo_search", SimpleNamespace(DDGS=Search))
    result = NewsMacroAgent().fetch_live_market_intelligence("TEST", "Test company")
    assert len(result["articles"]) == 1
    assert result["articles"][0]["source_url"] == "https://example.org/news"
    assert datetime.fromisoformat(result["articles"][0]["retrieved_at"]).tzinfo == timezone.utc
    assert result["macro_sentiment_score"] is None


def test_news_failure_never_counts_error_message_as_article(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)
    result = NewsMacroAgent().fetch_live_market_intelligence("TEST", "Test company")
    assert result["search_status"] == "unavailable"
    assert result["articles"] == [] and result["live_headlines"] == []
