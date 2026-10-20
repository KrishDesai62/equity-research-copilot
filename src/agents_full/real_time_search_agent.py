"""
Phase 4 (post-MVP). Performs live web and news searches via a search tool 
(e.g., Tavily or DuckDuckGo) to fetch real-time macroeconomic catalysts, 
earnings sentiment, and breaking market headlines for a given ticker.
"""

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Any, List
# Using a standard lightweight search wrapper like duckduckgo_search or tavily-python

logger = logging.getLogger("NewsMacroAgent")

class NewsMacroAgent:
    """
    Autonomous agent responsible for querying live external feeds, breaking news, 
    and macroeconomic reports to evaluate real-time sentiment and external catalysts.
    """
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def fetch_live_market_intelligence(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """
        Executes a live web search query to gather recent news headlines, 
        regulatory developments, and macro sentiment shifts.
        """
        logger.info(f"News/Macro Agent querying live web feeds for {company_name} ({ticker})...")

        query = f"{company_name} {ticker} stock news earnings macroeconomic impact"
        headlines = []
        articles = []
        retrieved_at = datetime.now(timezone.utc)
        search_status = "available"

        try:
            from duckduckgo_search import DDGS
            # Execute search against live web data sources
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=self.max_results))
                
            for item in results:
                source_url = item.get("url", "")
                published = item.get("date")
                if not source_url.startswith(("https://", "http://")) or not published:
                    continue
                try:
                    try:
                        publication_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    except ValueError:
                        publication_date = parsedate_to_datetime(published)
                    # Missing timezones cannot establish when evidence was public.
                    if publication_date.tzinfo is None or publication_date > retrieved_at:
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
                title = item.get("title", "No Title")
                snippet = item.get("body", "No description available.")
                headlines.append(f"• {title}: {snippet}")
                articles.append({"title": title, "snippet": snippet, "source_url": source_url,
                    "published_at": publication_date.isoformat(), "retrieved_at": retrieved_at.isoformat(),
                    "source": item.get("source"), "evidence_type": "search_snippet"})

            logger.info(f"Successfully retrieved {len(headlines)} live news snippets for {ticker}.")

        except Exception as e:
            logger.error(f"Live search failed for {ticker}: {e}")
            search_status = "unavailable"

        macro_summary = (
            f"REAL-TIME INTELLIGENCE SUMMARY FOR {ticker}: "
            f"Retrieved {len(headlines)} search snippets. "
            f"Search status: {search_status}. Sentiment analysis is unavailable."
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "live_headlines": headlines,
            "articles": articles,
            "retrieved_at": retrieved_at.isoformat(),
            "search_status": search_status,
            "macro_sentiment_score": None,
            "sentiment_status": "unavailable_no_validated_sentiment_model",
            "summary": macro_summary
        }

if __name__ == "__main__":
    # Standalone verification block
    agent = NewsMacroAgent()
    intelligence = agent.fetch_live_market_intelligence("AAPL", "Apple Inc.")
    print(intelligence["summary"])
    for headline in intelligence["live_headlines"][:2]:
        print(headline)
