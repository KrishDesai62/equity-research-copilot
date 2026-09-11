"""
MVP agent: single Q&A over the retrieval backbone. This is the whole
"agent layer" for the MVP — no bull/bear/risk/synthesis debate yet
(that's agents_full/, Phase 4). The point of building this now is to
prove retrieve() produces good enough grounded answers before you invest
in a four-agent architecture on top of it.
"""
from src.rag.retrieval import retrieve


def answer_question(question: str, ticker: str) -> dict:
    """
    TODO:
      1. chunks = retrieve(question, ticker=ticker, k=5)
      2. build a prompt: the question + retrieved chunk texts, each
         labeled with its section/filing_date so the model can cite them
      3. call the Anthropic API (see anthropic_api_in_artifacts pattern,
         or just the SDK directly if running outside artifacts) with that
         prompt
      4. return {answer: str, citations: [{section, filing_date}, ...]}

    MVP test: ask "what changed in risk factors this year" for one of
    your MVP tickers and confirm the answer cites an actual retrieved
    section, not a hallucinated one.
    """
    raise NotImplementedError
