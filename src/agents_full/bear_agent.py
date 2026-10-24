"""Collect candidate risk evidence without inventing a negative-return forecast."""
import logging
from src.agents_full.evidence import filing_evidence, finite_number

logger = logging.getLogger(__name__)


class BearAgent:
    def __init__(self, k_results=5, *, retriever=None):
        self.k_results = k_results
        self.retriever = retriever

    def analyze_bear_case(self, ticker, cik, ml_distress_probability=None, altman_z=None,
                          beneish_m=None, *, as_of_date=None):
        status = "available"
        try:
            evidence = filing_evidence(
                f"{ticker} material risks liquidity regulatory bankruptcy", ticker, cik,
                self.k_results, as_of_date, retriever=self.retriever)
            if not evidence:
                status = "no_cited_evidence"
        except Exception:
            logger.exception("Risk evidence retrieval failed for %s", ticker)
            evidence, status = [], "retrieval_unavailable"
        observations = []
        for name, value in (("Distress", ml_distress_probability), ("Altman Z", altman_z), ("Beneish M", beneish_m)):
            number = finite_number(value)
            if number is not None and (name != "Distress" or 0 <= number <= 1):
                observations.append(f"Supplied {name} score: {number:.4f}; validation not established by this agent.")
        return {"ticker": ticker, "cik": cik, "as_of_date": as_of_date,
            "quantitative_warnings": observations,
            "retrieved_risk_evidence": [e["text"] for e in evidence], "citations": evidence,
            "evidence_status": status, "recommendation": "ABSTAIN",
            "thesis_summary": f"{ticker}: {len(evidence)} cited candidate risk excerpts. "
                "Disclosures alone do not establish bankruptcy or negative investment returns."}
