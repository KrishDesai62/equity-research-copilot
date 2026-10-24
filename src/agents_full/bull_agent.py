"""Collect candidate growth evidence without inventing an investment conclusion."""
import logging
from src.agents_full.evidence import filing_evidence, finite_number

logger = logging.getLogger(__name__)


class BullAgent:
    def __init__(self, k_results=5, *, retriever=None):
        self.k_results = k_results
        self.retriever = retriever

    def analyze_bull_case(self, ticker, cik, ml_distress_probability=None, altman_z=None, *, as_of_date=None):
        status = "available"
        try:
            evidence = filing_evidence(
                f"{ticker} growth expansion revenue drivers profitability", ticker, cik,
                self.k_results, as_of_date, retriever=self.retriever)
            if not evidence:
                status = "no_cited_evidence"
        except Exception:
            logger.exception("Growth evidence retrieval failed for %s", ticker)
            evidence, status = [], "retrieval_unavailable"
        observations = []
        probability, z = finite_number(ml_distress_probability), finite_number(altman_z)
        if probability is not None and 0 <= probability <= 1:
            observations.append(f"Supplied distress score: {probability:.4f}; validation not established by this agent.")
        if z is not None:
            observations.append(f"Supplied Altman Z score: {z:.2f}; applicability requires review.")
        return {"ticker": ticker, "cik": cik, "as_of_date": as_of_date,
            "quantitative_strengths": observations,
            "retrieved_growth_evidence": [e["text"] for e in evidence], "citations": evidence,
            "evidence_status": status, "recommendation": "ABSTAIN",
            "thesis_summary": f"{ticker}: {len(evidence)} cited candidate growth excerpts. "
                "Relevance and investment implications require review; no upside conclusion is validated."}
