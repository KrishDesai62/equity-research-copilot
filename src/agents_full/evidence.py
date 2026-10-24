"""Cited, dated research excerpts; retrieval is not a directional conclusion."""
from datetime import date, datetime, timezone
import math

from src.rag.retrieval import retrieve


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def filing_evidence(query, ticker, cik, k, as_of_date=None, *, retriever=None):
    cutoff = date.fromisoformat(as_of_date) if as_of_date else datetime.now(timezone.utc).date()
    chunks = (retriever or retrieve)(query, ticker=ticker, filing_type=None, k=k,
                      date_range=("1900-01-01", cutoff.isoformat()))
    evidence = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        published = chunk.get("filing_date") or meta.get("filing_date")
        source = meta.get("source_url")
        accession = meta.get("accession_number")
        source_cik = meta.get("cik")
        if not published or not source or not accession or source_cik is None:
            continue
        if date.fromisoformat(published) > cutoff or str(source_cik).zfill(10) != str(cik).zfill(10):
            continue
        if not str(source).startswith(("https://", "http://")) or not chunk.get("text", "").strip():
            continue
        evidence.append({"text": chunk["text"], "source_url": source,
            "accession_number": accession, "filing_date": published,
            "section": chunk.get("section"), "cik": str(source_cik).zfill(10)})
    return evidence
