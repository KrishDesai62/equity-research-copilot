"""Grounded, extractive filing Q&A: cited passages without invented synthesis."""
from datetime import date
from src.rag.retrieval import retrieve


def answer_question(question: str, ticker: str, *, as_of=None, k=3, retriever=None) -> dict:
    if not question.strip() or not ticker.strip():
        raise ValueError("Question and ticker are required")
    if k < 1:
        raise ValueError("k must be positive")
    cutoff = date.fromisoformat(as_of) if as_of else date.today()
    from src.rag.query_relevance import expanded_query, rank_evidence
    rows = (retriever or retrieve)(expanded_query(question), ticker=ticker.upper(), k=max(20,k),
                                  date_range=("1900-01-01", cutoff.isoformat()))
    rows = rank_evidence(rows, question)
    citations, passages = [], []
    for row in rows:
        meta = row.get("metadata", {})
        if row.get("ticker") != ticker.upper() or not row.get("filing_date"):
            continue
        if date.fromisoformat(row["filing_date"]) > cutoff:
            continue
        if not meta.get("source_url") or not meta.get("accession_number"):
            continue
        if len(citations) >= k: break
        number = len(citations) + 1
        excerpt = row["text"]
        citations.append({"id": number, "source_url": meta["source_url"],
            "accession_number": meta["accession_number"], "cik": meta.get("cik"),
            "filing_date": row["filing_date"], "section": row.get("section"),
            "chunk_id": meta.get("chunk_id"), "excerpt": excerpt})
        passages.append(f"[{number}] {excerpt}")
    return {"answer": "Relevant filing evidence:\n\n" + "\n\n".join(passages) if passages else
            "No cited filing evidence was found for this company by the requested date.",
            "citations": citations, "status": "evidence_found" if passages else "unavailable",
            "answer_mode": "extractive", "as_of": cutoff.isoformat(),
            "limitation": "Passages support research; this mode does not synthesize conclusions or verify numerical reasoning."}
