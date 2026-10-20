"""Shared filing retrieval, with point-in-time metadata filters."""
from datetime import date
from functools import lru_cache
from src.rag import embeddings


@lru_cache(maxsize=1)
def default_store():
    from src.rag.vector_store import VectorStore
    return VectorStore("data/processed/chroma")


def retrieve(query, ticker=None, filing_type=None, date_range=None,
             section_title=None, has_table=None, k=5, *, store=None, model=None):
    if k < 1:
        raise ValueError("k must be positive")
    if store is None and model is None:
        from pathlib import Path
        from src.rag.local_index import DEFAULT_PATH, LocalIndex
        if Path(DEFAULT_PATH).exists():
            return LocalIndex().retrieve(query, ticker=ticker, filing_type=filing_type,
                date_range=date_range, section_title=section_title, has_table=has_table, k=k)
    clauses = []
    for field, value in (("ticker", ticker), ("filing_type", filing_type),
                         ("section_title", section_title), ("has_table", has_table)):
        if value is not None:
            clauses.append({field: value})
    if date_range:
        start, end = (date.fromisoformat(str(value)[:10]).toordinal() for value in date_range)
        if start > end:
            raise ValueError("Date range starts after it ends")
        clauses.extend([{"filing_date_ordinal": {"$gte": start}},
                        {"filing_date_ordinal": {"$lte": end}}])
    where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
    vector = embeddings.embed_texts(model, [query])[0]
    results = (store if store is not None else default_store()).query(vector, k=k, where=where)
    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]
    return [{"text": text, "ticker": meta.get("ticker"),
             "filing_type": meta.get("filing_type"), "section": meta.get("section_title"),
             "filing_date": meta.get("filing_date"), "metadata": meta,
             "similarity_score": 1.0 - distance}
            for text, meta, distance in zip(documents, metadata, distances)]
