"""
THE shared retrieval backbone. Every downstream consumer (Q&A agent in
MVP; feature extraction, bull/bear/risk/synthesis agents, portfolio
agent in later phases) calls retrieve() and nothing else — they should
never touch vector_store.py or embeddings.py directly. Keeping this as
the single choke point is what makes "one retrieval backbone, three
consumers" (from the spec) actually true instead of aspirational.
"""
from typing import Optional
import embeddings
import vector_store


def retrieve(
    query: str,
    ticker: Optional[str] = None,
    filing_type: Optional[str] = None,
    date_range: Optional[tuple[str, str]] = None,
    section_title: Optional[str] = None,  
    has_table: Optional[bool] = None,
    k: int = 5,
) -> list[dict]:
    query_vector = embeddings.embed_texts(None, [query])[0]
    where = {}
    if ticker: where['ticker'] = ticker
    if filing_type: where['filing_type'] = filing_type
    if date_range: where['filing_date'] = {'$gte': date_range[0], '$lte': date_range[1]}
    if section_title: where['section_title'] = section_title
    if has_table is not None: where['has_table'] = has_table
    
    results = vector_store.query(
        query_embedding=query_vector,
        where=where,
        n_results=k
    )
    
    
    formatted_results = []
    for i in range(len(results['documents'])):
        meta = results['metadatas'][i]
        formatted_results.append({
            'text': results['documents'][i],
            "ticker": meta.get("ticker"),
            "filing_type": meta.get("filing_type"),
            "section": meta.get("section_title"), 
            "filing_date": meta.get("filing_date"),
            "similarity_score": results['distances'][i]
        })
        
    return formatted_results