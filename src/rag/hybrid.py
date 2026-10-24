"""Bounded per-company latent semantic analysis blended with lexical retrieval.

Fits only date-eligible passages; no downloads or external embedding API.
"""
from functools import lru_cache
import json
import sqlite3
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from src.rag.query_relevance import expanded_query

_LOCK=threading.Lock()

@lru_cache(maxsize=8)
def corpus(path,mtime,ticker,start,end,section,form):
    clauses=['ticker=?','filing_date BETWEEN ? AND ?'];args=[ticker,start,end]
    for key,value in [('section_title',section),('filing_type',form)]:
        if value:clauses.append(key+'=?');args.append(value)
    with sqlite3.connect(path) as c:
        records=[json.loads(r[0]) for r in c.execute('SELECT payload FROM metadata WHERE '+' AND '.join(clauses)+' ORDER BY chunk_id LIMIT 2000',args)]
    if len(records)<3:return None
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=12000,min_df=1,sublinear_tf=True)
    matrix=vectorizer.fit_transform([r['text'] for r in records])
    dimensions=min(64,len(records)-1,matrix.shape[1]-1)
    if dimensions<2:return None
    with threadpool_limits(limits=1):
        svd=TruncatedSVD(n_components=dimensions,random_state=0)
        embedded=normalize(svd.fit_transform(matrix))
    return records,vectorizer,svd,embedded


def wrap(retriever):
    index=getattr(retriever,'__self__',None)
    if index is None or not hasattr(index,'path'):return retriever
    def hybrid(query,**kwargs):
        k=kwargs.get('k',5);lexical=retriever(query,**kwargs)
        if not kwargs.get('ticker') or kwargs.get('has_table'):return lexical
        path=index.path.resolve();start,end=kwargs.get('date_range') or ('1900-01-01','9999-12-31')
        with _LOCK:
            fitted=corpus(str(path),path.stat().st_mtime_ns,kwargs['ticker'],start,end,kwargs.get('section_title'),kwargs.get('filing_type'))
        if fitted is None:return lexical
        records,vectorizer,svd,embedded=fitted
        with threadpool_limits(limits=1):
            vector=normalize(svd.transform(vectorizer.transform([expanded_query(query)])))
        similarities=embedded@vector[0]
        candidates={};scores={}
        for rank,row in enumerate(lexical):
            key=row['metadata']['chunk_id'];candidates[key]=row;scores[key]=1/(60+rank+1)
        for rank,i in enumerate(np.argsort(-similarities)[:max(20,k)]):
            similarity=float(similarities[i])
            if similarity<.2:continue
            r=records[i];key=r['chunk_id']
            row=dict(text=r['text'],ticker=r['ticker'],section=r['section'],filing_type=r['filing_type'],filing_date=r['filing_date'],metadata=r,semantic_score=similarity,retrieval_method='hybrid_fts5_lsa')
            candidates.setdefault(key,row);scores[key]=scores.get(key,0)+1/(60+rank+1)
        return [dict(candidates[key],retrieval_method='hybrid_fts5_lsa') for key in sorted(scores,key=lambda key:-scores[key])[:k]]
    return hybrid
