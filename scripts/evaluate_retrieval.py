"""Reproduce a source-checked development retrieval comparison, without model scoring."""
import argparse
from html import escape
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
from src.agents.qa_agent import answer_question
from src.rag.local_index import LocalIndex,build_index
from src.rag.hybrid import wrap


def evaluate(fixture='evaluation/retrieval_development.json',frontend='frontend',output=None):
    spec=json.loads(Path(fixture).read_text());companies={};rows=[];hashes={}
    for ticker in sorted({c['ticker'] for c in spec['cases']}):
        path=Path(frontend)/'data/companies'/f'{ticker}.json'
        if not path.exists():continue
        hashes[ticker]=sha256(path.read_bytes()).hexdigest()
        company=json.loads(path.read_text());companies[ticker]=company
        rows.extend(dict(p,ticker=ticker,cik=company['company']['cik']) for p in company['passages'])
    known={r['chunk_id']:r for r in rows}
    for case in spec['cases']:
        for cid in case.get('relevant_ids',[]):
            if cid not in known or known[cid]['ticker']!=case['ticker'] or known[cid]['filing_date']>case['as_of']:raise ValueError('Invalid expected source annotation')
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'chunks.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        path=Path(directory)/'index.sqlite';build_index(source,path);index=LocalIndex(path);hybrid=wrap(index.retrieve)
        results=[]
        for case in spec['cases']:
            baseline=index.retrieve(case['question'],ticker=case['ticker'],date_range=('1900-01-01',case['as_of']),k=5)
            start=time.perf_counter();response=answer_question(case['question'],case['ticker'],as_of=case['as_of'],k=5,retriever=hybrid);elapsed=time.perf_counter()-start
            actual=[c['chunk_id'] for c in response['citations']];expected=set(case.get('relevant_ids',[]))
            old=[r['metadata']['chunk_id'] for r in baseline]
            reciprocal=lambda ids:next((1/(i+1) for i,cid in enumerate(ids) if cid in expected),0.)
            valid=all(c['chunk_id'] in known and c['filing_date']<=case['as_of'] and known[c['chunk_id']]['ticker']==case['ticker'] and c['excerpt']==known[c['chunk_id']]['text'] and c['source_url']==known[c['chunk_id']]['source_url'] for c in response['citations'])
            results.append(dict(id=case['id'],positive=bool(expected),baseline_hit_at_5=bool(expected.intersection(old)),hit_at_5=bool(expected.intersection(actual)),reciprocal_rank_at_5=reciprocal(actual),baseline_reciprocal_rank_at_5=reciprocal(old),citation_fidelity=valid,abstained=not actual,expected_abstention=case.get('expect_abstain',False),returned_ids=actual,elapsed_seconds=round(elapsed,4)))
    positives=[r for r in results if r['positive']];negatives=[r for r in results if r['expected_abstention']]
    report=dict(label_status=spec['label_status'],limitation=spec['limitation'],corpus_sha256=hashes,cases=len(results),positive_cases=len(positives),negative_cases=len(negatives),
        baseline_hit_at_5=sum(r['baseline_hit_at_5'] for r in positives)/len(positives),hit_at_5=sum(r['hit_at_5'] for r in positives)/len(positives),mrr_at_5=sum(r['reciprocal_rank_at_5'] for r in positives)/len(positives),baseline_mrr_at_5=sum(r['baseline_reciprocal_rank_at_5'] for r in positives)/len(positives),
        abstention_correct=sum(r['abstained'] for r in negatives),citation_fidelity_cases=sum(r['citation_fidelity'] for r in results),results=results)
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2))
        sections=[]
        for case,result in zip(spec['cases'],results):
            ids=list(dict.fromkeys(case.get('relevant_ids',[])+result['returned_ids']))
            entries=[]
            for cid in ids:
                row=known[cid]
                entries.append('<article><h3>'+escape(cid)+'</h3><p>'+escape(row['ticker']+' / '+row['filing_date']+' / '+row['section'])+'</p><p>'+escape(row['text'])+'</p></article>')
            sections.append('<section><h2>'+escape(case['id'])+'</h2><p>'+escape(case['question'])+'</p><p>Expected example IDs: '+escape(', '.join(case.get('relevant_ids',[])))+'</p>'+''.join(entries)+'</section>')
        Path(output).with_suffix('.review.html').write_text('<!doctype html><meta charset="utf-8"><title>Retrieval review</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;line-height:1.7}article{border:1px solid #ddd;padding:16px;margin:12px 0}h3{font-size:12px;overflow-wrap:anywhere}section{margin:50px 0}</style><h1>Retrieval development review</h1><p>Assistant-authored labels; independent human review pending. Existing rankings are visible, so this packet is for adjudication, not blind evaluation.</p>'+''.join(sections))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/processed/evaluation/retrieval_development_20261024.json');args=p.parse_args();r=evaluate(output=args.output);print(json.dumps({k:v for k,v in r.items() if k not in {'results','corpus_sha256'}},indent=2))
