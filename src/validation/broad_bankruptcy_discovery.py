"""Find primary-source candidates beyond Item 1.03; never admits labels."""
import json,re
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/broad_bankruptcy_discovery_20261019')
RAW=Path('data/raw/broad_bankruptcy_discovery_20261019')
QUERIES={'training_candidates':('2022-06-01','2023-05-19'),'later_candidates':('2025-04-02','2026-10-18')}


def page(label,start,end,offset):
    params={'q':'"chapter 11" OR "chapter 7"','forms':'8-K','startdt':start,'enddt':end,'dateRange':'custom','from':offset,'size':100}
    proof=coordinated_fetch('https://efts.sec.gov/LATEST/search-index?'+urlencode(params),RAW/label/(str(offset)+'.json'))
    payload=json.loads(Path(proof['path']).read_text())
    if payload.get('timed_out') or payload.get('_shards',{}).get('failed'):raise ValueError('Incomplete discovery')
    total=payload['hits']['total']
    if total['relation']!='eq' or total['value']>10000:raise ValueError('Search truncated; cannot admit discovery')
    query=payload['query'];filters=query['query']['bool']['filter']
    if query['from']!=offset or query['size']!=100:raise ValueError('Ignored discovery pagination')
    if {'range':{'file_date':{'gte':start,'lte':end}}} not in filters or {'terms':{'root_forms':['8-K']}} not in filters:raise ValueError('Ignored discovery filters')
    encoded=json.dumps(query).lower()
    if 'chapter 11' not in encoded or 'chapter 7' not in encoded:raise ValueError('Ignored discovery keywords')
    return payload,int(total['value']),proof


def run():
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    inputs={str(p):digest(p) for p in [Path(__file__),Path('config/outcome_expansion_20261019/acquisition.json')]}
    if saved.exists():
        record=json.loads(saved.read_text())
        if record['inputs']!=inputs or digest(ROOT/'candidates.csv')!=record['candidates_sha256']:raise ValueError('Frozen broad discovery changed')
        for s in record['sources']:
            if digest(s['path'])!=s['sha256']:raise ValueError('Broad search bytes changed')
        return record
    plan=json.loads(Path('config/outcome_expansion_20261019/acquisition.json').read_text())
    reserved=set(plan['prior_selected_ids'])|{c['cik'] for c in plan['training']+plan['fresh_reservation']}
    sources=[];rows=[];counts={}
    for label,(start,end) in QUERIES.items():
        first,total,proof=page(label,start,end,0);sources.append(proof);hits=list(first['hits']['hits'])
        for offset in range(100,total,100):
            payload,n,proof=page(label,start,end,offset)
            if n!=total:raise ValueError('Discovery changed during pagination')
            hits.extend(payload['hits']['hits']);sources.append(proof)
        if len(hits)!=total or len({h['_id'] for h in hits})!=total:raise ValueError('Missing/duplicate discovery hits')
        counts[label]=total
        for hit in hits:
            source=hit['_source'];accession=source['adsh']
            if source['form'] not in {'8-K','8-K/A'} or not start<=source['file_date']<=end or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',accession):raise ValueError('Invalid discovery hit')
            for raw in source['ciks']:
                if not re.fullmatch(r'\d{1,10}',raw):raise ValueError('Invalid discovery CIK')
                cik=raw.zfill(10)
                rows.append({'query':label,'cik':cik,'accession':accession,'filing_date':source['file_date'],'display_names':'; '.join(source['display_names']),
                  'indexed_item_103':'1.03' in source.get('items',[]),'previously_selected_or_reserved':cik in reserved,
                  'search_hit_id':hit['_id'],'label_admitted':False})
    frame=pd.DataFrame(rows).drop_duplicates(['query','cik','accession']);frame.to_csv(ROOT/'candidates.csv',index=False)
    new=frame.loc[~frame.previously_selected_or_reserved&~frame.indexed_item_103]
    result={'inputs':inputs,'sources':sources,'queries':QUERIES,'search_hits':counts,'candidate_rows':len(frame),
       'not_previously_selected_non_item103_companies':int(new.cik.nunique()),'candidates_sha256':digest(ROOT/'candidates.csv'),
       'labels_admitted':0,'production_approval':False,'model_predictions_generated':False,
       'limits':'Search candidates only: phrases may concern subsidiaries, counterparties or hypothetical events. Identity, petition date, prior features and complete horizon must be verified before label admission. Later candidates are not moved into training or used to tune a model.'}
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
