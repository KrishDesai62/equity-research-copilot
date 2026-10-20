"""Primary-source triage of new candidates; proposals never enter training here."""
import json,re
from pathlib import Path
from urllib.parse import quote
import pandas as pd
from src.validation.cohort_v3_data import digest
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.issuer_identity_admission_v3 import corroborate_identity
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/broad_candidate_review_20261019')
RAW=Path('data/raw/broad_candidate_review_20261019')


def run():
    discovery=Path('data/processed/validation/broad_bankruptcy_discovery_20261019')
    manifest=json.loads((discovery/'report.json').read_text());candidate_path=discovery/'candidates.csv'
    if digest(candidate_path)!=manifest['candidates_sha256']:raise ValueError('Broad candidates changed')
    frame=pd.read_csv(candidate_path,dtype={'cik':str});frame=frame.loc[~frame.previously_selected_or_reserved&~frame.indexed_item_103]
    eligible={};sources={};inputs={str(p):digest(p) for p in [candidate_path,discovery/'report.json',Path(__file__),Path('src/validation/issuer_identity_admission_v3.py')]}
    for label,path,anchor in [('training_candidates','data/processed/validation/cohort_v3/annual_frame.csv','2021-10-31'),('later_candidates','data/processed/validation/outcome_expansion_sources_20261019/annual_2025.csv','2025-04-01')]:
        inputs[path]=digest(path);annual=pd.read_csv(path,dtype={'cik':str});annual=annual.loc[annual.filed.lt(anchor)].sort_values('filed').drop_duplicates('cik',keep='last')
        sic=pd.to_numeric(annual.sic,errors='coerce');annual=annual.loc[sic.notna()&~sic.between(6000,6999)]
        names=annual.set_index('cik')['name'].to_dict()
        for row in frame.loc[frame['query'].eq(label)&frame.cik.isin(names)].itertuples():eligible[(label,row.cik,row.accession)]=(names[row.cik],anchor)
    documents={}
    for proof in manifest['sources']:
        if digest(proof['path'])!=proof['sha256']:raise ValueError('Broad search bytes changed')
        inputs[proof['path']]=proof['sha256'];label=Path(proof['path']).parent.name
        for hit in json.loads(Path(proof['path']).read_text())['hits']['hits']:
            source=hit['_source'];accession=source['adsh'];leaf=hit['_id'].split(':',1)[1]
            if '/' in leaf or '\\' in leaf or leaf in {'.','..'}:raise ValueError('Unsafe SEC document leaf')
            for raw in source['ciks']:
                cik=raw.zfill(10);key=(label,cik,accession)
                if key not in eligible:continue
                documents[(label,cik,hit['_id'])]={'label':label,'cik':cik,'accession':accession,'leaf':leaf,'filing_date':source['file_date'],
                    'owner_cik':source['ciks'][0],'annual_name':eligible[key][0],'prediction_date':eligible[key][1]}
    ROOT.mkdir(parents=True,exist_ok=True);results=[];errors=[];names={}
    for i,doc in enumerate(sorted(documents.values(),key=lambda d:(d['label'],d['cik'],d['accession'],d['leaf'])),1):
        print('Broader candidate',i,len(documents),doc['cik'],flush=True)
        try:
            cik=doc['cik']
            if cik not in names:
                p=RAW/'submissions'/('CIK'+cik+'.json');proof=coordinated_fetch('https://data.sec.gov/submissions/CIK'+cik+'.json',p);sources[proof['path']]=proof['sha256'];payload=json.loads(p.read_text())
                if str(payload['cik']).zfill(10)!=cik:raise ValueError('Candidate issuer mismatch')
                names[cik]=payload['name']
            url=f"https://www.sec.gov/Archives/edgar/data/{int(doc['owner_cik'])}/{doc['accession'].replace('-','')}/{quote(doc['leaf'],safe='-_.')}"
            proof=coordinated_fetch(url,RAW/'documents'/(cik+'_'+doc['accession']+'_'+doc['leaf']));sources[proof['path']]=proof['sha256']
            text=Path(proof['path']).read_text(errors='replace')
            if re.search('request rate threshold exceeded|undeclared automated tool',text,re.I):raise ValueError('SEC access page, not filing')
            assertions=corroborate_identity(text,filed=doc['filing_date'],names=[names[cik],doc['annual_name']])
            horizon=(pd.Timestamp(doc['prediction_date'])+pd.DateOffset(years=1)).date().isoformat()
            qualifying=[a for a in assertions if a['issuer_explicitly_included'] and doc['prediction_date']<a['petition_date']<=horizon]
            results.append({**doc,'horizon_end':horizon,'sec_name':names[cik],'source':proof,'assertions':assertions,'qualifying_assertions':qualifying,
                'review_status':'pending_independent_review','approved_label':None,'label_admitted':False,'model_predictions_included':False})
        except Exception as exc:errors.append({'cik':doc['cik'],'accession':doc['accession'],'error':str(exc)})
    write_json(ROOT/'proposals.json',results)
    result={'inputs':inputs,'source_hashes':sources,'eligible_company_windows':len({(k[0],k[1]) for k in eligible}),'requested_documents':len(documents),'completed_documents':len(results),'errors':errors,
       'proposed_company_windows_with_in_horizon_issuer_assertion':len({(r['label'],r['cik']) for r in results if r['qualifying_assertions']}),
       'proposal_sha256':digest(ROOT/'proposals.json'),'labels_admitted':0,'production_approval':False,'independent_reviews_completed':0,
       'limits':'Primary-source automated triage only. Complete feature availability, issuer versus subsidiary identity, date, prior petition history and independent review are required before any admission. No new model fitting or reserved cohort predictions.'}
    saved=ROOT/('report.json' if not errors else 'incomplete_report.json')
    if saved.exists() and json.loads(saved.read_text())!=result:raise ValueError('Frozen broader candidate review changed')
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
