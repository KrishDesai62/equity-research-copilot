"""Broader primary-document audit of the five existing false-alert negatives."""
import json,re
from pathlib import Path
import pandas as pd
from src.validation.cohort_v3_data import digest
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.issuer_identity_admission_v3 import corroborate_identity
from src.validation.filing_text_signals import visible_text
from src.validation.sec_disclosure_repair import outcome_submission_rows
from src.validation.sec_disclosure_data import TARGET
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/negative_coverage_audit_20261019')


def documents_for(record,cik,start,end):
    rows={};names=set()
    for source in record['sources']:
        if digest(source['path'])!=source['sha256']:raise ValueError('Coverage evidence changed')
        if '/submissions/' not in source['path']:continue
        payload=json.loads(Path(source['path']).read_text())
        if 'cik' in payload and str(payload['cik']).zfill(10)!=cik:raise ValueError('Coverage issuer mismatch')
        if payload.get('name'):names.add(payload['name'])
        block=payload.get('filings',{}).get('recent',payload)
        for row in outcome_submission_rows(block,cik,'CIK'+cik,start,end):
            if row['filing_type'] in {'8-K','8-K/A','10-K','10-Q'}:rows[row['accession_number']]=row
    return rows,names


def run():
    table=Path('data/processed/validation/recent_study_features_20261019/calibration.csv')
    alerts=Path('data/processed/validation/recent_operating_policy_20261019/alerts.csv')
    frame=pd.read_csv(table,dtype={'cik':str}).merge(pd.read_csv(alerts)[['unit_id','alerts_3']],on='unit_id',validate='one_to_one')
    frame=frame.loc[frame[TARGET].eq(0)&frame.alerts_3.eq(1)]
    ROOT.mkdir(parents=True,exist_ok=True);results=[];sources={};errors=[]
    for row in frame.to_dict('records'):
        marker=Path('data/processed/validation/recent_study_data_20261019/calibration_checkpoints')/(row['cik']+'.json')
        record=json.loads(marker.read_text());metadata,names=documents_for(record,row['cik'],row['prediction_date'],row['label_available_date'])
        names.add(row['company_name']);reviewed={r.get('accession',r.get('accession_number')) for r in record.get('reviews',[])}
        # Inspect every indexed current report skipped by the old reviewer,
        # not just bankruptcy items. Annual/quarterly bodies remain a gap.
        candidates=[d for d in metadata.values() if d['filing_type'] in {'8-K','8-K/A'} and d['accession_number'] not in reviewed and d['filing_date']>row['prediction_date']]
        documents=[]
        for doc in sorted(candidates,key=lambda x:x['accession_number']):
            key=row['cik']+'_'+doc['accession_number'];path=ROOT/'raw'/(key+'.html')
            print('Coverage document',key,flush=True)
            try:
                proof=coordinated_fetch(doc['source_url'],path);sources[proof['path']]=proof['sha256']
                payload=path.read_text(errors='replace');text=visible_text(path.read_bytes())
                terms=bool(re.search(r'\bbankrupt\w*\b|\bchapter\s*(?:7|11)\b|\breceivership\b',text,re.I))
                assertions=corroborate_identity(payload,filed=doc['filing_date'],names=sorted(names)) if terms else []
                qualifying=[a for a in assertions if a['issuer_explicitly_included'] and row['prediction_date']<a['petition_date']<=row['horizon_end']]
                documents.append({'accession':doc['accession_number'],'filing_date':doc['filing_date'],'source':proof,'bankruptcy_terms_present':terms,
                                  'assertions':assertions,'qualifying_issuer_assertions':qualifying,'requires_independent_review':terms})
            except Exception as exc:errors.append({'unit_id':row['unit_id'],'accession':doc['accession_number'],'error':str(exc)})
        results.append({'unit_id':row['unit_id'],'cik':row['cik'],'company_name':row['company_name'],'prediction_date':row['prediction_date'],'horizon_end':row['horizon_end'],
          'checkpoint':str(marker),'checkpoint_sha256':digest(marker),'extra_current_reports':len(candidates),'completed_extra_reports':len(documents),
          'uninspected_financial_report_bodies':sum(d['filing_type'] in {'10-K','10-Q'} for d in metadata.values()),'documents':documents})
    inputs={str(p):digest(p) for p in [table,alerts,Path(__file__),Path('src/validation/issuer_identity_admission_v3.py')]}
    result={'inputs':inputs,'companies':results,'source_hashes':sources,'errors':errors,
      'qualifying_assertions_found':sum(len(d['qualifying_issuer_assertions']) for r in results for d in r['documents']),
      'documents_needing_review':sum(d['requires_independent_review'] for r in results for d in r['documents']),
      'labels_changed':False,'independent_reviews_completed':0,'production_approval':False,'final_rows_used':False,
      'limits':'Expanded automated document coverage for five reused-development negatives only. Financial-report bodies, court records and all other companies remain outside this pass. A keyword or subsidiary bankruptcy is not an issuer label. No absent keyword establishes a verified negative.'}
    saved=ROOT/'report.json'
    if saved.exists() and json.loads(saved.read_text())!=result:raise ValueError('Frozen coverage audit changed')
    if not errors:write_json(saved,result)
    else:write_json(ROOT/'incomplete_report.json',result)
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
