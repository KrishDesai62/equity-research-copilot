"""Source-linked new positive training examples; no invented reviewer approval."""
import json
from pathlib import Path
import pandas as pd
from src.data_layer.research_dataset import write_json,write_jsonl
from src.validation.cohort_v3_data import digest
from src.validation.outcome_expansion_data import ROOT as DATA,BASE
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/expanded_training_reviews_20261019')


def run():
    path=DATA/'training.csv';report=json.loads((DATA/'training_report.json').read_text())
    if report['errors'] or digest(path)!=report['dataset_sha256']:raise ValueError('Expanded training evidence changed')
    frame=pd.read_csv(path,dtype={'cik':str});old=pd.read_csv(BASE,dtype={'cik':str})
    new=set(frame.loc[frame[TARGET].eq(1),'cik'])-set(old.loc[old[TARGET].eq(1),'cik'])
    tasks=[];proofs={}
    for row in frame.loc[frame.cik.isin(new)&frame[TARGET].eq(1)].to_dict('records'):
        marker=DATA/'training_checkpoints'/(row['cik']+'.json');record=json.loads(marker.read_text());assertions=[]
        for source in record['sources']:
            if digest(source['path'])!=source['sha256']:raise ValueError('New positive source changed')
            proofs[source['path']]=source['sha256']
        for review in record.get('event_reviews',[]):
            for assertion in review.get('assertions',[]):
                if assertion['issuer_explicitly_included'] and row['prediction_date']<assertion['petition_date']<=row['horizon_end'] and assertion['filing_date']<=row['label_available_date']:
                    matches=[s for s in record['sources'] if s.get('url')==assertion['source_url'] and s['sha256']==assertion['source_sha256']]
                    if len(matches)!=1:raise ValueError('Ambiguous primary source for new positive')
                    assertions.append({'source':matches[0],'assertion':assertion})
        if not assertions:raise ValueError('New positive lacks a primary issuer assertion')
        tasks.append({'unit_id':row['unit_id'],'cik':row['cik'],'prediction_date':row['prediction_date'],'horizon_end':row['horizon_end'],'evidence_cutoff':row['label_available_date'],
           'target':TARGET,'automated_proposed_label':1,'positive_statement_evidence':assertions,'source_checkpoint':{'path':str(marker),'sha256':digest(marker)},
           'review_status':'pending_independent_review','reviewers':[],'approved_label':None,'model_predictions_included':False})
    ROOT.mkdir(parents=True,exist_ok=True);inputs={str(p):digest(p) for p in [path,BASE,DATA/'training_report.json',Path(__file__)]}
    saved=ROOT/'manifest.json'
    if saved.exists():
        result=json.loads(saved.read_text())
        if result['inputs']!=inputs or digest(ROOT/'pending_reviews.jsonl')!=result['packet_sha256']:raise ValueError('New positive packet changed')
        return result
    write_jsonl(ROOT/'pending_reviews.jsonl',tasks)
    result={'inputs':inputs,'new_positive_companies':len(new),'review_units':len(tasks),'source_hashes':proofs,'packet_sha256':digest(ROOT/'pending_reviews.jsonl'),
       'independent_reviews_completed':0,'production_approval':False,'model_predictions_included':False};write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
