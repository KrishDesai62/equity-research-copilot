"""Export source-only review tasks for the newly reserved cohort, never scores it."""
import json
from pathlib import Path
import pandas as pd
from src.data_layer.research_dataset import write_json,write_jsonl
from src.validation.cohort_v3_data import digest
from src.validation.outcome_expansion_data import ROOT as DATA,PLAN
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/fresh_reservation_reviews_20261019')


def run():
    acquisition=json.loads((DATA/'fresh_reservation_report.json').read_text());path=DATA/'fresh_reservation.csv'
    if acquisition['errors'] or digest(path)!=acquisition['dataset_sha256']:raise ValueError('Reservation evidence incomplete or changed')
    frame=pd.read_csv(path,dtype={'cik':str});tasks=[];source_hashes={}
    for row in frame.loc[frame[TARGET].notna()].to_dict('records'):
        marker=DATA/'fresh_reservation_checkpoints'/(row['cik']+'.json');record=json.loads(marker.read_text())
        for source in record['sources']:
            if digest(source['path'])!=source['sha256']:raise ValueError('Reserved source changed')
            source_hashes[source['path']]=source['sha256']
        statements=[]
        for review in record.get('reviews',[]):
            for assertion in review.get('assertions',[]):
                if assertion['issuer_explicitly_included'] and row['prediction_date']<assertion['petition_date']<=row['horizon_end'] and assertion['filing_date']<=row['label_available_date']:
                    statements.append({'source':review['source'],'assertion':assertion})
        label=int(row[TARGET])
        if label==1 and not statements:raise ValueError('Reserved positive lacks primary issuer evidence')
        if label==0 and not record['reporting_followup_complete']:raise ValueError('Reserved negative lacks full follow-up')
        tasks.append({'unit_id':row['unit_id'],'cik':row['cik'],'company_name':row['company_name'],'prediction_date':row['prediction_date'],'horizon_end':row['horizon_end'],
          'evidence_cutoff':row['label_available_date'],'automated_proposed_label':label,'target':TARGET,
          'scope':'SEC-disclosed issuer Chapter 7/11 petition; not exhaustive court coverage',
          'source_checkpoint':{'path':str(marker),'sha256':digest(marker)},'positive_statement_evidence':statements,
          'reporting_followup_complete':record['reporting_followup_complete'],'unresolved_accessions':record['unresolved_accessions'],
          'review_status':'pending_independent_review','reviewers':[],'approved_label':None,'model_predictions_included':False})
    ROOT.mkdir(parents=True,exist_ok=True)
    inputs={str(p):digest(p) for p in [path,DATA/'fresh_reservation_report.json',PLAN,Path(__file__)]};saved=ROOT/'manifest.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs or digest(ROOT/'pending_reviews.jsonl')!=report['packet_sha256']:raise ValueError('Frozen reservation review packet changed')
        return report
    write_jsonl(ROOT/'pending_reviews.jsonl',tasks)
    unknown=frame.loc[frame[TARGET].isna(),['cik','company_name','unit_id','label_reason']].to_dict('records');write_json(ROOT/'unresolved.json',unknown)
    report={'inputs':inputs,'packet_sha256':digest(ROOT/'pending_reviews.jsonl'),'review_units':len(tasks),'positive_units':sum(t['automated_proposed_label'] for t in tasks),
       'negative_units':sum(t['automated_proposed_label']==0 for t in tasks),'unknown_units_not_admitted':len(unknown),'source_hashes':source_hashes,'independent_reviews_completed':0,
       'minimum_positive_companies':20,'positive_support_gate_passed':sum(t['automated_proposed_label'] for t in tasks)>=20,
       'model_predictions_included':False,'production_approval':False}
    write_json(saved,report)
    if tasks:
        task=tasks[0]
        write_json(ROOT/'response_template.json',{'unit_id':task['unit_id'],'target':TARGET,'packet_sha256':report['packet_sha256'],
          'checkpoint_sha256':task['source_checkpoint']['sha256'],'reviewer_id':'','reviewed_at':'','approved_label':None,
          'independent_review_attestation':False,'checked_primary_sources':False,'full_horizon_disclosure_coverage_checked':False,'notes':''})
    return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
