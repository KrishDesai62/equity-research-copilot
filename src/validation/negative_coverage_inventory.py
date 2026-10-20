"""Inventory label-evidence gaps across negative rows; no automatic relabeling."""
import json,re
from pathlib import Path
import pandas as pd
from src.validation.sec_filing_signals import read_metadata
from src.validation.matched_development_text import checkpoint_map
from src.validation.sec_disclosure_data import TARGET
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/negative_coverage_inventory_20261019')


def run():
    data=Path('data/processed/validation')
    paths={'training':data/'outcome_expansion_20261019/training.csv','development':data/'recent_study_data_20261019/calibration.csv','reserved_fresh':data/'outcome_expansion_20261019/fresh_reservation.csv'}
    mapping=checkpoint_map()
    for parent,role in [('recent_study_data_20261019','training'),('outcome_expansion_20261019','training')]:
        for p in (data/parent/(role+'_checkpoints')).glob('*.json'):
            r=json.loads(p.read_text())
            if any('/submissions/' in s['path'] for s in r.get('sources',[])):mapping[p.stem]=str(p)
    frames={r:pd.read_csv(p,dtype={'cik':str}) for r,p in paths.items()};rows=[];queue={};proofs={};errors=[]
    for role,frame in frames.items():
        negatives=frame.loc[frame[TARGET].eq(0)]
        for cik,group in negatives.groupby('cik'):
            print('Coverage inventory',role,cik,flush=True)
            try:
                if role=='development':marker=data/'recent_study_data_20261019/calibration_checkpoints'/(cik+'.json')
                elif role=='reserved_fresh':marker=data/'outcome_expansion_20261019/fresh_reservation_checkpoints'/(cik+'.json')
                else:marker=Path(mapping[cik])
                record=json.loads(marker.read_text());proofs[str(marker)]=digest(marker)
                metadata,sources=read_metadata(cik,marker)
                for source in sources:proofs[source['path']]=source['sha256']
                reviews=record.get('event_reviews',record.get('reviews',[]));reviewed={r.get('accession_number',r.get('accession')) for r in reviews}
                for row in group.to_dict('records'):
                    relevant=[d for d in metadata if row['prediction_date']<d['date']<=row['label_available_date']]
                    current=[d for d in relevant if d['form'] in {'8-K','8-K/A'}]
                    skipped=[d for d in current if d['accession'] not in reviewed]
                    financial=[d for d in relevant if d['form'] in {'10-K','10-Q'}]
                    post_horizon=any(d['date']>=row['horizon_end'] for d in financial)
                    rows.append({'role':role,'cik':cik,'unit_id':row['unit_id'],'prediction_date':row['prediction_date'],'horizon_end':row['horizon_end'],'evidence_cutoff':row['label_available_date'],
                        'source_checkpoint':str(marker),'current_reports':len(current),'current_reports_not_in_original_review':len(skipped),
                        'financial_report_bodies_not_covered_by_original_event_reviewer':len(financial),'has_indexed_financial_report_covering_post_horizon_date':post_horizon,
                        'coverage_status':'requires_broader_primary_document_review','court_record_clearance':False})
                    for doc in skipped:
                        key=cik+'_'+doc['accession']
                        if key not in queue:queue[key]={'cik':cik,'accession':doc['accession'],'filing_date':doc['date'],'items':doc['items'],
                            'filing_index_url':f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{doc['accession'].replace('-','')}/{doc['accession']}-index.html",'affected_units':[]}
                        queue[key]['affected_units'].append(row['unit_id'])
            except Exception as exc:errors.append({'role':role,'cik':cik,'error':str(exc)})
    ROOT.mkdir(parents=True,exist_ok=True);inputs={str(p):digest(p) for p in [*paths.values(),Path(__file__),Path('src/validation/sec_filing_signals.py')]}
    table=pd.DataFrame(rows);table.to_csv(ROOT/'negative_rows.csv',index=False);write_json(ROOT/'unreviewed_current_reports.json',list(queue.values()))
    result={'inputs':inputs,'source_hashes':proofs,'negative_rows_audited':len(rows),'rows_with_unreviewed_current_reports':int(table.current_reports_not_in_original_review.gt(0).sum()),
       'unique_unreviewed_current_reports':len(queue),'rows_without_indexed_post_horizon_financial_report':int((~table.has_indexed_financial_report_covering_post_horizon_date).sum()),
       'counts_by_role':table.groupby('role').size().to_dict(),'errors':errors,
       'outputs':{str(p):digest(p) for p in [ROOT/'negative_rows.csv',ROOT/'unreviewed_current_reports.json']},
       'labels_changed':False,'independent_reviews_completed':0,'production_approval':False,'model_predictions_generated':False,
       'limits':'Metadata and review-inventory audit, not full primary-text review or court clearance. Post-horizon filing date alone does not establish complete financial-period coverage. Unreviewed financial report bodies remain gaps even when all current reports were examined. Reserved-company scores and financial features are not used.'}
    write_json(ROOT/('report.json' if not errors else 'incomplete_report.json'),result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
