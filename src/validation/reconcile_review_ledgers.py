"""Deduplicate the checkpoint-level coverage queue against earlier review ledgers."""
import json
from pathlib import Path
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/reconciled_coverage_20261019')
DIRECTORIES=['cohort_v3/company_checkpoints','sec_disclosure_v1/checkpoints','sec_disclosure_v2/checkpoints',
 'sec_disclosure_training_v2/checkpoints','historical_training_20261019/checkpoints','recent_development_20261019/checkpoints',
 'recent_study_data_20261019/training_checkpoints','recent_study_data_20261019/calibration_checkpoints',
 'outcome_expansion_20261019/training_checkpoints','outcome_expansion_20261019/fresh_reservation_checkpoints']


def run():
    inventory=Path('data/processed/validation/negative_coverage_inventory_20261019/unreviewed_current_reports.json')
    queue=json.loads(inventory.read_text());keys={d['cik']+'_'+d['accession'] for d in queue};found={};inputs={str(inventory):digest(inventory),str(Path(__file__)):digest(__file__)}
    for directory in DIRECTORIES:
        for path in sorted((Path('data/processed/validation')/directory).glob('*.json')):
            r=json.loads(path.read_text());reviews=r.get('event_reviews',r.get('reviews',[]))
            if not reviews:continue
            sources={s['sha256']:s for s in r.get('sources',[]) if 'sha256' in s}
            for review in reviews:
                accession=review.get('accession_number',review.get('accession'));key=path.stem+'_'+str(accession)
                if key not in keys:continue
                proof=review.get('source') or sources.get(review.get('source_sha256'))
                if not proof:continue
                if digest(proof['path'])!=proof['sha256']:raise ValueError('Prior ledger primary bytes changed')
                inputs[str(path)]=digest(path);inputs[proof['path']]=proof['sha256']
                found.setdefault(key,[]).append({'checkpoint':str(path),'source':proof})
    unresolved=[d for d in queue if d['cik']+'_'+d['accession'] not in found]
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    result={'inputs':inputs,'checkpoint_inventory_reports':len(queue),'reports_with_other_recorded_body_review':len(found),
      'reports_without_a_body_review_in_scanned_ledgers':len(unresolved),'labels_changed':False,'production_approval':False,
      'limits':'Reconciles recorded body-review coverage only, not legal correctness, exhaustive court coverage, independent human review or completeness of every historical ledger. No predictions.'}
    if saved.exists():
        previous=json.loads(saved.read_text())
        if any(previous[k]!=v for k,v in result.items()):raise ValueError('Frozen ledger reconciliation changed')
        if digest(ROOT/'remaining_current_reports.json')!=previous['remaining_queue_sha256']:raise ValueError('Reconciled queue changed')
        return previous
    write_json(ROOT/'remaining_current_reports.json',unresolved);result['remaining_queue_sha256']=digest(ROOT/'remaining_current_reports.json');write_json(saved,result)
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
