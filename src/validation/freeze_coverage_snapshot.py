"""Freeze the retained research study together with reconciled coverage evidence."""
import json
from pathlib import Path
from src.validation.freeze_local_research import run as original,verify_files
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/frozen_coverage_snapshot_20261019')
EXTRA=[Path('data/processed/validation/reconciled_coverage_20261019/report.json'),
 Path('data/processed/validation/broad_bankruptcy_discovery_20261019/report.json'),
 Path('data/processed/validation/broad_candidate_review_20261019/report.json')]


def run():
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'manifest.json'
    if saved.exists():
        record=json.loads(saved.read_text());verify_files(record['files']);return record
    previous=original();files=dict(previous['files']);files[str(Path(__file__))]=digest(__file__)
    for path in EXTRA:
        record=json.loads(path.read_text());files[str(path)]=digest(path)
        for key in ['inputs','source_hashes']:
            files.update(record.get(key,{}))
        for proof in record.get('sources',[]):files[proof['path']]=proof['sha256']
    extra_outputs=[Path('data/processed/validation/reconciled_coverage_20261019/remaining_current_reports.json'),
       Path('data/processed/validation/broad_bankruptcy_discovery_20261019/candidates.csv'),
       Path('data/processed/validation/broad_candidate_review_20261019/proposals.json')]
    for path in extra_outputs:files[str(path)]=digest(path)
    verify_files(files)
    result={**previous,'files':files,'coverage_evidence_reconciled':True,'new_verified_labels_admitted':0,
       'blockers':previous['blockers']+['Broad discovery/primary triage did not supply additional admitted positive companies.'],
       'limits':'Research snapshot only. Missing body reviews and independent outcome review are explicit blockers. Neither original nor fresh reserved cohort was scored.'}
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
