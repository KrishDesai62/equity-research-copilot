"""Validate actual independent review responses; never manufactures sign-offs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET

PACKET=Path('data/processed/validation/final_outcome_audit_v2_20261019')
ROOT=Path('data/processed/validation/current_review_status_20261019')


def validate_response(record, task, packet_sha, *, now):
    reviewer=record.get('reviewer_id','')
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('A real reviewer identifier is required')
    if record.get('target')!=TARGET or record.get('unit_id')!=task['unit_id']:raise ValueError('Review target/unit mismatch')
    if record.get('packet_sha256')!=packet_sha or record.get('checkpoint_sha256')!=task['source_checkpoint']['sha256']:raise ValueError('Review evidence version mismatch')
    if record.get('independent_review_attestation') is not True or record.get('checked_primary_sources') is not True:raise ValueError('Real independent source-review attestations are required')
    day=datetime.fromisoformat(record['reviewed_at'].replace('Z','+00:00'))
    if day.tzinfo is None or day>now or day.date().isoformat()<task['evidence_cutoff']:raise ValueError('Review timestamp is naive, premature or in the future')
    label=record.get('approved_label')
    if label is not None and (type(label)is not int or label not in {0,1}):raise ValueError('Review label must be binary or explicitly uncertain')
    if label==0 and record.get('full_horizon_disclosure_coverage_checked') is not True:raise ValueError('Negative review needs complete disclosure follow-up')
    if not str(record.get('notes','')).strip():raise ValueError('Review needs evidence notes')
    return reviewer.strip().casefold(),label


def assess(tasks,responses,packet_sha,*,now):
    groups={key:{} for key in tasks}
    for record in responses:
        unit=record.get('unit_id')
        if unit not in tasks:raise ValueError('Unknown review unit')
        reviewer,label=validate_response(record,tasks[unit],packet_sha,now=now)
        if reviewer in groups[unit]:raise ValueError('Duplicate reviewer response; use an explicit revised review file')
        groups[unit][reviewer]=label
    pending,uncertain,corrections,approved=[],[],[],[]
    for unit,answers in groups.items():
        if len(answers)<2:pending.append(unit);continue
        labels=set(answers.values())
        if None in labels or len(labels)!=1:uncertain.append(unit)
        elif next(iter(labels))!=tasks[unit]['automated_proposed_label']:corrections.append(unit)
        else:approved.append(unit)
    return {'review_units':len(tasks),'approved_current_units':len(approved),'pending_units':pending,
        'disagreement_or_uncertain_units':uncertain,'label_revision_required_units':corrections,
        'independent_reviews_complete':bool(tasks) and len(approved)==len(tasks),
        'identity_independence_verification':'Operator-supplied human attestations; this validator does not authenticate people or prove legal conclusions.',
        'production_approval':False}


def run(reviews=None):
    manifest=json.loads((PACKET/'manifest.json').read_text());packet=PACKET/'pending_final_reviews.jsonl'
    if digest(packet)!=manifest['packet_sha256']:raise ValueError('Review packet changed')
    tasks={r['unit_id']:r for r in map(json.loads,packet.read_text().splitlines())}
    responses=[] if reviews is None else [json.loads(line) for line in Path(reviews).read_text().splitlines() if line.strip()]
    result=assess(tasks,responses,manifest['packet_sha256'],now=datetime.now(timezone.utc))
    result.update(packet_sha256=manifest['packet_sha256'],review_file_sha256=digest(reviews) if reviews else None)
    ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'status.json',result)
    task=next(iter(tasks.values()))
    template={'unit_id':task['unit_id'],'target':TARGET,'packet_sha256':manifest['packet_sha256'],
        'checkpoint_sha256':task['source_checkpoint']['sha256'],'reviewer_id':'','reviewed_at':'',
        'approved_label':None,'checked_primary_sources':False,'independent_review_attestation':False,
        'full_horizon_disclosure_coverage_checked':False,'notes':''}
    write_json(ROOT/'response_template.json',template)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reviews',type=Path);args=parser.parse_args()
    result=run(args.reviews)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_units')},indent=2))
