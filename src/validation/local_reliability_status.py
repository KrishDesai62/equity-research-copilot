"""Offline evidence integrity and reliability status; never scores the final test."""
import json
from pathlib import Path
from src.validation.cohort_v3_data import digest

STUDIES={
 'probability_model':Path('data/processed/validation/recent_study_models_20261019/selection.json'),
 'held_out_policy':Path('data/processed/validation/recent_operating_policy_20261019/report.json'),
 'bootstrap_policy':Path('data/processed/validation/local_reliability_20261019/report.json')}


def policy_passes(partitions):
    return bool(partitions) and all(p.get('all_companies_covered') is True and
        p.get('precision') is not None and p['precision']>=.5 and
        p.get('recall_among_covered') is not None and p['recall_among_covered']>=.5 for p in partitions)


def run():
    checks={};errors=[];records={}
    for name,path in STUDIES.items():
        try:
            record=json.loads(path.read_text());records[name]=record
            for source,sha in {**record['inputs'],**record.get('outputs',{})}.items():
                if digest(source)!=sha:raise ValueError('Changed evidence: '+source)
            if 'alerts_sha256' in record and digest(path.with_name('alerts.csv'))!=record['alerts_sha256']:raise ValueError('Changed policy alerts')
            checks[name]={'integrity_verified':True}
            if 'partitions' in record:checks[name]['all_partitions_meet_precision_recall_with_full_coverage']=policy_passes(record['partitions'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            checks[name]={'integrity_verified':False};errors.append({'study':name,'error':str(exc)})
    passed=checks.get('held_out_policy',{}).get('all_partitions_meet_precision_recall_with_full_coverage',False)
    return {'status':'blocked','production_approval':False,'local_evidence_integrity_verified':not errors,'checks':checks,'errors':errors,
            'reliability_gate_passed':False,
            'blockers':([] if passed else ['The existing policy does not meet precision and recall requirements across all company partitions.'])+[
                'Development companies have been reused for model selection; these results are not independent validation.',
                'Independent outcome review and approved untouched-final evaluation are still required.'],
            'final_evaluation_started':Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json').exists(),
            'limits':'Read-only offline evidence audit. No downloads, AWS deployment, predictions, threshold release or final scoring.'}

if __name__=='__main__':print(json.dumps(run(),indent=2))
