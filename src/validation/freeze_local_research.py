"""Freeze the retained model and evidence as research; never approves production."""
import json
from pathlib import Path
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/frozen_local_research_20261019')
FILES=[
 'data/processed/validation/recent_study_models_20261019/research_candidate.joblib',
 'data/processed/validation/recent_study_models_20261019/selection.json',
 'data/processed/validation/recent_study_features_20261019/training.csv',
 'data/processed/validation/recent_study_features_20261019/calibration.csv',
 'data/processed/validation/recent_operating_policy_20261019/report.json',
 'data/processed/validation/recent_operating_policy_20261019/alerts.csv',
 'data/processed/validation/simple_baselines_20261019/selection.json',
 'data/processed/validation/altman_baseline_20261019/report.json',
 'data/processed/validation/negative_coverage_audit_20261019/report.json',
 'data/processed/validation/negative_coverage_inventory_20261019/report.json',
 'data/processed/validation/negative_coverage_inventory_20261019/negative_rows.csv',
 'data/processed/validation/negative_coverage_inventory_20261019/unreviewed_current_reports.json',
 'config/recent_study_20261019/models.json','config/release_policy.json',
 'config/outcome_expansion_20261019/reserved_companies.json',
 'src/ml/financial_history.py','src/ml/detection_models.py','src/ml/design_calibration.py',
 'src/validation/threshold_policy.py','src/validation/calibration_refinement.py',
 'src/validation/issuer_identity_admission_v3.py','src/validation/sec_disclosure_data.py',
 'src/validation/negative_coverage_audit.py','src/validation/negative_coverage_inventory.py',
 'src/validation/freeze_local_research.py','requirements-local.lock']


def verify_files(files):
    for path,sha in files.items():
        if digest(path)!=sha:raise ValueError('Frozen research file changed: '+str(path))


def run():
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'manifest.json'
    if saved.exists():
        result=json.loads(saved.read_text());verify_files(result['files']);return result
    model=json.loads(Path(FILES[1]).read_text())
    if digest(FILES[0])!=model['outputs'][FILES[0]]:raise ValueError('Retained model integrity failed')
    paths=set(FILES)
    # Include exact recorded dependencies and output seals from each saved study.
    for name in FILES:
        if name.endswith('.json'):
            record=json.loads(Path(name).read_text())
            if not isinstance(record,dict):continue
            for key in ['inputs','outputs','source_hashes']:
                values=record.get(key,{})
                if isinstance(values,dict):
                    for path,sha in values.items():
                        if isinstance(sha,str) and len(sha)==64:
                            if digest(path)!=sha:raise ValueError('Evidence dependency changed: '+path)
                            paths.add(path)
    result={'files':{p:digest(p) for p in sorted(paths)},'retained_model':FILES[0],
       'target':'target_sec_disclosed_petition_1yr','research_only':True,'production_approval':False,'release_threshold':None,
       'policy':'No final evaluation authorized by this manifest; keep original and fresh reservations unscored. A new study needs a new version, not edits to these artifacts.',
       'blockers':['Held-company precision/recall stability requirement fails.','Negative-label primary-document coverage has known gaps.','Independent label review and adequately supported untouched evaluation remain incomplete.']}
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
