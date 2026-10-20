"""Resume acquisition, exact-file features, and fixed research comparison locally."""
import argparse
import json
from pathlib import Path
import time
from src.validation.cohort_v3_data import digest
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/recent_study_data_20261019')
SEAL=Path('config/recent_study_20261019/experiment_seal.json')


def verify_protocol():
    paths=[Path('config/recent_study_20261019/models.json'),Path('src/validation/recent_study_models.py'),Path('src/validation/recent_study_features.py'),Path(__file__)]
    record={str(p):digest(p) for p in paths}
    if SEAL.exists():
        if json.loads(SEAL.read_text())!=record:raise ValueError('Precommitted experiment code/config changed')
    else:
        with SEAL.open('x') as handle:json.dump(record,handle,indent=2)
    return record


def run(wait=False,timeout=7200):
    seal=verify_protocol();started=time.monotonic()
    if wait:
        while not all((ROOT/(r+'_report.json')).exists() for r in ['training','calibration']):
            if time.monotonic()-started>timeout:raise TimeoutError('Acquisition is still incomplete; rerun to resume')
            time.sleep(10)
    else:
        from src.validation.recent_study_data import prepare
        for role in ['training','calibration']:prepare(role)
    errors={r:json.loads((ROOT/(r+'_report.json')).read_text())['errors'] for r in ['training','calibration']}
    if any(errors.values()):
        write_json(ROOT/'pipeline_status.json',{'status':'acquisition_errors','errors':errors,'production_approval':False})
        raise ValueError('Acquisition errors require review; no model fitted')
    from src.validation.recent_study_features import run as features
    report=features()
    if report['errors']:raise ValueError('Incomplete source filings; no model fitted')
    from src.validation.recent_study_models import run as models
    result=models()
    write_json(ROOT/'pipeline_status.json',{'status':'research_comparison_complete','experiment_seal':seal,'production_approval':False,'model_report':'data/processed/validation/recent_study_models_20261019/selection.json'})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-acquisition',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.wait_for_acquisition),indent=2))
