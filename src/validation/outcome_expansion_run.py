"""Resume source-checked expansion and fitting without scoring reserved companies."""
import argparse
import json
from pathlib import Path
import time
from src.validation.cohort_v3_data import digest
from src.validation.outcome_expansion_data import ROOT
from src.data_layer.research_dataset import write_json


def run(wait=False):
    if wait:
        start=time.monotonic()
        while not (ROOT/'training_report.json').exists():
            if time.monotonic()-start>7200:raise TimeoutError('Acquisition remains incomplete; resume after checking errors')
            time.sleep(10)
    else:
        from src.validation.outcome_expansion_data import run as acquire
        acquire('training')
    for p,sha in json.loads(Path('config/outcome_expansion_20261019/model_seal.json').read_text()).items():
        if digest(p)!=sha:raise ValueError('Precommitted model experiment changed')
    report=json.loads((ROOT/'training_report.json').read_text())
    if report['errors']:raise ValueError('Acquisition errors remain; do not fit')
    from src.validation.outcome_expansion_features import run as features
    evidence=features()
    if evidence['errors']:raise ValueError('Unresolved feature sources; do not fit')
    from src.validation.outcome_expansion_models import run as models
    result=models()
    write_json(ROOT/'pipeline_status.json',{'status':'research_comparison_complete','fresh_reservation_scored':False,'production_approval':False})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-acquisition',action='store_true');args=parser.parse_args();print(json.dumps(run(args.wait_for_acquisition),indent=2))
