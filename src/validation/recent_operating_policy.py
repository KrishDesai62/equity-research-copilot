"""Audit deployable threshold selection on the reused recent development cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.ml.artifacts import load_model
from src.ml.detection_models import FEATURES
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET
from src.validation.threshold_policy import held_out_policy,summarize_policy
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/recent_operating_policy_20261019')


def run():
    root=Path('data/processed/validation/recent_study_models_20261019')
    selection=json.loads((root/'selection.json').read_text())
    artifact=root/'research_candidate.joblib';table=Path('data/processed/validation/recent_study_features_20261019/calibration.csv')
    if digest(table)!=selection['inputs'][str(table)]:raise ValueError('Development data changed')
    config_path=Path('config/recent_study_20261019/models.json');config=json.loads(config_path.read_text())
    if digest(config_path)!=selection['inputs'][str(config_path)]:raise ValueError('Original partition configuration changed')
    payload=load_model(artifact,expected_sha256=selection['outputs'][str(artifact)])
    if payload['target']!=TARGET or payload['feature_cols']!=FEATURES:raise ValueError('Unexpected model contract')
    cal=pd.read_csv(table,dtype={'cik':str});base=payload['model'].model
    with threadpool_limits(limits=2):raw=base.predict_proba(cal[FEATURES])[:,list(base.classes_).index(1)]
    y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float);partitions=[]
    predictions=cal[['cik','unit_id',TARGET,'design_weight']].copy()
    for number,suffix in enumerate(config['sensitivity_suffixes']):
        folds=np.array([fold_for(c,config['seed']+suffix,config['outer_folds']) for c in cal.cik])
        alerts,details=held_out_policy(raw,y,w,cal.cik,folds,config['minimum_precision'],config['minimum_recall'])
        predictions['alerts_'+str(number)]=alerts
        partitions.append({'partition':number,**summarize_policy(y,w,alerts),'folds':details})
    paths=[artifact,table,config_path,root/'selection.json',Path(__file__),Path('src/validation/threshold_policy.py')]
    inputs={str(p):digest(p) for p in paths};ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        result=json.loads(saved.read_text())
        if result['inputs']!=inputs:raise ValueError('Frozen policy study changed')
        if digest(ROOT/'alerts.csv')!=result['alerts_sha256']:raise ValueError('Frozen alerts changed')
        return result
    predictions.to_csv(ROOT/'alerts.csv',index=False)
    result={'inputs':inputs,'partitions':partitions,'alerts_sha256':digest(ROOT/'alerts.csv'),'production_approval':False,'release_threshold':None,'final_rows_used':False,
            'limits':'Threshold chosen only on other companies raw scores, never held outcomes. Infeasible thresholds abstain. Base model was previously selected on this reused development cohort: this is an operating-policy diagnostic, not independent final validation. Historical operating limits were descriptive oracle bounds, not deployable performance.'}
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
