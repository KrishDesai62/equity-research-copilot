"""Fixed detection experiment using frozen development inputs; no final I/O."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model,save_model
from src.ml.design_calibration import WeightedSigmoid,choose_threshold,metrics,reliability
from src.ml.detection_models import FEATURES,ProbabilityVote,fit_candidate
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.development_errors import operating_limits
from src.validation.market_context_model_development import ROOT as PREVIOUS,verify_saved as verify_previous
from src.validation.sec_disclosure_data import TARGET
from src.validation.sec_disclosure_models_v4 import validate_development

CONFIG=Path('config/development_20261019/detection_models.json')
ROOT=Path('data/processed/validation/detection_models_20261019')
SOURCE=PREVIOUS/'expanded'


def verify_saved():
    record=json.loads((ROOT/'selection.json').read_text())
    if record['config_sha256']!=digest(CONFIG) or record['implementation_sha256']!=digest(__file__):
        raise ValueError('Frozen detection experiment changed')
    for path,expected in record['sealed_files'].items():
        if digest(path)!=expected:raise ValueError('Frozen detection input or artifact changed')
    return record


def crossfit(base,cal,config):
    y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float)
    raw=base.predict_proba(cal[FEATURES])[:,list(base.classes_).index(1)]
    folds=cal.cik.map(lambda c:fold_for(c,config['seed'],config['calibration_folds'])).to_numpy()
    p=np.full(len(cal),np.nan)
    for fold in np.unique(folds):
        held=folds==fold
        mapper=WeightedSigmoid(base,config['calibration_regularization']).fit_scores(raw[~held],y[~held],w[~held])
        p[held]=mapper.predict_scores(raw[held])
    return p


def run():
    marker=Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json')
    if marker.exists():raise ValueError('Final evaluation already started; do not tune on this reservation')
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.fit.lock').open('x') as handle:handle.write('Detection experiment')
    try:
        with threadpool_limits(limits=2):return _run()
    finally:(ROOT/'.fit.lock').unlink()


def _run():
    if (ROOT/'selection.json').exists():return verify_saved()
    previous=verify_previous(SOURCE)
    config=json.loads(CONFIG.read_text())
    train,cal=[pd.read_csv(SOURCE/(r+'_inputs.csv'),dtype={'cik':str}) for r in ['training','calibration']]
    validate_development(train,cal)
    if train[FEATURES].isna().all(axis=1).any() or cal[FEATURES].isna().all(axis=1).any():raise ValueError('Empty model inputs')
    y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float)
    predictions=cal[['cik','unit_id',TARGET,'design_weight']].copy();fitted={};candidates=[]
    for name in config['models']:
        print('Detection model',name,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if name=='frozen_current_reference':
                path=SOURCE/'candidate.joblib'
                artifact=load_model(path,expected_sha256=previous['sealed_files'][str(path)])
                if artifact['feature_cols']!=FEATURES or artifact['target']!=TARGET:raise ValueError('Reference target/schema changed')
                model=artifact['model'].model
            elif name in config['ensembles']:
                model=ProbabilityVote([fitted[member] for member in config['ensembles'][name]])
            else:model=fit_candidate(train,name)
        fitted[name]=model;scores=crossfit(model,cal,config);predictions[name]=scores
        threshold=choose_threshold(y,scores,w,config['minimum_precision'],config['minimum_recall'])
        candidates.append({'name':name,'threshold':threshold,'metrics':metrics(y,scores,w,threshold),
            'operating_limits':operating_limits(predictions,name),
            'converged':not any(issubclass(item.category,ConvergenceWarning) for item in caught),
            'warnings':sorted({str(item.message) for item in caught})})
    old=pd.read_csv(SOURCE/'calibration_predictions.csv',dtype={'cik':str})
    if not old[['cik','unit_id',TARGET,'design_weight']].equals(predictions[['cik','unit_id',TARGET,'design_weight']]):raise ValueError('Development cohort changed')
    if not np.allclose(predictions.frozen_current_reference,old[previous['selected']['name']],rtol=1e-9,atol=1e-10):raise ValueError('Frozen baseline scores did not reproduce')
    valid=[c for c in candidates if c['converged']];feasible=[c for c in valid if c['threshold'] is not None]
    if not valid:raise ValueError('No converged candidate')
    chosen=min(feasible or valid,key=lambda c:(c['metrics']['log_loss'],c['metrics']['brier'],c['name']))
    base=fitted[chosen['name']]
    calibrated=WeightedSigmoid(base,config['calibration_regularization']).fit_scores(base.predict_proba(cal[FEATURES])[:,1],y,w)
    save_model(calibrated,FEATURES,ROOT/'candidate.joblib',trained_through=cal.label_available_date.max(),target=TARGET)
    predictions.to_csv(ROOT/'calibration_predictions.csv',index=False)
    paths=[SOURCE/'selection.json',SOURCE/'candidate.joblib',SOURCE/'training_inputs.csv',SOURCE/'calibration_inputs.csv',SOURCE/'calibration_predictions.csv',
           ROOT/'candidate.joblib',ROOT/'calibration_predictions.csv',Path('src/ml/detection_models.py'),Path('src/ml/design_calibration.py'),Path('src/ml/market_context_models.py'),Path('requirements-filing-text.lock')]
    result={'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__),
        'sealed_files':{str(p):digest(p) for p in paths},'selected':chosen,'candidates':candidates,
        'development_feasible':bool(feasible),'training':previous['training'],'calibration':previous['calibration'],
        'features':FEATURES,'selected_calibration_bins':reliability(y,predictions[chosen['name']],w),
        'baseline_scores_reproduced':True,'evaluation_weights_unchanged':True,
        'final_rows_loaded':False,'production_ready':False,'limits':config['limits']}
    write_json(ROOT/'selection.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
