"""Actual base/expanded text-feature experiments; never reads final feature rows."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model
from src.ml.design_calibration import WeightedSigmoid,choose_threshold,metrics,reliability
from src.ml.text_signal_models import FEATURES,fit_candidate
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.filing_text_signals import ROOT as TEXT,prepare as prepare_text,checkpoint_map
from src.validation.sec_filing_signals import FEATURES as FILING,read_metadata,signals_at
from src.validation.sec_disclosure_data import TARGET
from src.validation.sec_disclosure_models_v4 import validate_development

CONFIG=Path('config/development_20261019/models.json')
ROOT=Path('data/processed/validation/text_models_20261019')


def enrich(frame,checkpoints):
    rows={};values=[]
    for row in frame.itertuples():
        if row.cik not in rows:rows[row.cik],_=read_metadata(row.cik,checkpoints[row.cik])
        values.append(signals_at(rows[row.cik],row.prediction_date))
    return pd.concat([frame.drop(columns=FILING+['filing_signals_latest_public_date'],errors='ignore').reset_index(drop=True),pd.DataFrame(values)],axis=1)


def crossfit_baseline(y,weights,folds):
    result=np.full(len(y),np.nan)
    for fold in np.unique(folds):
        held=folds==fold
        if not (~held).any():raise ValueError('Empty baseline calibration training fold')
        result[held]=np.average(y[~held],weights=weights[~held])
    return result


def verify_saved(folder):
    saved=json.loads((folder/'selection.json').read_text())
    if saved['config_sha256']!=digest(CONFIG) or saved['implementation_sha256']!=digest(__file__):raise ValueError('Frozen text experiment changed')
    for path,expected in saved['sealed_files'].items():
        if digest(path)!=expected:raise ValueError('Frozen text experiment input/artifact changed')
    return saved


def run(phase):
    if phase not in {'base','expanded'}:raise ValueError('Invalid training phase')
    folder=ROOT/phase;folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'.fit.lock'
    with lock.open('x') as handle:handle.write('Text model fitting')
    try:return _run(phase,folder)
    finally:lock.unlink()


def _run(phase,folder):
    if Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json').exists():raise ValueError('Final consumed; no more development fits for this reservation')
    if (folder/'selection.json').exists():return verify_saved(folder)
    report=prepare_text(phase)
    if report['errors']:raise ValueError('Resolve all text acquisition errors before fitting')
    plan=json.loads((TEXT/(phase+'_acquisition.json')).read_text())
    config=json.loads(CONFIG.read_text());checkpoints=checkpoint_map()
    train,cal=[enrich(pd.read_csv(TEXT/(phase+'_'+r+'.csv'),dtype={'cik':str}),checkpoints) for r in ['training','calibration']]
    validate_development(train,cal)
    for frame in [train,cal]:
        if frame[FEATURES].isna().all(axis=1).any() or (pd.to_datetime(frame.filing_signals_latest_public_date)>=pd.to_datetime(frame.prediction_date)).any():raise ValueError('Missing/future feature evidence')
        for row in frame.itertuples():
            doc=plan['documents'][plan['unit_document_map'][row.unit_id]]
            if doc['filing_date']>=row.prediction_date:raise ValueError('Future text document')
    train.to_csv(folder/'training_inputs.csv',index=False);cal.to_csv(folder/'calibration_inputs.csv',index=False)
    folds=cal.cik.map(lambda c:fold_for(c,config['seed'],config['calibration_folds'])).to_numpy()
    y,w=cal[TARGET].to_numpy(int),cal.design_weight.to_numpy(float)
    predictions=cal[['cik','unit_id',TARGET,'design_weight']].copy()
    predictions['constant_oof']=crossfit_baseline(y,w,folds)
    candidates,fitted=[],{}
    for name in config['models']:
        print('Text model',phase,name,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');base=fit_candidate(train,name)
        raw=base.predict_proba(cal[FEATURES])[:,1];scores=np.full(len(cal),np.nan)
        for fold in range(config['calibration_folds']):
            held=folds==fold
            if held.any():
                mapper=WeightedSigmoid(base,config['calibration_regularization']).fit_scores(raw[~held],y[~held],w[~held])
                scores[held]=mapper.predict_scores(raw[held])
        threshold=choose_threshold(y,scores,w,config['minimum_precision'],config['minimum_recall'])
        candidates.append({'name':name,'threshold':threshold,'metrics':metrics(y,scores,w,threshold),
            'converged':not any(issubclass(item.category,ConvergenceWarning) for item in caught),
            'warnings':sorted({str(item.message) for item in caught})})
        fitted[name]=base;predictions[name]=scores
    valid=[c for c in candidates if c['converged']]
    feasible=[c for c in valid if c['threshold'] is not None]
    if not valid:raise ValueError('No converged text candidate')
    chosen=min(feasible or valid,key=lambda c:(c['metrics']['log_loss'],c['metrics']['brier'],c['name']))
    base=fitted[chosen['name']]
    wrapper=WeightedSigmoid(base,config['calibration_regularization']).fit_scores(base.predict_proba(cal[FEATURES])[:,1],y,w)
    save_model(wrapper,FEATURES,folder/'candidate.joblib',trained_through=cal.label_available_date.max(),target=TARGET)
    predictions.to_csv(folder/'calibration_predictions.csv',index=False)
    source_paths=[folder/'candidate.joblib',folder/'calibration_predictions.csv',folder/'training_inputs.csv',folder/'calibration_inputs.csv',
        TEXT/(phase+'_report.json'),TEXT/(phase+'_acquisition.json'),Path('src/ml/text_signal_models.py'),
        Path('src/validation/filing_text_signals.py'),Path('src/validation/sec_filing_signals.py'),Path('requirements-filing-text.lock'),
        Path('src/ml/design_calibration.py'),Path('src/ml/robust_models.py')]
    for role in ['training','calibration']:source_paths.append(TEXT/(phase+'_'+role+'.csv'))
    description=lambda f:{'rows':len(f),'companies':int(f.cik.nunique()),'positive_companies':int(f.loc[f[TARGET].eq(1),'cik'].nunique())}
    result={'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__),'phase':phase,
        'sealed_files':{str(p):digest(p) for p in source_paths},'selected':chosen,'candidates':candidates,
        'training':description(train),'calibration':description(cal),'features':FEATURES,
        'constant_baseline_oof':metrics(y,predictions.constant_oof,w),
        'selected_calibration_bins':reliability(y,predictions[chosen['name']],w),
        'development_feasible':bool(feasible),'constant_for_future_test':float(np.average(y,weights=w)),
        'reference_base_training_rows':299,'final_rows_loaded':False,'production_ready':False,'limits':config['limits']}
    write_json(folder/'selection.json',result);return result


def select_phase():
    results=[verify_saved(ROOT/phase) for phase in ['base','expanded']]
    a,b=[pd.read_csv(ROOT/r['phase']/'calibration_inputs.csv',dtype={'cik':str}) for r in results]
    if not a[['unit_id',TARGET,'design_weight']+FEATURES].equals(b[['unit_id',TARGET,'design_weight']+FEATURES]):raise ValueError('Different development cohorts across phases')
    feasible=[r for r in results if r['development_feasible']]
    chosen=min(feasible or results,key=lambda r:(r['selected']['metrics']['log_loss'],r['selected']['metrics']['brier'],r['phase']))
    output={'phase':chosen['phase'],'selected':chosen['selected'],'development_feasible':bool(feasible),
        'phase_selection_hashes':{r['phase']:digest(ROOT/r['phase']/'selection.json') for r in results},
        'final_used_for_selection':False,'production_ready':False}
    path=ROOT/'frozen_selection.json'
    if path.exists() and json.loads(path.read_text())!=output:raise ValueError('Frozen phase selection changed')
    if not path.exists():write_json(path,output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['base','expanded','select'],default='base');args=parser.parse_args()
    print(json.dumps(select_phase() if args.phase=='select' else run(args.phase),indent=2))
