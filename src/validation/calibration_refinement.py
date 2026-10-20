"""Nested calibration study with company exclusion and fixed-seed sensitivity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model,save_model
from src.ml.design_calibration import WeightedSigmoid,choose_threshold,metrics,reliability
from src.ml.detection_models import FEATURES
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.development_errors import operating_limits
from src.validation.detection_development import SOURCE,verify_saved as verify_detection
from src.validation.detection_research_candidate import ROOT as SPECIALIST,run as verify_specialist
from src.validation.market_context_model_development import verify_saved as verify_reference
from src.validation.sec_disclosure_data import TARGET

CONFIG=Path('config/development_20261019/calibration_refinement.json')
ROOT=Path('data/processed/validation/calibration_refinement_20261019')


def select_regularization(raw,y,w,companies,seed,config):
    folds=np.array([fold_for(c,seed,config['inner_folds']) for c in companies])
    results=[]
    for regularization in config['regularization_candidates']:
        p=np.full(len(y),np.nan)
        for fold in np.unique(folds):
            held=folds==fold
            if len(np.unique(y[~held]))!=2:break
            mapper=WeightedSigmoid(None,regularization).fit_scores(raw[~held],y[~held],w[~held])
            p[held]=mapper.predict_scores(raw[held])
        if np.isfinite(p).all():
            results.append((metrics(y,p,w)['log_loss'],regularization))
    if not results:return config['fallback_regularization'],True
    return min(results)[1],False


def nested_scores(raw,y,w,companies,folds,seed,config,method):
    raw=np.asarray(raw,float);y=np.asarray(y,int);w=np.asarray(w,float)
    companies=np.asarray(companies);folds=np.asarray(folds)
    if len(set(companies))!=len(companies):raise ValueError('Expected one calibration prediction per company')
    scores=np.full(len(y),np.nan);details=[]
    for fold in np.unique(folds):
        held=folds==fold
        if len(np.unique(y[~held]))!=2:raise ValueError('Outer calibration fit requires both outcomes')
        if method=='nested':
            regularization,fallback=select_regularization(raw[~held],y[~held],w[~held],companies[~held],seed+':inner:'+str(fold),config)
        elif method=='fixed':regularization,fallback=config['fallback_regularization'],False
        else:raise ValueError('Unknown calibration method')
        mapper=WeightedSigmoid(None,regularization).fit_scores(raw[~held],y[~held],w[~held])
        scores[held]=mapper.predict_scores(raw[held])
        details.append({'fold':int(fold),'regularization':regularization,'fallback':fallback,
                        'slope':float(mapper.calibrator_.coef_[0,0]),'fit_companies':int((~held).sum()),'scored_companies':int(held.sum())})
    return scores,details


def verify_saved():
    result=json.loads((ROOT/'selection.json').read_text())
    if result['config_sha256']!=digest(CONFIG) or result['implementation_sha256']!=digest(__file__):raise ValueError('Frozen calibration experiment changed')
    for path,expected in result['sealed_files'].items():
        if digest(path)!=expected:raise ValueError('Frozen calibration source or artifact changed')
    return result


def run():
    if Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json').exists():raise ValueError('Final evaluation has started; development is closed')
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.fit.lock').open('x') as handle:handle.write('Nested calibration refinement')
    try:
        with threadpool_limits(limits=2):return _run()
    finally:(ROOT/'.fit.lock').unlink()


def _run():
    if (ROOT/'selection.json').exists():return verify_saved()
    reference=verify_reference(SOURCE);specialist=verify_specialist();verify_detection()
    config=json.loads(CONFIG.read_text())
    cal=pd.read_csv(SOURCE/'calibration_inputs.csv',dtype={'cik':str})
    y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float);companies=cal.cik.to_numpy()
    paths={'probability_reference':SOURCE/'candidate.joblib','detection_specialist':SPECIALIST/'research_detection.joblib'}
    hashes={'probability_reference':reference['sealed_files'][str(paths['probability_reference'])], 'detection_specialist':specialist['artifact_sha256']}
    models={};candidates=[];prediction=cal[['cik','unit_id',TARGET,'design_weight']].copy();baseline_replay={}
    for name,path in paths.items():
        payload=load_model(path,expected_sha256=hashes[name])
        if payload['target']!=TARGET or payload['feature_cols']!=FEATURES:raise ValueError('Calibration target/schema mismatch')
        base=payload['model'].model;models[name]=base
        raw=base.predict_proba(cal[FEATURES])[:,list(base.classes_).index(1)]
        for method in config['methods']:
            print('Calibration refinement',name,method,flush=True)
            results=[];fold_details=[]
            for number,suffix in enumerate(config['sensitivity_suffixes']):
                seed=config['seed']+suffix
                folds=np.array([fold_for(c,seed,config['outer_folds']) for c in companies])
                p,details=nested_scores(raw,y,w,companies,folds,seed,config,method)
                threshold=choose_threshold(y,p,w,config['minimum_precision'],config['minimum_recall'])
                column=name+'__'+method+'__'+str(number);prediction[column]=p
                results.append({'partition':number,'metrics':metrics(y,p,w,threshold),'threshold':threshold,
                                'operating_limits':operating_limits(prediction,column)})
                fold_details.append(details)
            candidate={'name':name+'__'+method,'base_model':name,'method':method,'partitions':results,'fold_details':fold_details,
                       'median_log_loss':float(np.median([r['metrics']['log_loss'] for r in results])),
                       'median_brier':float(np.median([r['metrics']['brier'] for r in results])),
                       'all_calibration_slopes_positive':all(d['slope']>0 for ds in fold_details for d in ds)}
            candidates.append(candidate)
            if method=='fixed':
                original=pd.read_csv(SOURCE/'calibration_predictions.csv') if name=='probability_reference' else pd.read_csv('data/processed/validation/detection_models_20261019/calibration_predictions.csv')
                old_column=reference['selected']['name'] if name=='probability_reference' else specialist['candidate']['name']
                baseline_replay[name]=bool(np.allclose(prediction[name+'__fixed__0'],original[old_column],rtol=1e-9,atol=1e-10))
    if not all(baseline_replay.values()):raise ValueError('Frozen reference calibration scores did not reproduce')
    eligible=[c for c in candidates if c['all_calibration_slopes_positive']]
    if not eligible:raise ValueError('No monotonic calibration candidate')
    chosen=min(eligible,key=lambda c:(c['median_log_loss'],c['median_brier'],c['name']))
    base=models[chosen['base_model']];raw=base.predict_proba(cal[FEATURES])[:,1]
    regularization,fallback=(select_regularization(raw,y,w,companies,config['seed']+':full-refit',config)
                             if chosen['method']=='nested' else (config['fallback_regularization'],False))
    mapper=WeightedSigmoid(base,regularization).fit_scores(raw,y,w)
    if mapper.calibrator_.coef_[0,0]<=0:raise ValueError('Final research mapper is not monotonic')
    save_model(mapper,FEATURES,ROOT/'research_candidate.joblib',trained_through=cal.label_available_date.max(),target=TARGET)
    prediction.to_csv(ROOT/'development_predictions.csv',index=False)
    sealed=[*paths.values(),SOURCE/'selection.json',SOURCE/'calibration_inputs.csv',SOURCE/'calibration_predictions.csv',
            SPECIALIST/'report.json',Path('data/processed/validation/detection_models_20261019/selection.json'),
            Path('data/processed/validation/detection_models_20261019/calibration_predictions.csv'),
            ROOT/'research_candidate.joblib',ROOT/'development_predictions.csv',Path('src/ml/design_calibration.py'),Path('src/validation/cohort_v3.py'),Path('requirements-local.lock')]
    primary=chosen['partitions'][0];column=chosen['name']+'__0'
    result={'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__),
            'sealed_files':{str(p):digest(p) for p in sealed},'selected':chosen,'candidates':candidates,
            'primary_partition':primary,'selected_calibration_bins':reliability(y,prediction[column],w),
            'final_mapper_regularization':regularization,'final_mapper_fallback':fallback,
            'baseline_replay':baseline_replay,'feature_cols':FEATURES,'development_companies':len(cal),
            'release_threshold':None,'production_approval':False,'final_rows_loaded':False,
            'limits':config['limits']}
    write_json(ROOT/'selection.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
