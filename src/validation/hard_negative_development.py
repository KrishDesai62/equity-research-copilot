"""Fixed same-cohort comparison after newer public-data acquisition completes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model,save_model
from src.ml.design_calibration import WeightedSigmoid,choose_threshold,metrics
from src.ml.detection_models import FEATURES as BASE_FEATURES,loss_weights
TRENDS=[]
from src.ml.hard_negative_model import fit as fit_hard_negative
from sklearn.ensemble import HistGradientBoostingClassifier
from src.validation.threshold_policy import held_out_policy,summarize_policy
FEATURES=BASE_FEATURES+TRENDS
from src.validation.calibration_refinement import nested_scores,select_regularization
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.development_errors import operating_limits
from src.validation.sec_disclosure_data import TARGET
from src.validation.sec_disclosure_models_v4 import validate_development
from src.validation.recent_study_features import ROOT as SOURCE

ROOT=Path('data/processed/validation/hard_negative_20261019')
CONFIG=Path('config/hard_negative_20261019/models.json')
REFERENCES={'frozen_reference':Path('data/processed/validation/recent_study_models_20261019/research_candidate.joblib')}


def run():
    if Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json').exists():raise ValueError('Final evaluation has started')
    from src.validation.recent_study_features import run as verify_features
    report=verify_features()
    if report.get('errors'):raise ValueError('Incomplete feature evidence')
    config=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    paths=[CONFIG,Path(__file__),SOURCE/'report.json',SOURCE/'training.csv',SOURCE/'calibration.csv',*REFERENCES.values(),Path('src/ml/detection_models.py'),Path('src/validation/calibration_refinement.py'),Path('src/ml/design_calibration.py'),Path('src/ml/hard_negative_model.py'),Path('src/validation/threshold_policy.py')]
    inputs={str(p):digest(p) for p in paths};saved=ROOT/'selection.json'
    if saved.exists():
        result=json.loads(saved.read_text())
        if result['inputs']!=inputs:raise ValueError('Frozen new study changed')
        for p,sha in result['outputs'].items():
            if digest(p)!=sha:raise ValueError('Frozen model output changed')
        return result
    train,cal=[pd.read_csv(SOURCE/(r+'.csv'),dtype={'cik':str}) for r in ['training','calibration']]
    validate_development(train,cal)
    if len(cal.loc[cal[TARGET].eq(1)])<10:raise ValueError('Insufficient newer positive company support for planned comparison')
    y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float);companies=cal.cik.to_numpy()
    predictions=cal[['cik','unit_id',TARGET,'design_weight']].copy();candidates=[];models={}
    with threadpool_limits(limits=2):
        for name in config['models']:
            print('Newer model',name,flush=True)
            if name in REFERENCES:
                artifact=load_model(REFERENCES[name],expected_sha256=inputs[str(REFERENCES[name])])
                if artifact['target']!=TARGET or artifact['feature_cols']!=BASE_FEATURES:raise ValueError('Reference schema mismatch')
                base=artifact['model'].model
            else:
                base,mining=fit_hard_negative(train,config)
                write_json(ROOT/'mining_audit.json',mining)
            models[name]=base;columns=BASE_FEATURES if name=='frozen_reference' else FEATURES
            raw=base.predict_proba(cal[columns])[:,list(base.classes_).index(1)]
            partitions=[];details=[]
            for number,suffix in enumerate(config['sensitivity_suffixes']):
                seed=config['seed']+suffix;folds=np.array([fold_for(c,seed,config['outer_folds']) for c in companies])
                p,fold_details=nested_scores(raw,y,w,companies,folds,seed,config,'nested')
                column=name+'__'+str(number);predictions[column]=p
                threshold=choose_threshold(y,p,w,config['minimum_precision'],config['minimum_recall'])
                alerts,policy_details=held_out_policy(raw,y,w,companies,folds,config['minimum_precision'],config['minimum_recall'])
                partitions.append({'metrics':metrics(y,p,w,threshold),'threshold':threshold,'operating_limits':operating_limits(predictions,column),'held_out_policy':summarize_policy(y,w,alerts),'policy_fold_details':policy_details})
                details.extend(fold_details)
            candidates.append({'name':name,'partitions':partitions,'median_log_loss':float(np.median([p['metrics']['log_loss'] for p in partitions])),
                               'monotonic':all(d['slope']>0 for d in details),'fold_details':details})
        eligible=[c for c in candidates if c['monotonic']]
        if not eligible:raise ValueError('No monotonic research candidate')
        stable=[c for c in eligible if all(p['held_out_policy']['all_companies_covered'] and p['held_out_policy']['precision'] is not None and p['held_out_policy']['precision']>=.5 and p['held_out_policy']['recall_among_covered'] is not None and p['held_out_policy']['recall_among_covered']>=.5 for p in c['partitions'])]
        chosen=min(stable or eligible,key=lambda c:(c['median_log_loss'],c['name']));base=models[chosen['name']]
        columns=BASE_FEATURES if chosen['name']=='frozen_reference' else FEATURES
        raw=base.predict_proba(cal[columns])[:,list(base.classes_).index(1)]
        regularization,_=select_regularization(raw,y,w,companies,config['seed']+':full-refit',config)
        calibrated=WeightedSigmoid(base,regularization).fit_scores(raw,y,w)
        if calibrated.calibrator_.coef_[0,0]<=0:raise ValueError('Nonmonotonic full mapper')
        save_model(calibrated,columns,ROOT/'research_candidate.joblib',trained_through=cal.label_available_date.max(),target=TARGET)
    predictions.to_csv(ROOT/'development_predictions.csv',index=False)
    result={'inputs':inputs,'candidates':candidates,'selected':chosen,'training_rows':len(train),'training_positive_companies':int(train.loc[train[TARGET].eq(1),'cik'].nunique()),
            'development_companies':len(cal),'development_positive_companies':int(y.sum()),'regularization':regularization,
            'outputs':{str(p):digest(p) for p in [ROOT/'research_candidate.joblib',ROOT/'development_predictions.csv']},
            'production_approval':False,'release_threshold':None,'final_rows_used':False,
            'feature_cols':columns,'stable_candidate_on_all_partitions':bool(stable),
            'limits':'Fixed training-only hard-negative research comparison on reused development, not independent validation. SEC disclosure target; conditional population and complete-followup exclusions limit generalization.'}
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
