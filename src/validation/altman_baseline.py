"""Matched-coverage ranking comparison; Altman proxy scores are not probabilities."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from src.ml.simple_baselines import altman_proxy
from src.ml.artifacts import load_model
from src.ml.detection_models import FEATURES
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.threshold_policy import held_out_policy,summarize_policy
from src.validation.sec_disclosure_data import TARGET
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/altman_baseline_20261019')


def run():
    table=Path('data/processed/validation/outcome_expansion_features_20261019/calibration.csv')
    original=Path('data/processed/validation/recent_study_models_20261019')
    selection=json.loads((original/'selection.json').read_text());artifact=original/'research_candidate.joblib'
    frame=pd.read_csv(table,dtype={'cik':str});score=altman_proxy(frame);valid=np.isfinite(score)
    matched=frame.loc[valid].copy();z=score[valid]
    payload=load_model(artifact,expected_sha256=selection['outputs'][str(artifact)])
    model=payload['model'].model
    with threadpool_limits(limits=2):reference=model.predict_proba(matched[FEATURES])[:,list(model.classes_).index(1)]
    config_path=Path('config/recent_study_20261019/models.json');config=json.loads(config_path.read_text())
    y=matched[TARGET].to_numpy(int);w=matched.design_weight.to_numpy(float);results={}
    # Strictly monotone bounded encoding used solely by the alert-policy helper.
    # Neither this transform nor -Z is presented as a bankruptcy probability.
    transformed=.5+np.arctan(-z)/np.pi
    for name,ranking,policy_score in [('altman_operating_income_proxy',-z,transformed),('matched_frozen_reference',reference,reference)]:
        parts=[]
        for suffix in config['sensitivity_suffixes']:
            folds=np.array([fold_for(c,config['seed']+suffix,config['outer_folds']) for c in matched.cik])
            alerts,detail=held_out_policy(policy_score,y,w,matched.cik,folds,.5,.5)
            parts.append({'policy':summarize_policy(y,w,alerts),'fold_details':detail})
        results[name]={'weighted_roc_auc':float(roc_auc_score(y,ranking,sample_weight=w)),
           'weighted_pr_auc':float(average_precision_score(y,ranking,sample_weight=w)),'partitions':parts}
    paths=[table,artifact,original/'selection.json',config_path,Path(__file__),Path('src/ml/simple_baselines.py'),Path('src/validation/threshold_policy.py')]
    result={'inputs':{str(p):digest(p) for p in paths},'full_development_companies':len(frame),'covered_companies':int(valid.sum()),
       'missing_score_companies':int((~valid).sum()),'covered_positive_companies':int(y.sum()),'weighted_coverage':float(frame.loc[valid,'design_weight'].sum()/frame.design_weight.sum()),
       'results':results,'formula_source':'https://www.textbiz.org/projects/defaultprediction/zscores.pdf',
       'formula':'6.56 WC/TA + 3.26 RE/TA + 6.72 operating-income-proxy/TA + 1.05 (TA-TL)/TL',
       'limits':'Book-equity nonmanufacturer Z-double-prime coefficients; operating income is an EBIT proxy, not an exact replication. Missing inputs abstain, no imputation. Matched-coverage reused-development diagnostics only; no probability claim, release or final scoring.',
       'production_approval':False,'final_rows_used':False}
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists() and json.loads(saved.read_text())!=result:raise ValueError('Frozen Altman baseline changed')
    write_json(saved,result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
