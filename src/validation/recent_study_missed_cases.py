"""Descriptive audit of existing development misses; does not retune a model."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET
from src.ml.detection_models import FEATURES
from src.data_layer.research_dataset import write_json

ROOT=Path('data/processed/validation/missed_cases_20261019')


def run():
    report=Path('data/processed/validation/detection_specialist_20261019/report.json')
    scores=Path('data/processed/validation/detection_models_20261019/calibration_predictions.csv')
    inputs=Path('data/processed/validation/market_context_models_20261019/expanded/calibration_inputs.csv')
    spec=json.loads(report.read_text());threshold=spec['diagnostic_operating_point']['threshold'];name=spec['candidate']['name']
    predictions=pd.read_csv(scores,dtype={'cik':str});frame=pd.read_csv(inputs,dtype={'cik':str})
    if set(predictions.unit_id)!=set(frame.unit_id):raise ValueError('Audit cohort mismatch')
    frame=frame.merge(predictions[['unit_id',name]],on='unit_id',validate='one_to_one').copy()
    frame['detected_at_archived_diagnostic_threshold']=frame[name].ge(threshold)|np.isclose(frame[name],threshold,rtol=0,atol=1e-14)
    frame['financial_feature_age_days']=(pd.to_datetime(frame.prediction_date)-pd.to_datetime(frame.source_filing_date)).dt.days
    frame['missing_model_feature_count']=frame[FEATURES].isna().sum(axis=1)
    frame['has_context_warning']=frame[['context_going_concern_rate','context_liquidity_shortfall_rate','context_covenant_default_rate','context_forbearance_rate']].fillna(0).gt(0).any(axis=1)
    positives=frame.loc[frame[TARGET].eq(1)].copy();misses=positives.loc[~positives.detected_at_archived_diagnostic_threshold]
    if len(misses)!=spec['diagnostic_operating_point']['unweighted_confusion']['fn']:raise ValueError('Archived miss count did not reproduce')
    cols=['cik','company_name','unit_id','prediction_date','source_filing_date','financial_feature_age_days','label_available_date','label_reason',name,'design_weight','detected_at_archived_diagnostic_threshold','missing_model_feature_count','has_context_warning',*FEATURES]
    cols=list(dict.fromkeys(c for c in cols if c in positives))
    ROOT.mkdir(parents=True,exist_ok=True);positives[cols].sort_values(name).to_csv(ROOT/'positive_case_audit.csv',index=False)
    result={'inputs':{str(p):digest(p) for p in [report,scores,inputs,Path(__file__)]},'positive_companies':len(positives),'missed_companies':len(misses),
            'misses_with_context_warning':int(misses.has_context_warning.sum()),'misses_with_missing_features':int(misses.missing_model_feature_count.gt(0).sum()),
            'misses_financial_feature_age_over_120_days':int(misses.financial_feature_age_days.gt(120).sum()),
            'diagnostic_threshold':threshold,'csv_threshold_comparison_absolute_tolerance':1e-14,'threshold_was_not_a_release_threshold':True,'final_rows_used':False,
            'limitations':'Descriptive reused-development audit. A warning signal is a lexical proxy, not proof of predictive usefulness or a verified explanation. No retraining or threshold selection.'}
    write_json(ROOT/'report.json',result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
