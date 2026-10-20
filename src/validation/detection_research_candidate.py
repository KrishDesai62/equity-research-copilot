"""Export a development-selected detection specialist, never a release model.

This operating-point analysis is post hoc on repeatedly used development data.
The production precision/recall floors remain unchanged.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model
from src.ml.design_calibration import WeightedSigmoid,choose_threshold,metrics
from src.ml.detection_models import FEATURES,fit_candidate
from src.validation.cohort_v3_data import digest
from src.validation.detection_development import ROOT as STUDY,SOURCE,CONFIG,crossfit,verify_saved
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/detection_specialist_20261019')


def choose_specialist(candidates):
    eligible=[c for c in candidates if c['converged'] and c['name'] not in {'frozen_current_reference','native_vote','diverse_vote'}]
    if not eligible:raise ValueError('No converged single-model specialist')
    return max(eligible,key=lambda c:(c['operating_limits']['maximum_recall_at_precision_50pct'],
                                      c['metrics']['average_precision'],-c['metrics']['log_loss']))


def run():
    result=verify_saved();chosen=choose_specialist(result['candidates'])
    ROOT.mkdir(parents=True,exist_ok=True);report_path=ROOT/'report.json'
    inputs={str(p):digest(p) for p in [STUDY/'selection.json',STUDY/'calibration_predictions.csv',SOURCE/'training_inputs.csv',SOURCE/'calibration_inputs.csv',Path(__file__),Path('src/ml/detection_models.py'),CONFIG]}
    if report_path.exists():
        saved=json.loads(report_path.read_text())
        if saved['inputs']!=inputs or saved['artifact_sha256']!=digest(ROOT/'research_detection.joblib'):
            raise ValueError('Frozen research specialist changed')
        return saved
    config=json.loads(CONFIG.read_text())
    train,cal=[pd.read_csv(SOURCE/(r+'_inputs.csv'),dtype={'cik':str}) for r in ['training','calibration']]
    with threadpool_limits(limits=2):
        model=fit_candidate(train,chosen['name']);scores=crossfit(model,cal,config)
        saved=pd.read_csv(STUDY/'calibration_predictions.csv')
        if not np.allclose(scores,saved[chosen['name']],rtol=1e-9,atol=1e-10):raise ValueError('Refitted specialist did not reproduce saved predictions')
        y=cal[TARGET].to_numpy(int);w=cal.design_weight.to_numpy(float)
        # This is an operating-curve diagnostic, not a lower release requirement.
        threshold=choose_threshold(y,scores,w,config['minimum_precision'],0.)
        if threshold is None:raise ValueError('Specialist has no positive-support diagnostic point')
        wrapper=WeightedSigmoid(model,config['calibration_regularization']).fit_scores(model.predict_proba(cal[FEATURES])[:,1],y,w)
    artifact=ROOT/'research_detection.joblib'
    save_model(wrapper,FEATURES,artifact,trained_through=cal.label_available_date.max(),target=TARGET)
    alerts=scores>=threshold
    confusion={'tp':int((alerts&(y==1)).sum()),'fp':int((alerts&(y==0)).sum()),
               'fn':int((~alerts&(y==1)).sum()),'tn':int((~alerts&(y==0)).sum())}
    report={'inputs':inputs,'candidate':chosen,'artifact_sha256':digest(artifact),
            'diagnostic_operating_point':{'threshold':threshold,'metrics':metrics(y,scores,w,threshold),'unweighted_confusion':confusion},
            'release_threshold':chosen['threshold'],'release_precision_floor':config['minimum_precision'],
            'release_recall_floor':config['minimum_recall'],'production_approval':False,
            'refit_predictions_reproduced':True,'final_rows_loaded':False,
            'selection_scope':'Post-hoc development specialist: maximize recall at precision >= .5, then PR-AUC. Not independent validation or a production threshold.',
            'tradeoff':'Improves alert ranking on development; probability loss is worse than the retained current reference.'}
    write_json(report_path,report);return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
