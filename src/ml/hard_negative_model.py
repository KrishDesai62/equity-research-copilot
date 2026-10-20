"""Mine high-risk negative training rows using held-out companies only."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from src.ml.detection_models import FEATURES,loss_weights
from src.validation.sec_disclosure_data import TARGET
from src.validation.cohort_v3 import fold_for


def classifier():
    return HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=10,learning_rate=.03,early_stopping=False,random_state=42)


def fit(frame,config):
    folds=frame.cik.map(lambda c:fold_for(c,config['mining_seed'],3)).to_numpy()
    raw=np.full(len(frame),np.nan)
    for fold in np.unique(folds):
        held=folds==fold;fitting=frame.loc[~held]
        model=classifier();model.fit(fitting[FEATURES],fitting[TARGET].astype(int),sample_weight=loss_weights(fitting,'balanced'))
        raw[held]=model.predict_proba(frame.loc[held,FEATURES])[:,list(model.classes_).index(1)]
    if not np.isfinite(raw).all():raise ValueError('Incomplete held-company mining')
    y=frame[TARGET].to_numpy(int);hard=(y==0)&(raw>=config['mining_score_threshold'])
    weights=frame.design_weight.to_numpy(float).copy();weights[hard]*=config['hard_negative_multiplier']
    weights[y==1]*=weights[y==0].sum()/weights[y==1].sum();weights/=weights.mean()
    model=classifier();model.fit(frame[FEATURES],y,sample_weight=weights)
    return model,{'mined_negative_rows':int(hard.sum()),'mined_negative_companies':int(frame.loc[hard,'cik'].nunique()),
                  'mining_companies':int(frame.cik.nunique()),'development_rows_used_for_mining':False}
