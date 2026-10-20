"""Native-missing tree candidates and aligned probability ensembles."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml.market_context_models import FEATURES
from src.validation.sec_disclosure_data import TARGET


def loss_weights(frame, mode):
    y=frame[TARGET].to_numpy(int)
    weights=frame.design_weight.to_numpy(float).copy()
    if set(y)!={0,1} or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Expected observed binary outcomes and positive design weights')
    if mode=='company_balanced':
        weights/=frame.groupby('cik').cik.transform('size').to_numpy(float)
    elif mode=='recent_balanced':
        years=pd.to_datetime(frame.prediction_date).dt.year.to_numpy()
        weights*=np.exp2(-(years.max()-years)/3.)
    elif mode not in {'design','sqrt','balanced'}:
        raise ValueError('Unknown training loss weighting')
    ratio=weights[y==0].sum()/weights[y==1].sum()
    if mode=='sqrt':weights[y==1]*=np.sqrt(ratio)
    elif mode.endswith('balanced'):weights[y==1]*=ratio
    return weights/weights.mean()


class ProbabilityVote:
    def __init__(self, models):
        if not models:raise ValueError('An ensemble needs fitted members')
        self.models=list(models);self.classes_=np.array([0,1])

    def predict_proba(self, X):
        probabilities=[]
        for model in self.models:
            classes=list(model.classes_)
            if len(classes)!=2 or set(classes)!={0,1}:raise ValueError('Nonbinary ensemble member')
            matrix=np.asarray(model.predict_proba(X),dtype=float)
            if matrix.shape!=(len(X),2) or not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any() or not np.allclose(matrix.sum(axis=1),1):
                raise ValueError('Invalid ensemble probability distribution')
            probabilities.append(matrix[:,classes.index(1)])
        p=np.mean(probabilities,axis=0)
        return np.column_stack([1-p,p])


def fit_candidate(frame,name):
    if name.startswith('native_'):
        mode=name.removeprefix('native_')
        regularized=mode=='regularized_sqrt'
        weights=loss_weights(frame,'sqrt' if regularized else mode)
        model=HistGradientBoostingClassifier(max_iter=350 if regularized else 250,
            max_leaf_nodes=5 if regularized else 7,min_samples_leaf=25 if regularized else 15,
            l2_regularization=20 if regularized else 10,learning_rate=.03,
            early_stopping=False,random_state=42)
    elif name=='xgb_native_sqrt':
        from xgboost import XGBClassifier
        weights=loss_weights(frame,'sqrt')
        model=XGBClassifier(n_estimators=400,max_depth=3,min_child_weight=5,reg_lambda=20,
            reg_alpha=1,learning_rate=.025,subsample=.8,colsample_bytree=.8,
            random_state=42,n_jobs=2,eval_metric='logloss')
    elif name=='extra_trees_balanced':
        weights=loss_weights(frame,'balanced')
        classifier=ExtraTreesClassifier(n_estimators=500,max_depth=8,min_samples_leaf=3,
            max_features='sqrt',n_jobs=2,random_state=42)
        model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),('classifier',classifier)])
        model.fit(frame[FEATURES],frame[TARGET].astype(int),classifier__sample_weight=weights)
        return model
    else:raise ValueError('Unknown detection candidate')
    model.fit(frame[FEATURES],frame[TARGET].astype(int),sample_weight=weights)
    return model
