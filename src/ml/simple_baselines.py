"""Transparent financial ranking baseline; never treats missing inputs as health."""
import numpy as np
import pandas as pd
from scipy.special import expit
from src.ml.robust_models import train_candidate
from src.validation.sec_disclosure_data import TARGET,FEATURES as FINANCIAL
from src.ml.detection_models import FEATURES


def fit_logistic(frame,name):
    columns=FINANCIAL if name=='logistic_financial' else FEATURES
    if name not in {'logistic_financial','logistic_all'}:raise ValueError('Unknown simple baseline')
    return train_candidate(frame,columns,TARGET,'logistic_balanced',sample_weight=frame.design_weight.to_numpy(float)),columns


def altman_proxy(frame):
    # Altman's non-manufacturer Z-double-prime coefficients, without EMS +3.25.
    # Operating-income/asset input is an EBIT proxy; report this limitation.
    assets=pd.to_numeric(frame.Assets,errors='coerce').to_numpy(float)
    debt=pd.to_numeric(frame.Liabilities,errors='coerce').to_numpy(float)
    ratios=frame[['wc_to_ta','re_to_ta','ebit_to_ta']].to_numpy(float)
    valid=np.isfinite(assets)&np.isfinite(debt)&(assets>0)&(debt>0)&np.isfinite(ratios).all(axis=1)
    score=np.full(len(frame),np.nan)
    score[valid]=ratios[valid]@np.array([6.56,3.26,6.72])+1.05*(assets[valid]-debt[valid])/debt[valid]
    return score
