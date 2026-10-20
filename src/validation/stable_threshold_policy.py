"""Bootstrap threshold consensus using fitting companies only."""
import numpy as np
from src.ml.design_calibration import choose_threshold, validate_weights


def consensus_threshold(raw,y,weights,config):
    y,w=validate_weights(y,weights);raw=np.asarray(raw,float)
    if raw.shape!=y.shape or not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():raise ValueError('Invalid raw scores')
    n=config['bootstrap_replicates'];q=config['threshold_quantile'];fraction=config['minimum_feasible_fraction']
    if not isinstance(n,int) or n<1 or not 0<=q<=1 or not 0<=fraction<=1:raise ValueError('Invalid consensus configuration')
    rng=np.random.default_rng(config['seed']);thresholds=[]
    for _ in range(n):
        indices=rng.integers(0,len(y),size=len(y))
        threshold=choose_threshold(y[indices],raw[indices],w[indices],config['minimum_precision'],config['minimum_recall'])
        if threshold is not None:thresholds.append(threshold)
    support=len(thresholds)/n
    threshold=float(np.quantile(thresholds,q,method='higher')) if thresholds and support>=fraction else None
    return threshold,{'feasible_bootstrap_fraction':support,'bootstrap_replicates':n,'threshold':threshold,'abstained':threshold is None}


def crossfit_consensus(raw,y,weights,companies,folds,config):
    y,w=validate_weights(y,weights);raw=np.asarray(raw,float);companies=np.asarray(companies);folds=np.asarray(folds)
    if raw.shape!=y.shape or companies.shape!=y.shape or folds.shape!=y.shape:raise ValueError('Mismatched consensus inputs')
    if len(set(companies))!=len(y) or len(set(folds))<2:raise ValueError('Distinct companies and multiple folds required')
    if not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():raise ValueError('Invalid scores')
    alerts=np.full(len(y),np.nan);details=[]
    for fold in np.unique(folds):
        held=folds==fold
        threshold,detail=consensus_threshold(raw[~held],y[~held],w[~held],{**config,'seed':config['seed']+int(fold)})
        if threshold is not None:alerts[held]=(raw[held]>=threshold).astype(float)
        details.append({'fold':int(fold),'fit_companies':int((~held).sum()),'scored_companies':int(held.sum()),**detail})
    return alerts,details
