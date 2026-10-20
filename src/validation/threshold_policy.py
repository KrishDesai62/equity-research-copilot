"""Company-held-out threshold diagnostics; no held label selects its own policy."""
import numpy as np
from src.ml.design_calibration import choose_threshold,validate_weights


def held_out_policy(raw,y,weights,companies,folds,min_precision=.5,min_recall=.5):
    y,w=validate_weights(y,weights)
    raw=np.asarray(raw,float);companies=np.asarray(companies);folds=np.asarray(folds)
    if raw.shape!=y.shape or companies.shape!=y.shape or folds.shape!=y.shape:raise ValueError('Mismatched policy inputs')
    if not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():raise ValueError('Invalid raw probabilities')
    if len(set(companies))!=len(companies) or len(set(folds))<2:raise ValueError('Distinct company rows and at least two folds required')
    alerts=np.full(len(y),np.nan);details=[]
    for fold in np.unique(folds):
        held=folds==fold
        threshold=choose_threshold(y[~held],raw[~held],w[~held],min_precision,min_recall)
        if threshold is not None:alerts[held]=(raw[held]>=threshold).astype(float)
        details.append({'fold':int(fold),'threshold':threshold,'fit_companies':int((~held).sum()),'scored_companies':int(held.sum()),'abstained':threshold is None})
    return alerts,details


def summarize_policy(y,weights,alerts):
    y,w=validate_weights(y,weights);a=np.asarray(alerts,float)
    if a.shape!=y.shape or not set(a[np.isfinite(a)])<={0,1}:raise ValueError('Invalid alerts')
    covered=np.isfinite(a);yes=covered&(a==1)
    tp=float(w[yes&(y==1)].sum());fp=float(w[yes&(y==0)].sum());fn=float(w[covered&(a==0)&(y==1)].sum())
    return {'covered_companies':int(covered.sum()),'abstained_companies':int((~covered).sum()),'weighted_coverage':float(w[covered].sum()/w.sum()),
            'precision':tp/(tp+fp) if tp+fp else None,'recall_among_covered':tp/(tp+fn) if tp+fn else None,
            'positive_weight_abstained':float(w[~covered&(y==1)].sum()),'weighted_true_positives':tp,'weighted_false_positives':fp,
            'all_companies_covered':bool(covered.all())}
