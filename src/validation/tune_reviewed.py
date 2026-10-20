"""Tune only reviewed development firms, then freeze a candidate for a later test."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model
from src.ml.design_calibration import WeightedSigmoid, choose_threshold, metrics
from src.ml.disclosure_features import CHANGE_FEATURES, FINANCIAL_FEATURES
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES
from src.ml.robust_models import train_candidate
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import TARGET
from src.validation.prospective_evaluation import freeze
from src.validation.reviewed_outcomes import load_reviews

MODELS = ['logistic_regularized', 'logistic_balanced', 'xgb_regularized', 'xgb_sqrt_balance', 'neural_network']


def tune(reservation, feature_path, review_path, output, *, as_of):
    plan = json.loads(Path(reservation).read_text())
    roles = {r['cik']: r['role'] for r in plan['companies']}
    reviews = load_reviews(review_path, as_of=as_of)
    features = pd.read_csv(feature_path, dtype={'cik': str})
    if TARGET in features or not (set(features.cik) | set(reviews.cik)) <= {c for c, r in roles.items() if r != 'final'}:
        raise ValueError('Development input must contain only reserved non-final firms and separate reviewed labels')
    frame = features.merge(reviews, on=['cik', 'prediction_date'], validate='one_to_one')
    if len(frame) != len(features) or len(frame) != len(reviews):
        raise ValueError('Every development feature row must have exactly one reviewed outcome')
    frame['role'] = frame.cik.map(roles)
    public, prediction = pd.to_datetime(frame.feature_filing_date), pd.to_datetime(frame.prediction_date)
    latest = pd.to_datetime(frame.latest_input_filing_date)
    if (public.isna().any() or prediction.isna().any() or latest.isna().any() or (public >= prediction).any() or (latest > public).any()
            or ((prediction - public).dt.days > 180).any()):
        raise ValueError('Development inputs contain stale or future public financials')
    if 'change_available_date' in frame and (pd.to_datetime(frame.change_available_date) >= prediction).any():
        raise ValueError('Development disclosure features are from the future')
    if 'design_weight' not in frame:
        frame['design_weight'] = 1.
    train, cal = [frame.loc[frame.role.eq(role)].copy() for role in ['training', 'calibration']]
    if train.empty or cal.empty or pd.to_datetime(train.label_available_date_1yr).max() >= pd.to_datetime(cal.prediction_date).min():
        raise ValueError('Development stages require disjoint completed outcome periods')
    if pd.to_datetime(cal.label_available_date_1yr).max() >= pd.Timestamp(plan['final_start']):
        raise ValueError('Calibration outcomes overlap the reserved final period')
    desired = BASE_FEATURES + EXTRA_FEATURES + FINANCIAL_FEATURES + CHANGE_FEATURES
    included = [c for c in desired if c in train and pd.to_numeric(train[c], errors='raise').notna().any()]
    omitted = [c for c in desired if c not in included]
    if not included:
        raise ValueError('No observed training features')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    candidates, failures, fitted = [], [], {}
    folds = cal.cik.map(lambda c: fold_for(c, plan['seed'], 3)).to_numpy()
    for name in MODELS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                base = train_candidate(train, included, TARGET, name, sample_weight=train.design_weight)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ValueError('Optimization did not converge within the fixed training budget')
            raw = base.predict_proba(cal[included])[:, 1]
            scores = np.full(len(cal), np.nan)
            for fold in range(3):
                held = folds == fold
                if held.any():
                    mapper = WeightedSigmoid(base).fit_scores(raw[~held], cal[TARGET].to_numpy()[~held], cal.design_weight.to_numpy()[~held])
                    scores[held] = mapper.predict_scores(raw[held])
            measured = metrics(cal[TARGET], scores, cal.design_weight)
            threshold = choose_threshold(cal[TARGET], scores, cal.design_weight, .5, .5)
            candidates.append({'name': name, 'metrics': measured, 'threshold': threshold})
            fitted[name] = base
        except (ValueError, RuntimeError) as exc:
            failures.append({'name': name, 'reason': str(exc)})
    if not candidates:
        write_json(out / 'failure.json', {'failures': failures, 'production_ready': False})
        raise ValueError('No valid development candidate; final test remains untouched')
    selected = min(candidates, key=lambda c: (c['metrics']['log_loss'], c['metrics']['brier'], c['name']))
    base = fitted[selected['name']]
    calibrated = WeightedSigmoid(base).fit_scores(base.predict_proba(cal[included])[:, 1], cal[TARGET], cal.design_weight)
    artifact = out / 'candidate.joblib'
    save_model(calibrated, included, artifact, trained_through=str(pd.to_datetime(cal.label_available_date_1yr).max().date()), target=TARGET)
    freeze(reservation, artifact, out / 'frozen_candidate.json', threshold=selected['threshold'])
    report = {'candidates': candidates, 'selected': selected, 'failures': failures, 'features': included,
              'omitted_unobserved_training_features': omitted, 'final_used_for_selection': False,
              'input_sha256': {str(p): sha256(Path(p).read_bytes()).hexdigest() for p in [reservation, feature_path, review_path]},
              'production_ready': False, 'note': 'A completed candidate fit does not imply improvement or validated population calibration.'}
    write_json(out / 'development_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reservation', 'features', 'reviews', 'output', 'as-of']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(tune(args.reservation, args.features, args.reviews, args.output, as_of=args.as_of), indent=2))
