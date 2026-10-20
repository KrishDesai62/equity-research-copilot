"""Paired development-only feature comparison; never scores the consumed final test."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model
from src.ml.design_calibration import WeightedSigmoid, choose_threshold, metrics
from src.ml.disclosure_features import FINANCIAL_FEATURES
from src.ml.robust_models import QuantileClipper, train_candidate
from src.validation.cohort_v3 import FEATURES, fold_for
from src.validation.cohort_v3_data import ROOT, CONFIG, TARGET, digest

OUTPUT = Path('data/processed/validation/development_refinement_20261018')


def development_rows(path):
    with Path(path).open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['role'] in {'training', 'calibration'}]
    frame = pd.DataFrame(rows).replace('', np.nan)
    eligible = frame.eligible.eq('True') & frame.valid_for_training.eq('True')
    frame = frame.loc[eligible].copy()
    numeric = list(dict.fromkeys(FEATURES + [TARGET, 'design_weight', 'Assets', 'Cash', 'Liabilities', 'NetIncomeLoss_ttm', 'OperatingCashFlow_ttm']))
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    assets = frame.Assets.where(frame.Assets > 0)
    frame['accruals_to_assets'] = (frame.NetIncomeLoss_ttm - frame.OperatingCashFlow_ttm) / assets
    frame['net_debt_proxy_to_assets'] = (frame.Liabilities - frame.Cash) / assets
    frame['operating_cash_burn_to_cash'] = (-frame.OperatingCashFlow_ttm).clip(lower=0) / frame.Cash.where(frame.Cash > 0)
    return frame.replace([np.inf, -np.inf], np.nan)


def fit_base(frame, columns, name):
    if name != 'neural_network':
        return train_candidate(frame, columns, TARGET, name, sample_weight=frame.design_weight)
    model = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
        ('clipper', QuantileClipper()), ('scaler', RobustScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1., max_iter=2000,
            batch_size=min(64, len(frame)), random_state=42, early_stopping=False))])
    weight = frame.design_weight.to_numpy(float)
    model.fit(frame[columns], frame[TARGET], classifier__sample_weight=weight / weight.mean())
    return model


def run(output=OUTPUT):
    output = Path(output)
    source = ROOT / 'dataset.csv'
    manifest = json.loads((ROOT / 'dataset_manifest.json').read_text())
    if digest(source) != manifest['dataset_sha256'] or digest(CONFIG) != manifest['config_sha256']:
        raise ValueError('Original dataset/config seal changed')
    if output.exists():
        saved = json.loads((output / 'report.json').read_text())
        if saved['source_sha256'] != digest(source) or saved['artifact_sha256'] != digest(output / 'candidate.joblib') or saved['predictions_sha256'] != digest(output / 'development_predictions.csv'):
            raise ValueError('Saved refinement integrity failure')
        return saved
    config = json.loads(CONFIG.read_text())
    frame = development_rows(source)
    train, cal = [frame.loc[frame.role.eq(role)].copy() for role in ['training', 'calibration']]
    if set(train.cik) & set(cal.cik) or pd.to_datetime(train.label_available_date_1yr).max() >= pd.to_datetime(cal.prediction_date).min():
        raise ValueError('Development company/time leakage')
    folds = cal.cik.map(lambda c: fold_for(c, config['seed'], 3)).to_numpy()
    results, fitted, predictions = [], {}, cal[['unit_id', 'cik', TARGET, 'design_weight']].copy()
    predictions['company_fold'] = folds
    for feature_set, columns in [('original', FEATURES), ('financial_extension', FEATURES + FINANCIAL_FEATURES)]:
        for name in config['models']:
            key = feature_set + ':' + name
            print('Fitting', key, flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                base = fit_base(train, columns, name)
            raw = base.predict_proba(cal[columns])[:, 1]
            scores = np.full(len(cal), np.nan)
            for fold in range(3):
                held = folds == fold
                if held.any():
                    mapper = WeightedSigmoid(base, config['calibration_regularization']).fit_scores(
                        raw[~held], cal[TARGET].to_numpy()[~held], cal.design_weight.to_numpy()[~held])
                    scores[held] = mapper.predict_scores(raw[held])
            threshold = choose_threshold(cal[TARGET], scores, cal.design_weight, .5, .5)
            measured = metrics(cal[TARGET], scores, cal.design_weight, threshold)
            result = {'candidate': key, 'features': columns, 'metrics': measured, 'threshold': threshold,
                'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                'warnings': sorted(set(str(w.message) for w in caught))}
            results.append(result)
            fitted[key] = base
            predictions[key] = scores
    valid = [r for r in results if r['converged']]
    if not valid:
        raise ValueError('No converged development model')
    selected = min(valid, key=lambda r: (r['metrics']['log_loss'], r['metrics']['brier'], r['candidate']))
    base, columns = fitted[selected['candidate']], selected['features']
    calibrated = WeightedSigmoid(base, config['calibration_regularization']).fit_scores(
        base.predict_proba(cal[columns])[:, 1], cal[TARGET], cal.design_weight)
    output.mkdir(parents=True, exist_ok=False)
    save_model(calibrated, columns, output / 'candidate.joblib', trained_through='2017-12-31', target=TARGET)
    predictions.to_csv(output / 'development_predictions.csv', index=False)
    report = {'source_sha256': digest(source), 'artifact_sha256': digest(output / 'candidate.joblib'),
        'predictions_sha256': digest(output / 'development_predictions.csv'), 'candidates': results, 'selected': selected,
        'training_rows': len(train), 'calibration_rows': len(cal), 'training_positive_companies': int(train.loc[train[TARGET].eq(1)].cik.nunique()),
        'calibration_positive_companies': int(cal.loc[cal[TARGET].eq(1)].cik.nunique()),
        'selection': 'Minimum company-cross-fitted weighted development log loss, then Brier loss; no final evaluation.',
        'final_test_rescored': False, 'production_ready': False,
        'limitations': ['Previously used development cohort; results can be optimistic after repeated development.',
            'No new outcome labels or recent population coverage.', 'Filing-change features unavailable for these disjoint issuers; no missing changes filled as zero.',
            'Neural training budget fixed at 2000 iterations for both feature sets. No candidate was promoted.']}
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
