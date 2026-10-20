"""Expanded, strictly development-only model comparison and frozen selection."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model
from src.ml.design_calibration import WeightedSigmoid, choose_threshold, metrics
from src.ml.robust_models import QuantileClipper
from src.validation.cohort_v3 import fold_for
from src.validation.cohort_v3_data import digest
from src.validation.refine_development import fit_base
from src.validation.sec_disclosure_data import FEATURES, TARGET
from src.validation.sec_disclosure_admission import ROOT as ADMISSION, run as admit

CONFIG = Path('config/sec_disclosure_expansion/models_v2.json')
ROOT = Path('data/processed/validation/sec_disclosure_models_v2')
OLD_TARGET = 'target_bankruptcy_next_1yr'


def train_extra(frame, name):
    y = frame[TARGET].to_numpy(int)
    weights = frame.design_weight.to_numpy(float)
    if not set(y) == {0, 1} or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid expanded training labels/weights')
    weights = weights / weights.mean()
    balance = {c: weights.sum() / (2 * weights[y == c].sum()) for c in (0, 1)}
    if name == 'logistic_c1_balanced':
        classifier = LogisticRegression(C=1., class_weight=balance, max_iter=2000, random_state=42)
    elif name == 'hist_gradient_balanced':
        classifier = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=7,
            learning_rate=.03, min_samples_leaf=15, l2_regularization=10, early_stopping=False, random_state=42)
        weights *= np.array([balance[c] for c in y])
    elif name == 'random_forest_balanced':
        classifier = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=4,
            max_features='sqrt', n_jobs=2, random_state=42)
        weights *= np.array([balance[c] for c in y])
    else:
        raise ValueError('Unknown additional model')
    model = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
        ('clipper', QuantileClipper()), ('scaler', RobustScaler()), ('classifier', classifier)])
    model.fit(frame[FEATURES], y, classifier__sample_weight=weights)
    return model


def validate_development(train, cal):
    if train.empty or cal.empty or set(train.cik) & set(cal.cik):
        raise ValueError('Empty or overlapping development cohorts')
    for frame, column in [(train, 'feature_filing_date'), (cal, 'source_filing_date')]:
        if frame[TARGET].nunique() != 2 or frame.unit_id.duplicated().any():
            raise ValueError('Invalid development outcomes/units')
        if (pd.to_datetime(frame[column]) >= pd.to_datetime(frame.prediction_date)).any() or (pd.to_datetime(frame.latest_input_filing_date) > pd.to_datetime(frame[column])).any():
            raise ValueError('Future financial input leakage')
    if pd.to_datetime(train.label_available_date).max() >= pd.to_datetime(cal.prediction_date).min():
        raise ValueError('Training outcome time leakage')


def verify_sources():
    for directory in ['sec_disclosure_training_v2', 'sec_disclosure_v2']:
        root = Path('data/processed/validation') / directory
        report = json.loads((root / 'report.json').read_text())
        if report['errors']:
            raise ValueError('Resolve source acquisition errors before model fitting')
        dataset_key = 'dataset_sha256' if directory == 'sec_disclosure_training_v2' else 'combined_final_sha256'
        dataset_name = 'dataset.csv' if directory == 'sec_disclosure_training_v2' else 'combined_final_dataset.csv'
        if digest(root / dataset_name) != report[dataset_key]:
            raise ValueError('Acquired dataset changed')
        for proof in report['checkpoint_proofs']:
            marker = root / 'checkpoints' / (proof['cik'] + '.json')
            if digest(marker) != proof['sha256']:
                raise ValueError('Source checkpoint changed')
            for source in json.loads(marker.read_text())['sources']:
                if digest(source['path']) != source['sha256']:
                    raise ValueError('Source bytes changed')
    from src.validation.sec_disclosure_experiment import verify_acquisition
    verify_acquisition()


def verify_saved():
    result = json.loads((ROOT / 'selection.json').read_text())
    if result['config_sha256'] != digest(CONFIG) or result['implementation_sha256'] != digest(__file__):
        raise ValueError('Frozen expanded selection config/implementation changed')
    for path, expected in result['sealed_files'].items():
        if digest(path) != expected:
            raise ValueError('Frozen expanded model input/artifact changed: ' + path)
    return result


def _run():
    verify_sources()
    admission = admit()
    if (ROOT / 'selection.json').exists():
        return verify_saved()
    if Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json').exists():
        raise ValueError('Final cohort already consumed; cannot tune another candidate')
    ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    # The final table is never loaded by the model-fitting path. Its descriptive
    # support count is a source-admission gate, not a model selection input.
    train, cal = [pd.read_csv(ADMISSION / (r + '.csv'), dtype={'cik': str}) for r in ['training', 'calibration']]
    train, cal = [f.loc[f[TARGET].notna()].copy() for f in [train, cal]]
    validate_development(train, cal)
    folds = cal.cik.map(lambda c: fold_for(c, config['seed'], config['calibration_folds'])).to_numpy()
    y, weight = cal[TARGET].to_numpy(int), cal.design_weight.to_numpy(float)
    fitting = train.copy()
    fitting[OLD_TARGET] = fitting[TARGET]
    candidates, fitted = [], {}
    predictions = cal[['cik', 'unit_id', TARGET, 'design_weight']].copy()
    for name in config['models']:
        print('Fitting expanded SEC model', name, flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            base = train_extra(train, name) if name in {'logistic_c1_balanced', 'hist_gradient_balanced', 'random_forest_balanced'} else fit_base(fitting, FEATURES, name)
        raw = base.predict_proba(cal[FEATURES])[:, 1]
        scores = np.full(len(cal), np.nan)
        for fold in range(config['calibration_folds']):
            held = folds == fold
            if held.any():
                mapper = WeightedSigmoid(base, config['calibration_regularization']).fit_scores(raw[~held], y[~held], weight[~held])
                scores[held] = mapper.predict_scores(raw[held])
        threshold = choose_threshold(y, scores, weight, config['minimum_precision'], config['minimum_recall'])
        measured = metrics(y, scores, weight, threshold)
        candidates.append({'name': name, 'threshold': threshold, 'metrics': measured,
            'converged': not any(issubclass(v.category, ConvergenceWarning) for v in caught),
            'warnings': sorted({str(v.message) for v in caught})})
        fitted[name] = base
        predictions[name] = scores
    valid = [c for c in candidates if c['converged']]
    feasible = [c for c in valid if c['threshold'] is not None]
    if not valid:
        raise ValueError('No converged candidate')
    selected = min(feasible or valid, key=lambda c: (c['metrics']['log_loss'], c['metrics']['brier'], c['name']))
    base = fitted[selected['name']]
    calibrated = WeightedSigmoid(base, config['calibration_regularization']).fit_scores(base.predict_proba(cal[FEATURES])[:, 1], y, weight)
    save_model(calibrated, FEATURES, ROOT / 'candidate.joblib', trained_through=cal.label_available_date.max(), target=TARGET)
    predictions.to_csv(ROOT / 'calibration_predictions.csv', index=False)
    sources = [ROOT / 'candidate.joblib', ROOT / 'calibration_predictions.csv',
        ADMISSION / 'training.csv', ADMISSION / 'calibration.csv', ADMISSION / 'final.csv',
        ADMISSION / 'report.json', ADMISSION / 'evidence.json']
    sources += [Path(p) for p in ['src/validation/sec_disclosure_admission.py', 'src/validation/refine_development.py',
        'src/ml/robust_models.py', 'src/ml/design_calibration.py', 'requirements-local.lock']]
    description = lambda f: {'rows': len(f), 'companies': int(f.cik.nunique()),
        'positive_companies': int(f.loc[f[TARGET].eq(1), 'cik'].nunique())}
    result = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'sealed_files': {str(p): digest(p) for p in sources}, 'selected': selected, 'candidates': candidates,
        'training': description(train), 'calibration': description(cal), 'features': FEATURES,
        'final_support': admission['counts']['final'], 'final_used_for_selection': False,
        'final_predictions_created': False, 'production_ready': False,
        'old_historical_final_retired_into_training': True, 'development_feasible': bool(feasible),
        'calibration_constant_baseline': float(np.average(y, weights=weight)),
        'limits': config['limits']}
    write_json(ROOT / 'selection.json', result)
    positive_support = admission['counts']['final']['positive_companies'] >= config['minimum_final_positive_companies']
    write_json(ROOT / 'evaluation_readiness.json', {'development_precision_recall': bool(feasible),
        'final_positive_support': bool(positive_support), 'required_positive_companies': config['minimum_final_positive_companies'],
        'observed_positive_companies': admission['counts']['final']['positive_companies'],
        'independent_human_label_review': False, 'trading_validation_complete': False,
        'live_aws_validation_complete': False, 'final_test_consumed': False, 'production_ready': False})
    return result


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / '.selection.lock'
    with lock.open('x') as handle:
        handle.write('Expanded model selection in progress')
    try:
        return _run()
    finally:
        lock.unlink()


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
