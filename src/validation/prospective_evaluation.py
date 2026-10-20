"""Freeze a candidate, then evaluate reviewed recent outcomes once on reserved firms."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model
from src.ml.design_calibration import metrics, reliability, validate_weights
from src.validation.reviewed_outcomes import load_reviews
from src.validation.cohort_v3_data import TARGET


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze(reservation, artifact, output, *, threshold=None):
    if Path(reservation).with_suffix('.evaluation_started.json').exists():
        raise ValueError('Reserved final cohort is already consumed; do not freeze another candidate for it')
    plan = json.loads(Path(reservation).read_text())
    checksum = digest(artifact)
    model = load_model(artifact, expected_sha256=checksum)
    if model['target'] != TARGET or pd.Timestamp(model['trained_through']) >= pd.Timestamp(plan['final_start']):
        raise ValueError('Candidate target/cutoff does not match future evaluation')
    if threshold is not None and (not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Invalid frozen operating threshold')
    from src.validation.future_cohort import exposed_companies
    if {c['cik'] for c in plan['companies'] if c['role'] != 'training'} & exposed_companies(ignore=reservation):
        raise ValueError('Candidate reservation contains exposed companies')
    frozen = {'reservation': str(reservation), 'reservation_sha256': digest(reservation),
              'artifact': str(artifact), 'artifact_sha256': checksum, 'threshold': threshold,
              'features': model['feature_cols'], 'created_at': datetime.now(timezone.utc).isoformat(),
              'final_labels_used_for_selection': False, 'production_ready': False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(frozen, handle, indent=2)
    return frozen


def evaluate(frozen_path, feature_path, review_path, output, *, as_of):
    frozen = json.loads(Path(frozen_path).read_text())
    if digest(frozen['reservation']) != frozen['reservation_sha256'] or digest(frozen['artifact']) != frozen['artifact_sha256']:
        raise ValueError('Reserved protocol or frozen artifact changed')
    plan = json.loads(Path(frozen['reservation']).read_text())
    expected = {c['cik'] for c in plan['companies'] if c['role'] == 'final'}
    features = pd.read_csv(feature_path, dtype={'cik': str})
    reviews = load_reviews(review_path, as_of=as_of)
    if not expected or set(features.cik) != expected or set(reviews.cik) != expected:
        raise ValueError('Final sources must cover every reserved final company, without replacements')
    if (features.prediction_date.ne(plan['final_start']).any() or reviews.prediction_date.ne(plan['final_start']).any()):
        raise ValueError('Prediction dates differ from the reserved final anchor')
    if TARGET in features.columns:
        raise ValueError('Final feature input must not contain labels')
    data = features.merge(reviews, on=['cik', 'prediction_date'], validate='one_to_one')
    if len(data) != len(expected) or data[frozen['features']].isna().all(axis=0).any():
        raise ValueError('Incomplete final feature coverage')
    prediction = pd.to_datetime(data.prediction_date)
    public = pd.to_datetime(data.feature_filing_date)
    latest = pd.to_datetime(data.latest_input_filing_date)
    if public.isna().any() or prediction.isna().any() or latest.isna().any() or (public >= prediction).any() or (latest > public).any() or ((prediction - public).dt.days > 180).any():
        raise ValueError('Future or stale financial inputs')
    if data[frozen['features']].isna().mean(axis=1).gt(.25).any():
        raise ValueError('Final financial rows exceed the missing-input limit')
    if 'change_available_date' in data:
        dates = pd.to_datetime(data.change_available_date)
        if (dates.notna() & dates.ge(prediction)).any():
            raise ValueError('Future disclosure change features')
    model = load_model(frozen['artifact'], expected_sha256=frozen['artifact_sha256'])
    if model['feature_cols'] != frozen['features']:
        raise ValueError('Model feature schema changed')
    if list(model['model'].classes_) != [0, 1]:
        raise ValueError('Final model must predict binary classes 0/1 in order')
    weights = data.design_weight if 'design_weight' in data else np.ones(len(data))
    validate_weights(data[TARGET], weights)
    out = Path(output)
    if out.exists():
        raise FileExistsError('Final output already exists')
    consumed = Path(frozen['reservation']).with_suffix('.evaluation_started.json')
    with consumed.open('x') as handle:
        json.dump({'frozen_sha256': digest(frozen_path), 'features_sha256': digest(feature_path),
                   'reviews_sha256': digest(review_path), 'consumed': True}, handle)
    out.mkdir(parents=True, exist_ok=False)  # Exclusive consumption before prediction.
    write_json(out / 'evaluation_started.json', {'frozen_sha256': digest(frozen_path),
        'features_sha256': digest(feature_path), 'reviews_sha256': digest(review_path), 'consumed': True})
    p = model['model'].predict_proba(data[frozen['features']])[:, 1]
    measured = metrics(data[TARGET], p, weights, frozen['threshold'])
    result = {'companies': len(data), 'positive_companies': int(data[TARGET].sum()),
        'metrics': measured,
        'gates': {'minimum_final_companies': len(data) >= 50, 'minimum_positive_companies': int(data[TARGET].sum()) >= 20,
                  'precision_floor': measured.get('precision', 0) >= .5, 'recall_floor': measured.get('recall', 0) >= .5,
                  'population_sampling_reviewed': False, 'probability_calibration_validated': False},
        'calibration_bins': reliability(data[TARGET], p, weights),
        'representative_population': False, 'validated_calibration': False, 'production_ready': False,
        'note': 'Reviewed outcome/timing checks do not certify population sampling or probability calibration. No automatic production promotion.'}
    data['predicted_probability'] = p
    data.to_csv(out / 'predictions.csv', index=False)
    write_json(out / 'result.json', result)
    write_json(out / 'receipt.json', {'result_sha256': digest(out / 'result.json'), 'predictions_sha256': digest(out / 'predictions.csv')})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('freeze')
    for name in ['reservation', 'artifact', 'output']:
        create.add_argument('--' + name, required=True)
    create.add_argument('--threshold', type=float)
    score = commands.add_parser('evaluate-once')
    for name in ['frozen', 'features', 'reviews', 'output', 'as-of']:
        score.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze(args.reservation, args.artifact, args.output, threshold=args.threshold)
    else:
        result = evaluate(args.frozen, args.features, args.reviews, args.output, as_of=args.as_of)
    print(json.dumps(result, indent=2))
