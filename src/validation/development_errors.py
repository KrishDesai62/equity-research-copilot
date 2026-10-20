"""Explain saved out-of-company-fold errors without scoring the reserved final."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data_layer.research_dataset import write_json
from src.ml.design_calibration import metrics
from src.ml.filing_signal_models import FEATURES
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET
from src.validation.sec_disclosure_models_v4 import verify_saved

CONFIG = Path('config/development_20261019/analysis.json')
ROOT = Path('data/processed/validation/development_errors_20261019')


def summarize(frame, score, threshold):
    return metrics(frame[TARGET], frame[score], frame.design_weight, threshold)


def operating_limits(frame, score):
    p, r, _ = precision_recall_curve(frame[TARGET], frame[score], sample_weight=frame.design_weight)
    return {'maximum_recall_at_precision_50pct': float(r[p >= .5].max()) if (p >= .5).any() else 0.,
        'maximum_precision_at_recall_50pct': float(p[r >= .5].max()) if (r >= .5).any() else 0.}


def run():
    selection = verify_saved()
    config = json.loads(CONFIG.read_text())
    sources = {'training': Path('data/processed/validation/sec_filing_signals_v4/training.csv'),
        'calibration': Path('data/processed/validation/sec_filing_signals_v4/calibration.csv'),
        'predictions': Path('data/processed/validation/sec_disclosure_models_v4/calibration_predictions.csv'),
        'annual_frame': Path('data/processed/validation/sec_disclosure_v1/annual_frame.csv')}
    source_hashes = {k: digest(v) for k, v in sources.items()}
    ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'report.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved['source_hashes'] != source_hashes or saved['config_sha256'] != digest(CONFIG) or saved['implementation_sha256'] != digest(__file__):
            raise ValueError('Frozen error-analysis inputs changed')
        for name, expected in saved['outputs'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen diagnostic output changed')
        return saved
    train, cal = [pd.read_csv(sources[r], dtype={'cik': str}) for r in ['training', 'calibration']]
    train, cal = [f.loc[f[TARGET].notna()].copy() for f in [train, cal]]
    predictions = pd.read_csv(sources['predictions'], dtype={'cik': str})
    frame = cal.merge(predictions, on=['cik', 'unit_id', TARGET, 'design_weight'], validate='one_to_one')
    if len(frame) != len(cal):
        raise ValueError('Prediction/label join lost development companies')
    score = selection['selected']['name']
    annual = pd.read_csv(sources['annual_frame'], dtype={'cik': str})
    annual = annual.loc[annual.frame_year.eq(2022) & annual.filed.lt('2022-12-31')].sort_values('filed').drop_duplicates('cik', keep='last')
    frame['sic'] = frame.cik.map(annual.set_index('cik').sic)
    frame['industry_division'] = pd.to_numeric(frame.sic, errors='coerce').floordiv(1000).fillna(-1).astype(int)
    for f in [train, frame]:
        f['size_band'] = pd.cut(f.Assets, bins=[-np.inf] + config['size_edges_usd'] + [np.inf],
            labels=['under_100m', '100m_to_400m', '400m_to_2b', 'over_2b']).astype(str)
    drift = []
    for feature in FEATURES:
        baseline = pd.to_numeric(train[feature], errors='coerce')
        recent = pd.to_numeric(frame[feature], errors='coerce')
        iqr = baseline.quantile(.75) - baseline.quantile(.25)
        drift.append({'feature': feature, 'train_median': baseline.median(), 'recent_median': recent.median(),
            'median_shift_train_iqr': (recent.median() - baseline.median()) / iqr if iqr > 0 else None,
            'train_missing_fraction': float(baseline.isna().mean()), 'recent_missing_fraction': float(recent.isna().mean())})
    pd.DataFrame(drift).to_csv(ROOT / 'feature_shift.csv', index=False)
    columns = ['unit_id', 'cik', 'company_name', TARGET, 'design_weight', score, 'Assets', 'size_band', 'industry_division']
    frame.loc[frame[TARGET].eq(1), columns].sort_values(score).to_csv(ROOT / 'missed_positive_candidates.csv', index=False)
    frame.loc[frame[TARGET].eq(0), columns].sort_values(score, ascending=False).to_csv(ROOT / 'false_alert_candidates.csv', index=False)
    segments = []
    for field in ['size_band', 'industry_division']:
        for group, part in frame.groupby(field):
            segments.append({'grouping': field, 'group': str(group), **summarize(part, score, .05)})
    write_json(ROOT / 'segments.json', segments)
    mix = {role: {str(group): {'rows': len(part), 'positive_companies': int(part.loc[part[TARGET].eq(1), 'cik'].nunique())}
        for group, part in f.groupby('size_band')} for role, f in [('training', train), ('calibration', frame)]}
    result = {'source_hashes': source_hashes, 'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'selected_reference': score, 'operating_limits': operating_limits(frame, score), 'company_size_mix': mix,
        'diagnostic_thresholds': {str(t): summarize(frame, score, t) for t in config['diagnostic_thresholds']},
        'model_operating_limits': {c['name']: operating_limits(frame, c['name']) for c in selection['candidates']},
        'outputs': {n: digest(ROOT / n) for n in ['feature_shift.csv', 'missed_positive_candidates.csv', 'false_alert_candidates.csv', 'segments.json']},
        'final_rows_loaded': False, 'new_model_fitted': False,
        'limits': 'Small weighted development subgroups and repeated selection are descriptive, not independent inference or release thresholds.'}
    write_json(report_path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
