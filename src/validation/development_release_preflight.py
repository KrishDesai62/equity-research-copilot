"""Audit development support and release gates without opening final-test rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET

ROOT = Path('data/processed/validation/development_preflight_20261019')


def weight_support(frame):
    """Kish support and concentration at company level, separately by outcome."""
    if frame.unit_id.duplicated().any():
        raise ValueError('Duplicate development units')
    result = {}
    for label, name in [(0, 'negative'), (1, 'positive')]:
        part = frame.loc[frame[TARGET].eq(label)]
        # This recent calibration cohort must contain one prediction per company.
        if part.cik.duplicated().any():
            raise ValueError('Expected one recent prediction per company')
        weights = part.design_weight.to_numpy(float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Invalid design weights')
        total = float(weights.sum())
        result[name] = {
            'companies': len(part), 'weight_sum': total,
            'effective_sample_size': float(total * total / (weights @ weights)) if len(weights) else 0.,
            'largest_company_weight_fraction': float(weights.max() / total) if len(weights) else None,
        }
    return result


def release_blockers(selection, review):
    """No scoring, artifact promotion or held-out label access occurs here."""
    reasons = []
    if not selection.get('development_feasible', False):
        reasons.append('No development threshold satisfies both precision and recall requirements')
    if not review.get('independent_reviews_complete', False):
        reasons.append('Independent outcome review is incomplete')
    # This preflight cannot grant release: support, a one-use evaluation and
    # operational validation must be checked by their own release workflow.
    return reasons


def run():
    calibration = Path('data/processed/validation/issuer_identity_admission_v2_20261019/calibration.csv')
    if not calibration.exists():
        calibration = Path('data/processed/validation/sec_filing_signals_v4/calibration.csv')
    frame = pd.read_csv(calibration, dtype={'cik': str})
    frame = frame.loc[frame[TARGET].notna()].copy()
    result = {'calibration_sha256': digest(calibration), 'support': weight_support(frame),
              'final_rows_loaded': False, 'final_predictions_created': False,
              'production_approval': False,
              'interpretation': 'Weighted effective sample size describes concentration, not the number of independently verified outcomes. Do not remove a valid high-weight company or change its weight to improve metrics.'}
    review_path = Path('data/processed/validation/disclosure_review_status_20261019/status.json')
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    selection_path = Path('data/processed/validation/matched_models_20261019/frozen_selection.json')
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    result['release_blockers'] = release_blockers(selection, review)
    if not selection_path.exists():
        result['release_blockers'].insert(0, 'Matched development experiment and selection are not complete')
    result['release_blockers'].append('Reserved final outcomes and review packet need the updated issuer-identity audit')
    result['release_blockers'].append('A one-use final evaluation and operational release validation have not been completed')
    result['selection_sha256'] = digest(selection_path) if selection_path.exists() else None
    result['review_sha256'] = digest(review_path) if review_path.exists() else None
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'support.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
