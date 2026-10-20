"""Reserve a new company/time experiment before receiving reviewed labels."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import PLAN
from src.validation.reviewed_outcomes import load_reviews


def exposed_companies(*, ignore=None):
    excluded = set()
    for path in Path('config').rglob('*.json'):
        if ignore is not None and path.resolve() == Path(ignore).resolve():
            continue
        value = json.loads(path.read_text())
        if isinstance(value, dict) and isinstance(value.get('companies'), list):
            excluded |= {str(c['cik']).zfill(10) for c in value['companies'] if isinstance(c, dict) and c.get('cik')}
    for path in [Path('data/processed/validation/expanded/features.csv'),
                 Path('data/processed/validation/reserved_20261017/development_features.csv')]:
        if path.exists():
            excluded |= set(pd.read_csv(path, usecols=['cik'], dtype={'cik': str}).cik)
    return excluded


def reserve(companies, output, *, seed, final_start, training_cases=()):
    path = Path(output)
    # Include every V3 assigned company, not only companies with usable features.
    excluded = exposed_companies()
    if PLAN.exists():
        excluded |= {c['cik'] for c in json.loads(PLAN.read_text())['companies']}
    ciks = [str(c).zfill(10) for c in companies]
    if not ciks or len(set(ciks)) != len(ciks) or any(not c.isdigit() or len(c) != 10 or int(c) <= 0 for c in ciks):
        raise ValueError('Reserve unique valid company CIKs')
    if set(ciks) & excluded:
        raise ValueError('Reservation includes previously exposed companies')
    known = [str(c).zfill(10) for c in training_cases]
    if len(set(known)) != len(known) or set(known) & set(ciks) or any(not c.isdigit() or len(c) != 10 or int(c) <= 0 for c in known):
        raise ValueError('Known-case training additions must be unique valid CIKs outside the blind sample')
    if pd.Timestamp(final_start) <= pd.Timestamp('2021-12-31'):
        raise ValueError('Final predictions must be later than the V3 final periods')
    records = []
    for cik in sorted(ciks):
        number = int(sha256(f'{seed}:role:{cik}'.encode()).hexdigest(), 16) % 3
        records.append({'cik': cik, 'role': ['training', 'calibration', 'final'][number]})
    records.extend({'cik': cik, 'role': 'training', 'selection': 'known-case development only'} for cik in sorted(known))
    plan = {'version': 1, 'seed': seed, 'final_start': final_start, 'companies': records,
            'labels_present_at_reservation': bool(known), 'known_case_training_companies': known,
            'final_labels_present_at_reservation': False, 'sampling_design_reviewed': False,
            'note': 'Company reservation alone does not establish population sampling or independent source review.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plan, handle, indent=2)
    return plan


def admit_reviews(reservation, reviews, *, as_of, output, frozen_candidate=None):
    plan_path, review_path = Path(reservation), Path(reviews)
    plan = json.loads(plan_path.read_text())
    frame = load_reviews(review_path, as_of=as_of)
    roles = {c['cik']: c['role'] for c in plan['companies']}
    if not set(frame.cik) <= set(roles):
        raise ValueError('Outcomes include companies outside the frozen reservation')
    frame['role'] = frame.cik.map(roles)
    if frame.role.eq('final').any():
        if frozen_candidate is None:
            raise ValueError('Freeze a candidate before admitting quarantined final outcomes')
        frozen = json.loads(Path(frozen_candidate).read_text())
        if frozen['reservation_sha256'] != sha256(plan_path.read_bytes()).hexdigest():
            raise ValueError('Candidate does not match this final reservation')
    start = pd.Timestamp(plan['final_start'])
    for role in ['training', 'calibration', 'final']:
        part = frame.loc[frame.role.eq(role)]
        if role == 'final' and (pd.to_datetime(part.prediction_date) < start).any():
            raise ValueError('Final outcome predates reserved final window')
        if role != 'final' and (pd.to_datetime(part.label_available_date_1yr) >= start).any():
            raise ValueError('Development outcome was unavailable before the final period')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    # Final labels remain quarantined for a subsequent frozen evaluation protocol.
    for role in ['training', 'calibration', 'final']:
        frame.loc[frame.role.eq(role)].to_csv(out / f'{role}_reviewed_outcomes.csv', index=False)
    report = {'reservation_sha256': sha256(plan_path.read_bytes()).hexdigest(),
              'reviews_sha256': sha256(review_path.read_bytes()).hexdigest(),
              'reviewed_rows': len(frame), 'production_ready': False,
              'next': 'Reconcile point-in-time financial features and sampling design; seal final labels before fitting.'}
    write_json(out / 'admission.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('reserve')
    create.add_argument('--companies', type=Path, required=True, help='JSON list of CIKs selected from a documented outcome-blind sampling frame')
    create.add_argument('--output', type=Path, required=True, help='Keep reservations under config/reservations/')
    create.add_argument('--seed', required=True)
    create.add_argument('--final-start', required=True)
    create.add_argument('--training-cases', type=Path, help='Optional known-case CIK list, forced to training and never calibration/final')
    admit = commands.add_parser('admit')
    for name in ['reservation', 'reviews', 'as-of', 'output']:
        admit.add_argument('--' + name, required=True)
    admit.add_argument('--frozen-candidate')
    args = parser.parse_args()
    if args.action == 'reserve':
        result = reserve(json.loads(args.companies.read_text()), args.output, seed=args.seed, final_start=args.final_start,
                         training_cases=json.loads(args.training_cases.read_text()) if args.training_cases else ())
    else:
        result = admit_reviews(args.reservation, args.reviews, as_of=args.as_of, output=args.output, frozen_candidate=args.frozen_candidate)
    print(json.dumps(result, indent=2))
