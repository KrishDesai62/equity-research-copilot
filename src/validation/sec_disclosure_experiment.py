"""Separate, one-use evaluation of the explicitly SEC-disclosed petition target.

Historical actual-bankruptcy development positives require a primary disclosure
cross-check before reuse. Historical negatives retain their BRD coverage scope.
No old final-test companies/labels are reused for training or selection.
"""
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data_layer.research_dataset import load_submissions, write_json
from src.ml.artifacts import load_model, save_model
from src.ml.design_calibration import WeightedSigmoid, metrics, choose_threshold, reliability
from src.validation.adjudicate_public_events import corroborate
from src.validation.cohort_v3 import fold_for, cluster_intervals
from src.validation.cohort_v3_data import digest
from src.validation.download import fetch
from src.validation.local_workflow import local_only
from src.validation.refine_development import development_rows, fit_base
from src.validation.sec_disclosure_availability import resolve_missing_financials
from src.validation.sec_disclosure_data import CONFIG, PLAN, RAW, ROOT, TARGET, FEATURES, PARSER

OLD_TARGET = 'target_bankruptcy_next_1yr'
EXPERIMENT = ROOT / 'experiment'


def historical_training():
    source = Path('data/processed/validation/cohort_v3/dataset.csv')
    seal = json.loads(source.with_name('dataset_manifest.json').read_text())
    if digest(source) != seal['dataset_sha256']:
        raise ValueError('Historical development source integrity failure')
    frame = development_rows(source)
    names = {c['cik']: c for c in json.loads(Path('config/cohort_v3_acquisition.json').read_text())['companies']}
    retained, evidence = [], []
    for cik, group in frame.groupby('cik', sort=True):
        positive = group.loc[group[OLD_TARGET].eq(1)]
        checked = []
        if not positive.empty:
            company = names[cik]
            start = (pd.to_datetime(positive.prediction_date).min() + pd.Timedelta(days=1)).date().isoformat()
            end = (pd.to_datetime(positive.prediction_date).max() + pd.DateOffset(years=1) + pd.Timedelta(days=200)).date().isoformat()
            sources = []
            filings, sec_name = load_submissions(company, start, end, RAW / 'historical_training', sources)
            for filing in filings:
                if filing['filing_type'] != '8-K' or '1.03' not in (filing['items'] or ''):
                    continue
                path = RAW / 'historical_training' / 'documents' / f"{cik}_{filing['accession_number']}.html"
                proof = fetch(filing['source_url'], path)
                sources.append(proof)
                for assertion in corroborate(path.read_text(errors='replace'), filed=filing['filing_date'], names=[sec_name, company['company_name']]):
                    if assertion['issuer_explicitly_included']:
                        checked.append({**assertion, 'disclosed_date': filing['filing_date'], 'source': proof})
            evidence.append({'cik': cik, 'sources': sources, 'assertions': checked})
        for row in group.to_dict('records'):
            label = int(row[OLD_TARGET])
            horizon = (pd.Timestamp(row['prediction_date']) + pd.DateOffset(years=1)).date().isoformat()
            if label and not any(row['prediction_date'] < a['petition_date'] <= horizon for a in checked):
                continue
            row[TARGET] = label
            row['reuse_scope'] = 'primary_confirmed_disclosed_positive' if label else 'historical_brd_covered_nonbankruptcy'
            retained.append(row)
    result = pd.DataFrame(retained)
    if result[TARGET].nunique() != 2:
        raise ValueError('Historical matched-target training lacks both classes')
    return result, evidence


def read_role(role):
    # Filter role before converting any outcome to avoid using final labels
    # during candidate fitting/calibration and threshold selection.
    with (ROOT / 'dataset.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['role'] == role and r[TARGET] != '']
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f'No resolved {role} outcomes')
    for key in FEATURES + [TARGET, 'design_weight']:
        frame[key] = pd.to_numeric(frame[key].replace('', np.nan), errors='raise')
    return frame


def verify_acquisition():
    report = json.loads((ROOT / 'report.json').read_text())
    resolve_missing_financials(report, RAW, ROOT)
    for path, expected in [(CONFIG, report['config_sha256']), (PLAN, report['plan_sha256']),
                            (ROOT / 'dataset.csv', report['dataset_sha256']),
                            (Path('src/validation/sec_disclosure_data.py'), report['implementation_sha256'])]:
        if digest(path) != expected:
            raise ValueError('Acquisition seal changed')
    for proof in report['checkpoint_proofs']:
        marker = ROOT / 'checkpoints' / (proof['cik'] + '.json')
        if digest(marker) != proof['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        record = json.loads(marker.read_text())
        if record['parser_sha256'] != digest(PARSER):
            raise ValueError('Parser changed after label acquisition')
        for source in record['sources']:
            if digest(source['path']) != source['sha256']:
                raise ValueError('Primary source changed')
    return report


def _select():
    verify_acquisition()
    if (EXPERIMENT / 'selection.json').exists():
        return verify_selection()
    # Resume incomplete fits using immutable source checkpoints. A completed
    # selection is verified/reused above, never overwritten or retuned.
    if (ROOT / 'final_evaluation_started.json').exists():
        raise ValueError('Cannot refit after final-test consumption')
    EXPERIMENT.mkdir(exist_ok=True)
    config = json.loads(CONFIG.read_text())
    train, evidence = historical_training()
    cal = read_role('calibration')
    if set(train.cik) & set(cal.cik):
        raise ValueError('Company leakage')
    if pd.to_datetime(train.label_available_date_1yr).max() >= pd.to_datetime(cal.prediction_date).min():
        raise ValueError('Historical training outcomes overlap calibration predictions')
    for frame, source_date in [(train, 'feature_filing_date'), (cal, 'source_filing_date')]:
        if (pd.to_datetime(frame[source_date]) >= pd.to_datetime(frame.prediction_date)).any() or (pd.to_datetime(frame.latest_input_filing_date) > pd.to_datetime(frame[source_date])).any():
            raise ValueError('Feature publication leakage')
    train.to_csv(EXPERIMENT / 'training.csv', index=False)
    write_json(EXPERIMENT / 'training_sources.json', evidence)
    folds = cal.cik.map(lambda c: fold_for(c, config['seed'], 3)).to_numpy()
    y, w = cal[TARGET].to_numpy(), cal.design_weight.to_numpy()
    candidates, fitted = [], {}
    predictions = cal[['cik', 'unit_id', TARGET, 'design_weight']].copy()
    # Existing fitter reads the historical target name; copy only the explicitly
    # harmonized labels, never the old study's held-out rows.
    fitting = train.copy()
    fitting[OLD_TARGET] = fitting[TARGET]
    for name in config['models']:
        print('Fitting SEC disclosure candidate', name, flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            base = fit_base(fitting, FEATURES, name)
        raw = base.predict_proba(cal[FEATURES])[:, 1]
        oof = np.full(len(cal), np.nan)
        for fold in range(3):
            held = folds == fold
            if held.any():
                if len(np.unique(y[~held])) != 2:
                    raise ValueError('Calibration fold lacks both outcomes')
                mapper = WeightedSigmoid(base, .1).fit_scores(raw[~held], y[~held], w[~held])
                oof[held] = mapper.predict_scores(raw[held])
        threshold = choose_threshold(y, oof, w, config['minimum_precision'], config['minimum_recall'])
        candidates.append({'name': name, 'metrics': metrics(y, oof, w, threshold), 'threshold': threshold,
            'converged': not any(issubclass(v.category, ConvergenceWarning) for v in caught),
            'warnings': sorted({str(v.message) for v in caught})})
        fitted[name] = base
        predictions[name] = oof
    valid = [c for c in candidates if c['converged']]
    if not valid:
        raise ValueError('No converged candidate')
    chosen = min(valid, key=lambda c: (c['metrics']['log_loss'], c['metrics']['brier'], c['name']))
    base = fitted[chosen['name']]
    wrapper = WeightedSigmoid(base, .1).fit_scores(base.predict_proba(cal[FEATURES])[:, 1], y, w)
    save_model(wrapper, FEATURES, EXPERIMENT / 'candidate.joblib', trained_through=cal.label_available_date.max(), target=TARGET)
    predictions.to_csv(EXPERIMENT / 'calibration_predictions.csv', index=False)
    selection = {'target': TARGET, 'dataset_sha256': digest(ROOT / 'dataset.csv'), 'config_sha256': digest(CONFIG),
        'plan_sha256': digest(PLAN), 'artifact_sha256': digest(EXPERIMENT / 'candidate.joblib'),
        'training_sha256': digest(EXPERIMENT / 'training.csv'), 'training_sources_sha256': digest(EXPERIMENT / 'training_sources.json'),
        'calibration_predictions_sha256': digest(EXPERIMENT / 'calibration_predictions.csv'),
        'implementation_sha256': digest(__file__), 'parser_sha256': digest(PARSER),
        'availability_resolution_sha256': digest(ROOT / 'availability_resolution.json'),
        'dependencies': {p: digest(p) for p in ['src/validation/refine_development.py',
            'src/ml/design_calibration.py', 'src/ml/robust_models.py',
            'src/ml/financial_history.py', 'src/validation/sec_disclosure_availability.py', 'requirements-local.lock']},
        'candidates': candidates, 'selected': chosen, 'constant_baseline': float(np.average(y, weights=w)),
        'training_rows': len(train), 'training_positive_companies': int(train.loc[train[TARGET].eq(1), 'cik'].nunique()),
        'calibration_rows': len(cal), 'calibration_positive_companies': int(cal.loc[cal[TARGET].eq(1), 'cik'].nunique()),
        'training_companies': sorted(set(train.cik)), 'calibration_companies': sorted(set(cal.cik)),
        'final_used_for_selection': False, 'production_ready': False}
    write_json(EXPERIMENT / 'selection.json', selection)
    return selection


def select():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / '.selection.lock'
    with lock.open('x') as handle:
        handle.write('SEC disclosure model selection in progress')
    try:
        return _select()
    finally:
        lock.unlink()


def verify_selection():
    saved = json.loads((EXPERIMENT / 'selection.json').read_text())
    for path, key in [(ROOT / 'dataset.csv', 'dataset_sha256'), (CONFIG, 'config_sha256'), (PLAN, 'plan_sha256'),
        (EXPERIMENT / 'candidate.joblib', 'artifact_sha256'), (EXPERIMENT / 'training.csv', 'training_sha256'),
        (EXPERIMENT / 'training_sources.json', 'training_sources_sha256'),
        (EXPERIMENT / 'calibration_predictions.csv', 'calibration_predictions_sha256'), (Path(__file__), 'implementation_sha256'), (PARSER, 'parser_sha256'), (ROOT / 'availability_resolution.json', 'availability_resolution_sha256')]:
        if digest(path) != saved[key]:
            raise ValueError('Frozen experiment integrity failure: ' + key)
    for path, expected in saved['dependencies'].items():
        if digest(path) != expected:
            raise ValueError('Experiment dependency changed: ' + path)
    for entry in json.loads((EXPERIMENT / 'training_sources.json').read_text()):
        for source in entry['sources']:
            if digest(source['path']) != source['sha256']:
                raise ValueError('Historical disclosure source changed')
    return saved


def evaluate_once():
    verify_acquisition()
    frozen = verify_selection()
    final_path = EXPERIMENT / 'final_evaluation.json'
    if final_path.exists():
        result = json.loads(final_path.read_text())
        if result['selection_sha256'] != digest(EXPERIMENT / 'selection.json') or result['predictions_sha256'] != digest(EXPERIMENT / 'final_predictions.csv'):
            raise ValueError('Saved final evaluation integrity failure')
        return result
    config = json.loads(CONFIG.read_text())
    test = read_role('final')
    if set(test.cik) & set(frozen['training_companies'] + frozen['calibration_companies']):
        raise ValueError('Final company leakage')
    minimums = {'final_companies': int(test.cik.nunique()),
        'final_positive_companies': int(test.loc[test[TARGET].eq(1), 'cik'].nunique())}
    if minimums['final_companies'] < config['minimum_final_companies'] or minimums['final_positive_companies'] < config['minimum_final_positive_companies']:
        result = {**minimums, 'production_ready': False, 'final_test_consumed': False,
            'status': 'insufficient_final_support_no_predictions_created',
            'required_final_companies': config['minimum_final_companies'],
            'required_positive_companies': config['minimum_final_positive_companies'],
            'selection_sha256': digest(EXPERIMENT / 'selection.json')}
        write_json(EXPERIMENT / 'evaluation_blocked.json', result)
        return result
    artifact = load_model(EXPERIMENT / 'candidate.joblib', expected_sha256=frozen['artifact_sha256'])
    if artifact['target'] != TARGET or artifact['feature_cols'] != FEATURES or artifact['trained_through'] >= test.prediction_date.min():
        raise ValueError('Final artifact schema or temporal leakage')
    # Marker is shared by all artifacts/output paths in this frozen cohort.
    with (ROOT / 'final_evaluation_started.json').open('x') as handle:
        json.dump({'selection_sha256': digest(EXPERIMENT / 'selection.json'), 'final_consumed': True}, handle)
    p = artifact['model'].predict_proba(test[FEATURES])[:, 1]
    threshold, baseline = frozen['selected']['threshold'], frozen['constant_baseline']
    weighted = metrics(test[TARGET], p, test.design_weight, threshold)
    constant = metrics(test[TARGET], np.full(len(test), baseline), test.design_weight)
    bootstrap = test.copy()
    bootstrap[OLD_TARGET] = bootstrap[TARGET]
    intervals = cluster_intervals(bootstrap, p, threshold, baseline)
    gates = {'minimum_final_companies': test.cik.nunique() >= config['minimum_final_companies'],
        'minimum_final_positive_companies': test.loc[test[TARGET].eq(1), 'cik'].nunique() >= config['minimum_final_positive_companies'],
        'minimum_calibration_positive_companies': frozen['calibration_positive_companies'] >= config['minimum_calibration_positive_companies'],
        'development_precision_recall': threshold is not None,
        'final_precision_recall': threshold is not None and weighted.get('precision', 0) >= config['minimum_precision'] and weighted.get('recall', 0) >= config['minimum_recall'],
        'brier_improvement_lower_bound_positive': intervals['brier_improvement_over_constant']['lower'] > 0,
        'representative_population_coverage_certified': False, 'independent_label_audit_complete': False,
        'trading_agents_separately_validated': False, 'live_operations_validated': False}
    out = test[['cik', 'unit_id', TARGET, 'design_weight']].copy()
    out['probability'] = p
    out.to_csv(EXPERIMENT / 'final_predictions.csv', index=False)
    result = {'target': TARGET, 'weighted': weighted, 'unweighted': metrics(test[TARGET], p, np.ones(len(test)), threshold),
        'constant_baseline': constant, 'company_bootstrap_95pct': intervals,
        'calibration_bins': reliability(test[TARGET], p, test.design_weight), 'gates': {k: bool(v) for k, v in gates.items()},
        'production_ready': bool(all(gates.values())), 'final_companies': int(test.cik.nunique()),
        'final_positive_companies': int(test.loc[test[TARGET].eq(1), 'cik'].nunique()),
        'selection_sha256': digest(EXPERIMENT / 'selection.json'), 'predictions_sha256': digest(EXPERIMENT / 'final_predictions.csv'),
        'warning': 'One-use, proxy-assisted and follow-up-eligible SEC disclosure evaluation; not exhaustive actual bankruptcy or a profitable trading claim. Test is consumed and cannot guide further tuning.'}
    write_json(final_path, result)
    return result


if __name__ == '__main__':
    with local_only():
        select()
        print(json.dumps(evaluate_once(), indent=2))
