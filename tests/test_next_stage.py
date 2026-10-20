from datetime import datetime, timezone
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.validation.reviewed_outcomes import verify_review
from src.validation.future_cohort import reserve
from src.ml.disclosure_features import add_features, report_features
from src.backtest.backtest_engine import evaluate as backtest
from src.operations.status import assess_health, handler
from src.operations.model_registry import ModelRegistry


def review(tmp_path, *, cik='0009999991', label=0):
    source = tmp_path / 'evidence.txt'
    source.write_text('Synthetic unit-test evidence, never a real reviewed outcome')
    record = {'cik': cik, 'prediction_date': '2024-12-31', 'target': label,
        'reviewers': ['fixture-reviewer-a', 'fixture-reviewer-b'], 'reviewed_on': '2026-01-02',
        'identity_verified': True, 'prior_bankruptcy_excluded': True,
        'evidence': [{'kind': 'bankruptcy_event' if label else 'complete_outcome_coverage',
            'url': 'https://example.org/unit-test-only', 'path': str(source),
            'sha256': sha256(source.read_bytes()).hexdigest(), 'retrieved_on': '2026-01-01'}]}
    if label:
        record.update(chapter='11', event_date='2025-06-01')
    else:
        record.update(coverage_through='2025-12-31', complete_event_coverage=True,
                      coverage_method='Synthetic full-year unit-test attestation')
    return record


def test_review_requires_full_followup_two_reviewers_and_unchanged_evidence(tmp_path):
    record = review(tmp_path)
    result = verify_review(record, as_of='2026-10-17')
    assert result['target_bankruptcy_next_1yr'] == 0
    record['coverage_through'] = '2025-10-01'
    with pytest.raises(ValueError, match='full-year'):
        verify_review(record, as_of='2026-10-17')
    record = review(tmp_path)
    record['reviewers'] = ['same', 'same']
    with pytest.raises(ValueError, match='distinct'):
        verify_review(record, as_of='2026-10-17')
    record = review(tmp_path)
    (tmp_path / 'evidence.txt').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_review(record, as_of='2026-10-17')


def test_positive_is_observed_at_event_without_full_postcase_year(tmp_path):
    record = review(tmp_path, label=1)
    assert verify_review(record, as_of='2026-10-17')['label_available_date_1yr'] == '2025-06-01'
    record['event_date'] = '2026-06-01'
    with pytest.raises(ValueError, match='inside the horizon'):
        verify_review(record, as_of='2026-10-17')


def test_future_reservation_rejects_old_firms_and_is_exclusive(tmp_path, monkeypatch):
    import src.validation.future_cohort as module
    monkeypatch.setattr(module, 'exposed_companies', lambda: {'0000000001'})
    monkeypatch.setattr(module, 'PLAN', tmp_path / 'missing')
    with pytest.raises(ValueError, match='previously exposed'):
        reserve(['1'], tmp_path / 'reservation.json', seed='precommitted', final_start='2024-12-31')
    result = reserve(['9999991', '9999992'], tmp_path / 'reservation.json', seed='precommitted', final_start='2024-12-31')
    assert not result['labels_present_at_reservation']
    with pytest.raises(FileExistsError):
        reserve(['9999991', '9999992'], tmp_path / 'reservation.json', seed='precommitted', final_start='2024-12-31')


def report():
    previous = {'cik': '0009999991', 'filing_type': '10-K', 'filing_date': '2023-11-01'}
    current = {**previous, 'filing_date': '2024-11-01', 'source_url': 'https://example.org/filing', 'accession_number': 'new'}
    return {'cik': current['cik'], 'previous': previous, 'current': current,
        'sections': [{'section': 'risk_factors', 'status': 'changed', 'text_similarity': .8,
                      'changes': [{'previous_text': 'Liquidity', 'current_text': 'Going concern and default'}]},
                     {'section': 'management_discussion', 'status': 'unchanged', 'text_similarity': 1., 'changes': []}]}


def test_disclosure_features_preserve_missingness_and_exclude_same_day():
    r = report()
    frame = pd.DataFrame([{'cik': '0009999991', 'filing_date': date, 'Assets': 100., 'Cash': 10.,
        'Liabilities': 80., 'NetIncomeLoss_ttm': 10., 'OperatingCashFlow_ttm': -5.,
        'latest_input_filing_date': date} for date in ['2024-11-01', '2024-11-02']])
    result = add_features(frame, [r])
    assert np.isnan(result.iloc[0].risk_text_change)
    assert result.iloc[1].risk_text_change == pytest.approx(.2)
    assert result.iloc[1].mda_text_change == 0
    assert result.iloc[1].risk_added_distress_mentions == 2
    assert result.iloc[1].accruals_to_assets == .15
    assert result.iloc[1].operating_cash_burn_to_cash == .5
    r['sections'][1] = {'section': 'management_discussion', 'status': 'unavailable'}
    assert np.isnan(report_features(r)['mda_text_change'])


def panel():
    rows = []
    for period, decision, entry, end in [(0, '2024-01-01', '2024-01-02', '2024-03-31'), (1, '2024-04-01', '2024-04-02', '2024-06-30')]:
        for ticker, returns in [('A', [.1, .2]), ('B', [-.1, -1.])]:
            rows.append({'ticker': ticker, 'decision_at': decision, 'available_at': decision,
                'model_trained_through': '2023-01-01', 'entry_at': entry, 'exit_at': end,
                'agent_action': ('buy' if period == 0 else 'hold') if ticker == 'A' else 'sell',
                'bankruptcy_probability': .05 if ticker == 'A' else .3,
                'news_sentiment': .5 if ticker == 'A' else -.5,
                'evidence_url': 'https://example.org/dated-evidence', 'total_return': returns[period]})
    return pd.DataFrame(rows)


def test_strategy_evaluation_matches_hand_arithmetic_and_includes_delisting():
    no_cost = backtest(panel(), cost_bps=0)
    assert no_cost['strategies']['agent']['total_return'] == pytest.approx(.155)
    assert no_cost['strategies']['buy_hold']['total_return'] == pytest.approx(-.34)
    assert backtest(panel(), cost_bps=20)['strategies']['agent']['total_return'] < .155
    assert not no_cost['validated_agent_profitability']


@pytest.mark.parametrize('change', ['future_news', 'future_model', 'same_day_execution', 'missing_failed_firm', 'invalid_return'])
def test_strategy_rejects_lookahead_and_incomplete_panel(change):
    frame = panel()
    if change == 'future_news': frame.loc[0, 'available_at'] = '2024-02-01'
    if change == 'future_model': frame.loc[0, 'model_trained_through'] = '2025-01-01'
    if change == 'same_day_execution': frame.loc[0, 'entry_at'] = '2024-01-01'
    if change == 'missing_failed_firm': frame = frame.iloc[:-1]
    if change == 'invalid_return': frame.loc[0, 'total_return'] = -1.1
    with pytest.raises(ValueError): backtest(frame)


def test_operator_health_rejects_stale_future_and_unauthenticated_requests():
    now = datetime(2026, 10, 17, 18, tzinfo=timezone.utc)
    assert not assess_health(None, now=now)['healthy']
    assert assess_health({'state': 'success', 'completed_at': '2026-10-17T17:00:00+00:00'}, now=now)['healthy']
    for timestamp in ['2026-10-14T00:00:00+00:00', '2026-10-18T00:00:00+00:00', '2026-10-17T17:00:00']:
        assert not assess_health({'state': 'success', 'completed_at': timestamp}, now=now)['healthy']
    assert handler({'rawPath': '/status'}, None)['statusCode'] == 401


def approval(artifact):
    from src.ml.release import assess
    policy = json.loads(__import__('pathlib').Path('config/release_policy.json').read_text())
    evidence = {'test_metrics': {'precision': .8, 'recall': .8}, 'test_companies': 50,
                'independent_positive_events': 20, 'independent_test': True, 'representative_population': True,
                'validated_calibration': True, 'preprocessing_fit_on': 'training rows only'}
    return {'status': 'approved', 'calibrated': True, 'released_on': '2026-10-17', 'threshold': .5,
            'artifact_sha256': sha256(artifact.read_bytes()).hexdigest(), 'evidence': evidence, 'assessment': assess(evidence, policy)}


def test_model_registry_rejects_research_and_rolls_back_only_valid_versions(tmp_path):
    from src.ml.artifacts import save_model
    registry = ModelRegistry(tmp_path / 'registry')
    artifact, release = tmp_path / 'model', tmp_path / 'release.json'
    model = DummyClassifier(strategy='prior').fit(pd.DataFrame({'x': [0., 1.]}), [0, 1])
    save_model(model, ['x'], artifact, trained_through='2022-12-31', target='target_bankruptcy_next_1yr')
    release.write_text(json.dumps({'released_on': '2026-10-17', 'status': 'research'}))
    with pytest.raises(ValueError, match='approved calibrated'):
        registry.register(artifact, release)
    release.write_text(json.dumps(approval(artifact)))
    first = registry.register(artifact, release)
    registry.activate(first)
    model = DummyClassifier(strategy='prior').fit(pd.DataFrame({'x': [0., 1., 2.]}), [0, 1, 1])
    save_model(model, ['x'], artifact, trained_through='2022-12-31', target='target_bankruptcy_next_1yr')
    release.write_text(json.dumps(approval(artifact)))
    second = registry.register(artifact, release)
    registry.activate(second)
    registry.rollback(first)
    assert (tmp_path / 'registry/current').resolve().name == first
    (tmp_path / 'registry/versions' / second / 'distress.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        registry.activate(second)
    assert (tmp_path / 'registry/current').resolve().name == first


def test_fresh_calibration_freezes_artifact_and_consumes_test_once(tmp_path, monkeypatch):
    from src.ml.artifacts import save_model
    from src.validation import prospective_evaluation as module
    import src.validation.future_cohort as future
    monkeypatch.setattr(future, 'exposed_companies', lambda **kwargs: set())
    artifact = tmp_path / 'model.joblib'
    model = DummyClassifier(strategy='prior').fit(pd.DataFrame({'x': [0., 1.]}), [0, 1])
    save_model(model, ['x'], artifact, trained_through='2022-12-31', target=module.TARGET)
    reservation = tmp_path / 'reservation.json'
    reservation.write_text(json.dumps({'final_start': '2024-12-31', 'companies': [
        {'cik': '0009999991', 'role': 'final'}, {'cik': '0009999992', 'role': 'final'}]}))
    frozen = tmp_path / 'frozen.json'
    module.freeze(reservation, artifact, frozen, threshold=.5)
    reviews = tmp_path / 'reviews.jsonl'
    reviews.write_text('\n'.join(json.dumps(review(tmp_path, cik=cik, label=label)) for cik, label in [('0009999991', 0), ('0009999992', 1)]))
    features = tmp_path / 'features.csv'
    pd.DataFrame([{'cik': cik, 'prediction_date': '2024-12-31', 'feature_filing_date': '2024-11-01',
                   'latest_input_filing_date': '2024-11-01', 'x': .2} for cik in ['0009999991', '0009999992']]).to_csv(features, index=False)
    output = tmp_path / 'evaluation'
    result = module.evaluate(frozen, features, reviews, output, as_of='2026-10-17')
    assert result['companies'] == 2 and result['positive_companies'] == 1
    assert not result['production_ready']
    with pytest.raises(FileExistsError):
        module.evaluate(frozen, features, reviews, output, as_of='2026-10-17')
    with pytest.raises(FileExistsError):
        module.evaluate(frozen, features, reviews, tmp_path / 'different-output', as_of='2026-10-17')
    with pytest.raises(ValueError, match='already consumed'):
        module.freeze(reservation, artifact, tmp_path / 'different-candidate.json')


def test_authenticated_operator_status_is_read_only(monkeypatch):
    import src.operations.status as module
    monkeypatch.setattr(module.S3Cache, 'from_env', lambda: object())
    monkeypatch.setattr(module.MonitorStore, 'read_json', lambda self, key: {
        'state': 'success', 'completed_at': datetime.now(timezone.utc).isoformat()})
    event = {'rawPath': '/status', 'requestContext': {'authorizer': {'iam': {'userArn': 'arn:aws:iam::123456789012:role/test'}}, 'http': {'method': 'GET'}}}
    result = module.handler(event, None)
    assert result['statusCode'] == 200
    assert not json.loads(result['body'])['production_model_approved']
    event['requestContext']['http']['method'] = 'POST'
    assert module.handler(event, None)['statusCode'] == 404


def test_reviewed_development_tuning_freezes_without_accessing_final(tmp_path, monkeypatch):
    import src.validation.tune_reviewed as tuning
    import src.validation.future_cohort as future
    monkeypatch.setattr(tuning, 'MODELS', ['logistic_regularized'])
    monkeypatch.setattr(future, 'exposed_companies', lambda **kwargs: set())
    companies, features, reviews = [], [], []
    for index in range(40):
        cik = str(9000100 + index).zfill(10)
        role = 'training' if index < 20 else 'calibration'
        year = 2018 if role == 'training' else 2021
        label = index % 2
        companies.append({'cik': cik, 'role': role})
        item = review(tmp_path, cik=cik, label=label)
        item['prediction_date'] = f'{year}-12-31'
        if label:
            item['event_date'] = f'{year+1}-06-01'
        else:
            item['coverage_through'] = f'{year+1}-12-31'
        reviews.append(item)
        features.append({'cik': cik, 'prediction_date': f'{year}-12-31',
                         'feature_filing_date': f'{year}-11-01', 'latest_input_filing_date': f'{year}-11-01',
                         'wc_to_ta': .8 if label else -.8, 'design_weight': 1.})
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'seed': 'unit-test-only', 'final_start': '2024-12-31', 'companies': companies + [{'cik': '0009999999', 'role': 'final'}]}))
    feature_path, review_path = tmp_path / 'features.csv', tmp_path / 'reviews.jsonl'
    pd.DataFrame(features).to_csv(feature_path, index=False)
    review_path.write_text('\n'.join(json.dumps(r) for r in reviews))
    output = tmp_path / 'candidate'
    report = tuning.tune(plan, feature_path, review_path, output, as_of='2026-10-17')
    assert not report['final_used_for_selection']
    assert not report['production_ready']
    assert (output / 'frozen_candidate.json').exists()
    assert not plan.with_suffix('.evaluation_started.json').exists()


def test_known_case_additions_are_restricted_to_training(tmp_path, monkeypatch):
    import src.validation.future_cohort as module
    monkeypatch.setattr(module, 'exposed_companies', lambda: {'0000000001'})
    monkeypatch.setattr(module, 'PLAN', tmp_path / 'missing')
    plan = module.reserve(['9999991'], tmp_path / 'plan.json', seed='fixed', final_start='2024-12-31', training_cases=['1'])
    known = [c for c in plan['companies'] if c['cik'] == '0000000001']
    assert known[0]['role'] == 'training'
    assert plan['labels_present_at_reservation']
    assert not plan['final_labels_present_at_reservation']
