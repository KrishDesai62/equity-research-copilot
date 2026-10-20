"""Outcome admission, temporal follow-up and sampling invariants."""
import json

import pandas as pd
import pytest

from src.validation.adjudicate_public_events import corroborate
from src.validation.sec_disclosure_data import CONFIG, decide_label, followup_complete, sampling_record


def test_agreement_date_cannot_become_petition_date():
    text = ('On April 30, 2023, Example Inc. entered into an agreement. '
            'On May 2, 2023, the Company filed voluntary petitions under Chapter 11.')
    found = corroborate(text, filed='2023-05-03', names=['Example Inc.'])
    assert [(a['petition_date'], a['issuer_explicitly_included']) for a in found] == [('2023-05-02', True)]


def test_subsidiary_only_cannot_establish_parent_event():
    found = corroborate('On May 2, 2023, Subsidiary LLC filed voluntary petitions under Chapter 11.',
        filed='2023-05-03', names=['Example Inc.'])
    assert found and not found[0]['issuer_explicitly_included']
    label, reason = decide_label(found, ['unresolved'], followup=True, anchor='2022-12-31', horizon='2023-12-31')
    assert label is None and reason == 'unresolved_item_103'


@pytest.mark.parametrize('text', [
    'On May 2, 2023, the Company filed voluntary petitions under Chapter 15.',
    'On May 2, 2023, the Company would have filed voluntary petitions under Chapter 11.',
    'On May 2, 2023, the Company filed involuntary petitions under Chapter 11.',
])
def test_non_target_statements_not_positive(text):
    assert not corroborate(text, filed='2023-05-03', names=['Example Inc.'])


def test_negative_requires_both_no_ambiguity_and_completed_followup():
    kwargs = dict(anchor='2022-12-31', horizon='2023-12-31')
    assert decide_label([], [], followup=True, **kwargs)[0] == 0
    assert decide_label([], ['amendment'], followup=True, **kwargs)[0] is None
    assert decide_label([], [], followup=False, **kwargs)[0] is None
    positive = dict(issuer_explicitly_included=True, petition_date='2023-02-01')
    assert decide_label([positive], [], followup=False, **kwargs)[0] == 1
    positive['petition_date'] = '2022-02-01'
    assert decide_label([positive], [], followup=True, **kwargs)[0] is None


def test_post_horizon_event_does_not_make_horizon_positive():
    event = dict(issuer_explicitly_included=True, petition_date='2024-01-01')
    assert decide_label([event], [], followup=True, anchor='2022-12-31', horizon='2023-12-31')[0] == 0


def test_reporting_followup_requires_horizon_and_no_long_gaps():
    frame = pd.DataFrame({'filing_date': ['2023-02-15', '2023-05-15', '2023-08-15', '2023-11-15', '2024-02-15'],
        'fiscal_period_end': ['2022-12-31', '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31']})
    kwargs = dict(baseline_date='2022-11-15', anchor='2022-12-31', horizon='2023-12-31', cutoff='2024-07-18')
    assert followup_complete(frame, **kwargs)
    assert not followup_complete(frame.iloc[:-1], **kwargs)
    assert not followup_complete(frame.drop(index=[1, 2]), **kwargs)


def test_joint_weights_apply_even_to_negative_snapshots_of_proxy_cases():
    config = json.loads(CONFIG.read_text())
    for year in [None, 2023, 2024, 2025]:
        for index in range(100):
            record = sampling_record(str(index).zfill(10), year, config)
            if record:
                assert record['design_weight'] == pytest.approx(1 / (record['sampling_probability'] * record['role_probability']))
                assert 0 < record['role_probability'] < 1
                assert record == sampling_record(str(index).zfill(10), year, config)


def test_acquisition_errors_block_fitting_before_artifact_creation(tmp_path, monkeypatch):
    from src.validation import sec_disclosure_experiment as experiment
    (tmp_path / 'report.json').write_text(json.dumps({'errors': [{'cik': '0000000001', 'error': 'timeout'}]}))
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Resolve acquisition errors'):
        experiment.select()
    assert not (tmp_path / 'experiment').exists()


def test_small_final_cohort_is_not_consumed(tmp_path, monkeypatch):
    from src.validation import sec_disclosure_experiment as experiment
    output = tmp_path / 'experiment'
    output.mkdir()
    (output / 'selection.json').write_text('{}')
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'EXPERIMENT', output)
    monkeypatch.setattr(experiment, 'verify_acquisition', lambda: {})
    monkeypatch.setattr(experiment, 'verify_selection', lambda: {'training_companies': [], 'calibration_companies': []})
    monkeypatch.setattr(experiment, 'read_role', lambda role: pd.DataFrame({'cik': ['0000000001'], experiment.TARGET: [1]}))
    result = experiment.evaluate_once()
    assert result['final_test_consumed'] is False
    assert not (tmp_path / 'final_evaluation_started.json').exists()
    assert not (output / 'final_predictions.csv').exists()


def test_missing_financials_resolution_never_changes_an_outcome(tmp_path):
    from src.validation.sec_disclosure_availability import resolve_missing_financials
    raw = tmp_path / 'raw'
    folder = raw / 'companyfacts'
    folder.mkdir(parents=True)
    cik = '0000000001'
    (folder / f'CIK{cik}.json').write_text(json.dumps({'cik': 1, 'facts': {'us-gaap': {}}}))
    report = {'errors': [{'cik': cik, 'error': f'No original 10-K/10-Q USD Assets facts for CIK{cik}'}]}
    (tmp_path / 'report.json').write_text(json.dumps(report))
    result = resolve_missing_financials(report, raw, tmp_path)
    assert result['outcome_labels_changed'] == 0
    assert result['resolved_financial_exclusions'][0]['no_outcome_label_assigned']
    assert resolve_missing_financials(report, raw, tmp_path) == result
    (folder / f'CIK{cik}.json').write_text(json.dumps({'cik': 1, 'facts': {'us-gaap': {'Assets': {'units': {'USD': [{'form': '10-K'}]}}}}}))
    with pytest.raises(ValueError, match='contradicted'):
        resolve_missing_financials(report, raw, tmp_path)
