"""Discovery completeness and sequential-cohort weighting regressions."""
import copy

import pandas as pd
import pytest

from src.validation.sec_disclosure_extension import union_probability, combined_rows
from src.validation.sec_event_inventory import validate_page, candidates_from_hits


@pytest.mark.parametrize('proxy,oldpi,role,newpi,expected', [
    (False, .1, .5, 1., .95), (False, .1, .5, .1, .14),
    (True, 1., .9, 1., .9), (True, 1., .1, .1, .1),
])
def test_union_probability_accounts_for_first_and_second_stage(proxy, oldpi, role, newpi, expected):
    assert union_probability(original_proxy=proxy, original_sampling=oldpi,
        final_role=role, extension_sampling=newpi) == pytest.approx(expected)


def test_union_probability_matches_enumerated_sampling_paths():
    # First-stage calibration is deliberately unavailable to the extension.
    for newpi in [.1, 1.]:
        first_final, first_cal, not_first = .1 * .5, .1 * .5, .9
        total = first_final + not_first * newpi
        excluded = first_cal + not_first * (1 - newpi)
        assert total + excluded == pytest.approx(1.)
        assert union_probability(original_proxy=False, original_sampling=.1,
            final_role=.5, extension_sampling=newpi) == pytest.approx(total)


def test_bad_sampling_probabilities_fail_closed():
    with pytest.raises(ValueError):
        union_probability(original_proxy=True, original_sampling=.1, final_role=.5, extension_sampling=1.)
    with pytest.raises(ValueError):
        union_probability(original_proxy=False, original_sampling=.1, final_role=.5, extension_sampling=0.)


def fixture_page():
    config = {'page_size': 100, 'item': '1.03', 'start': '2025-01-01', 'end': '2026-07-19'}
    payload = {'hits': {'total': {'relation': 'eq', 'value': 126}},
        'query': {'from': 0, 'size': 100, 'query': {'bool': {'filter': [
            {'terms': {'root_forms': ['8-K']}}, {'terms': {'items': ['1.03']}},
            {'range': {'file_date': {'gte': config['start'], 'lte': config['end']}}}]}}}}
    return config, payload


def test_discovery_validates_server_filters_and_pagination():
    config, payload = fixture_page()
    assert validate_page(payload, config, 0) == 126
    bad = copy.deepcopy(payload)
    bad['query']['query']['bool']['filter'].pop(1)
    with pytest.raises(ValueError, match='filters'):
        validate_page(bad, config, 0)
    with pytest.raises(ValueError, match='pagination'):
        validate_page(payload, config, 100)


@pytest.mark.parametrize('change', ['truncated', 'timed_out', 'shard_failed'])
def test_incomplete_search_cannot_establish_inventory(change):
    config, payload = fixture_page()
    if change == 'truncated':
        payload['hits']['total']['relation'] = 'gte'
    elif change == 'timed_out':
        payload['timed_out'] = True
    else:
        payload['_shards'] = {'failed': 1}
    with pytest.raises(ValueError):
        validate_page(payload, config, 0)


def test_search_hit_is_candidate_never_an_outcome_label():
    config, _ = fixture_page()
    hit = {'_id': '0000000001-25-000001:test.htm', '_source': {
        'adsh': '0000000001-25-000001', 'ciks': ['0000000001'],
        'form': '8-K/A', 'items': ['1.03'], 'file_date': '2025-05-01', 'display_names': ['Example Inc']}}
    rows = candidates_from_hits([hit], config)
    assert rows[0]['label_admitted'] is False
    hit['_source']['items'] = ['1.01']
    with pytest.raises(ValueError, match='Unrelated'):
        candidates_from_hits([hit], config)


def test_original_final_weights_are_updated_by_same_union_design(tmp_path, monkeypatch):
    from src.validation import sec_disclosure_extension as extension
    old = pd.DataFrame({'cik': ['0000000001', '0000000002'], 'role': ['final', 'final'],
        'proxy_year': [None, 2025], 'sampling_probability': [.1, 1.],
        'role_probability': [.5, .9], 'design_weight': [20., 1 / .9]})
    old.to_csv(tmp_path / 'dataset.csv', index=False)
    monkeypatch.setattr(extension, 'ORIGINAL_ROOT', tmp_path)
    result = combined_rows([], {'0000000001'}, {})
    assert result.design_weight.tolist() == pytest.approx([1 / .95, 1 / .9])
    assert pd.read_csv(tmp_path / 'dataset.csv').design_weight.tolist() == pytest.approx([20., 1 / .9])


def test_later_explicit_singular_case_statement_is_valid_positive_evidence():
    from src.validation.sec_disclosure_admission import corroborate_v2
    found = corroborate_v2('On February 10, 2025, Example Inc. commenced a chapter 11 case as debtor in possession.',
        filed='2025-04-10', names=['Example Inc.'])
    assert len(found) == 1 and found[0]['issuer_explicitly_included']
    assert found[0]['petition_date'] == '2025-02-10'
    assert not corroborate_v2('On February 10, 2024, Example Inc. commenced a chapter 11 case.',
        filed='2025-02-11', names=['Example Inc.'])


def test_updated_admission_keeps_date_binding_and_entity_guards():
    from src.validation.sec_disclosure_admission import corroborate_v2
    text = ('On April 30, 2025, Example Inc. entered into an agreement. '
        'On May 2, 2025, Subsidiary LLC commenced a chapter 11 case.')
    found = corroborate_v2(text, filed='2025-05-03', names=['Example Inc.'])
    assert [f['petition_date'] for f in found] == ['2025-05-02']
    assert not found[0]['issuer_explicitly_included']


def test_source_admission_does_not_create_zero_or_resolve_conflicts_by_guessing():
    from src.validation.sec_disclosure_admission import apply_evidence, TARGET
    frame = pd.DataFrame({'cik': ['1', '2', '3'], 'unit_id': ['1@date', '2@date', '3@date'],
        'prediction_date': ['2024-12-31'] * 3, 'horizon_end': ['2025-12-31'] * 3,
        'label_available_date': ['2026-07-19'] * 3, 'label_reason': ['pending'] * 3,
        TARGET: [None, 0, None]})
    event = {'issuer_explicitly_included': True, 'petition_date': '2025-05-01', 'filing_date': '2025-05-02'}
    result, changes = apply_evidence(frame, {'1': [event], '2': [event]})
    assert result.loc[0, TARGET] == 1
    assert pd.isna(result.loc[1, TARGET])  # Contradictory negative needs review.
    assert pd.isna(result.loc[2, TARGET])  # No evidence cannot supply a zero.
    assert len(changes) == 2


def test_outcome_normalization_excludes_only_irrelevant_form4():
    from src.validation.sec_disclosure_repair import outcome_submission_rows
    block = {'form': ['4', '8-K'], 'accessionNumber': ['0000000001-25-000001', '0000000001-25-000002'],
        'filingDate': ['2025-05-01'] * 2, 'primaryDocument': ['', 'event.htm'], 'items': ['', '1.03']}
    rows = outcome_submission_rows(block, '0000000001', 'TEST', '2025-01-01', '2025-12-31')
    assert len(rows) == 1 and rows[0]['filing_type'] == '8-K'
    block['primaryDocument'][1] = ''
    with pytest.raises(ValueError, match='Invalid SEC document path'):
        outcome_submission_rows(block, '0000000001', 'TEST', '2025-01-01', '2025-12-31')


def test_development_temporal_and_company_separation():
    from src.validation.sec_disclosure_models_v2 import validate_development, TARGET
    train = pd.DataFrame({'cik': ['1', '2'], 'unit_id': ['1@date', '2@date'], TARGET: [0, 1],
        'prediction_date': ['2020-12-31'] * 2, 'feature_filing_date': ['2020-11-15'] * 2,
        'latest_input_filing_date': ['2020-11-15'] * 2, 'label_available_date': ['2022-07-19'] * 2})
    cal = pd.DataFrame({'cik': ['3', '4'], 'unit_id': ['3@date', '4@date'], TARGET: [0, 1],
        'prediction_date': ['2022-12-31'] * 2, 'source_filing_date': ['2022-11-15'] * 2,
        'latest_input_filing_date': ['2022-11-15'] * 2, 'label_available_date': ['2024-07-18'] * 2})
    validate_development(train, cal)
    bad = train.copy()
    bad['label_available_date'] = '2023-01-01'
    with pytest.raises(ValueError, match='time leakage'):
        validate_development(bad, cal)
    bad = cal.copy()
    bad.loc[0, 'cik'] = '1'
    with pytest.raises(ValueError, match='overlapping'):
        validate_development(train, bad)
    bad = cal.copy()
    bad.loc[0, 'latest_input_filing_date'] = '2023-01-01'
    with pytest.raises(ValueError, match='Future financial'):
        validate_development(train, bad)


def test_balancing_uses_design_weights_instead_of_raw_class_counts():
    from src.validation.sec_disclosure_models_v2 import train_extra, TARGET, FEATURES
    import numpy as np
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(8, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = [0] * 6 + [1] * 2
    frame['design_weight'] = [10.] * 6 + [1.] * 2
    model = train_extra(frame, 'logistic_c1_balanced')
    balance = model.named_steps['classifier'].class_weight
    assert balance[1] / balance[0] == pytest.approx(30.)
    probabilities = model.predict_proba(frame[FEATURES])
    assert np.isfinite(probabilities).all()


def test_chapter7_liquidation_is_issuer_event_but_subsidiary_is_not():
    from src.validation.sec_disclosure_admission_v3 import corroborate_v3
    text = 'On January 17, 2025, Example Inc. and certain subsidiaries filed for liquidation under Chapter 7.'
    found = corroborate_v3(text, filed='2025-01-21', names=['Example Inc.'])
    assert found[0]['chapter'] == '7' and found[0]['issuer_explicitly_included']
    found = corroborate_v3(text.replace('Example Inc.', 'Subsidiary LLC'), filed='2025-01-21', names=['Example Inc.'])
    assert not found[0]['issuer_explicitly_included']


def test_sec_state_annotation_does_not_prevent_exact_legal_name_matching():
    from src.validation.sec_disclosure_admission_v3 import corroborate_v3, legal_name
    assert legal_name('Example Inc./DE') == 'Example Inc.'
    assert legal_name('Example Inc.') == 'Example Inc.'
    found = corroborate_v3('On December 15, 2025, Example Inc. filed voluntary petitions under Chapter 11.',
        filed='2025-12-17', names=['EXAMPLE INC./DE'])
    assert found[0]['issuer_explicitly_included']


def test_convergence_variant_copies_readonly_pandas_weights():
    import numpy as np
    from src.validation.sec_disclosure_models_v3 import fit_current, TARGET, FEATURES
    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(12, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = [0] * 8 + [1] * 4
    frame['design_weight'] = [10.] * 8 + [1.] * 4
    before = frame.design_weight.copy()
    model = fit_current(frame, 'logistic_balanced')
    assert model.named_steps['classifier'].max_iter == 10000
    assert model.named_steps['classifier'].C == .1
    pd.testing.assert_series_equal(before, frame.design_weight)


def test_filing_signals_never_include_prediction_day_or_future_events():
    from src.validation.sec_filing_signals import signals_at
    rows = [{'date': '2024-12-01', 'form': '8-K', 'items': '2.04'},
        {'date': '2024-12-31', 'form': '8-K', 'items': '2.04'},
        {'date': '2025-01-02', 'form': 'NT 10-Q', 'items': ''},
        {'date': '2024-03-01', 'form': '10-K', 'items': ''}]
    result = signals_at(rows, '2024-12-31')
    assert result['material_default_items_90d'] == 1
    assert result['late_report_notices_365d'] == 0
    assert result['filing_signals_latest_public_date'] == '2024-12-01'


def test_unindexed_current_report_makes_item_counts_unknown_not_zero():
    from src.validation.sec_filing_signals import signals_at
    rows = [{'date': '2024-12-01', 'form': '8-K', 'items': ''}]
    result = signals_at(rows, '2024-12-31')
    assert pd.isna(result['material_default_items_90d'])
    assert pd.isna(result['nonreliance_items_365d'])
    assert result['current_reports_90d'] == 1


def test_clipping_does_not_erase_rare_filing_signals():
    import numpy as np
    from src.ml.filing_signal_models import MetadataPreservingClipper
    values = np.zeros((200, 29))
    values[-1, 0] = 1000  # Financial outlier: clip it.
    values[-1, 21] = 1  # Rare public event: retain it.
    values[-1, 28] = 1  # Rare missingness indicator: retain it.
    result = MetadataPreservingClipper().fit_transform(values)
    assert result[-1, 0] == 0
    assert result[-1, 21] == result[-1, 28] == 1
