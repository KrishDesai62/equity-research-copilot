import pandas as pd

from src.validation.openfundex_audit import audit_frame
from src.validation.refine_development import development_rows, FEATURES, TARGET


def test_publisher_labels_are_quarantined_when_event_dates_disagree():
    frame = pd.DataFrame({'cik': [1, 2, 3], 'adsh': ['a', 'b', 'c'],
        'filing_date': ['2018-01-01'] * 3, 'period_end_date': ['2017-12-31'] * 3,
        'bankruptcy_date': ['2018-06-01', '2017-12-01', None], 'target_bankrupt_1y': [False, True, False]})
    registry = pd.DataFrame({'cik': ['0000000001'], 'filing_date': ['2018-06-01']})
    result = audit_frame(frame, registry)
    assert result['known_event_inside_filing_horizon_but_label_not_positive'] == 1
    assert result['positive_label_outside_known_filing_horizon'] == 1
    assert result['brd_positive_windows_without_upstream_positive'] == 1
    assert result['new_reviewed_negative_labels'] == 0
    assert not result['eligible_for_training']


def test_development_refinement_never_converts_final_labels(tmp_path):
    row = {name: 1. for name in FEATURES}
    row.update(cik='0000000001', role='training', eligible=True, valid_for_training=True,
        design_weight=1., Assets=100., Cash=10., Liabilities=80., NetIncomeLoss_ttm=10., OperatingCashFlow_ttm=-5.)
    row[TARGET] = 0
    final = dict(row, role='final')
    final[TARGET] = 'do-not-inspect-or-convert'
    path = tmp_path / 'source.csv'
    pd.DataFrame([row, final]).to_csv(path, index=False)
    result = development_rows(path)
    assert len(result) == 1
    assert result.iloc[0].accruals_to_assets == .15
    assert result.iloc[0].operating_cash_burn_to_cash == .5


def test_primary_petition_dates_exclude_future_and_unrelated_filings():
    from src.validation.verify_public_events import petition_dates
    text = '<p>On April 1, 2024, the Company filed voluntary petitions under Chapter 11.</p>'
    assert petition_dates(text, '2024-04-02') == ['2024-04-01']
    assert petition_dates(text, '2024-03-31') == []
    assert petition_dates(text, '2024-06-01') == []
    assert petition_dates('On April 1, 2024, the company filed its annual report.', '2024-04-02') == []
