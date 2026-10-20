import json
from pathlib import Path
import pandas as pd
import pytest
from src.validation.outcome_expansion_sources import CONFIG,chronology


def test_added_training_outcomes_mature_before_development_forecast():
    config=json.loads(CONFIG.read_text())
    assert chronology(config)==('2023-05-19','2024-12-17')
    config['new_training_anchor']='2021-12-31'
    with pytest.raises(ValueError,match='postdate'):chronology(config)


def test_fresh_reservation_uses_complete_window_and_disjoint_companies():
    config=json.loads(CONFIG.read_text());plan=json.loads(Path('config/outcome_expansion_20261019/acquisition.json').read_text())
    end=pd.Timestamp(config['fresh_evaluation_anchor'])+pd.DateOffset(years=1)+pd.Timedelta(days=config['reporting_grace_days'])
    assert end.date().isoformat()=='2026-10-18'
    fresh={c['cik'] for c in plan['fresh_reservation']};train={c['cik'] for c in plan['training']}
    assert fresh and not fresh&train and not fresh&set(plan['prior_selected_ids'])
    assert len(train)==len(plan['training'])
    assert all(0<c['inclusion_probability']<=1 for c in plan['training']+plan['fresh_reservation'])


@pytest.mark.skipif(not Path('data/processed/validation/fresh_reservation_reviews_20261019/manifest.json').is_file(), reason="Requires locally generated research evidence, excluded from Git")
def test_reserved_review_packet_does_not_claim_approval_or_predictions():
    folder=Path('data/processed/validation/fresh_reservation_reviews_20261019')
    manifest=json.loads((folder/'manifest.json').read_text())
    tasks=[json.loads(line) for line in (folder/'pending_reviews.jsonl').read_text().splitlines()]
    assert manifest['independent_reviews_completed']==0 and manifest['production_approval'] is False
    assert manifest['positive_support_gate_passed']==(manifest['positive_units']>=20)
    assert len(tasks)==manifest['review_units']
    for task in tasks:
        assert task['approved_label'] is None and task['reviewers']==[]
        assert task['model_predictions_included'] is False
        if task['automated_proposed_label']==1:assert task['positive_statement_evidence']
        else:assert task['reporting_followup_complete']


@pytest.mark.skipif(not Path('data/processed/validation/expanded_training_reviews_20261019/pending_reviews.jsonl').is_file(), reason="Requires locally generated research evidence, excluded from Git")
def test_new_training_positive_packet_has_in_window_primary_evidence():
    folder=Path('data/processed/validation/expanded_training_reviews_20261019')
    tasks=[json.loads(line) for line in (folder/'pending_reviews.jsonl').read_text().splitlines()]
    assert tasks
    for task in tasks:
        assert task['approved_label'] is None and not task['reviewers']
        assert not task['model_predictions_included']
        assert task['positive_statement_evidence']
        for proof in task['positive_statement_evidence']:
            assertion=proof['assertion']
            assert assertion['issuer_explicitly_included']
            assert task['prediction_date']<assertion['petition_date']<=task['horizon_end']
            assert assertion['filing_date']<=task['evidence_cutoff']
