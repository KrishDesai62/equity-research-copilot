"""New development studies: temporal evidence and unbiased acquisition controls."""
import numpy as np
import pandas as pd
import pytest

from src.validation.historical_training_expansion import union_probability
from src.validation.filing_text_signals import document_for, extract_signals, visible_text
from src.validation.development_errors import operating_limits


def test_training_union_weights_apply_to_both_acquisition_phases():
    assert union_probability(.1, 1) == 1
    assert union_probability(.1, .1) == pytest.approx(.19)
    assert union_probability(1, .1) == 1
    assert union_probability(.1, 0) == .1
    with pytest.raises(ValueError):union_probability(.1, -1)


def test_error_analysis_reports_precision_recall_tradeoff_with_design_weights():
    from src.validation.sec_disclosure_data import TARGET
    frame=pd.DataFrame({TARGET:[1,1,0], 'design_weight':[1.,1.,20.], 'p':[.8,.4,.7]})
    result=operating_limits(frame,'p')
    assert result['maximum_recall_at_precision_50pct'] == .5
    assert result['maximum_precision_at_recall_50pct'] == 1.


def test_visible_text_drops_hidden_xbrl_and_scripts_preserving_visible_tail():
    payload=b'<html><head><title>going concern</title></head><body><ix:hidden>substantial doubt</ix:hidden><script>going concern</script><div style="DISPLAY: none">going concern</div><p>ordinary operations</p> visible tail</body></html>'
    text=visible_text(payload)
    assert 'going concern' not in text and 'substantial doubt' not in text
    assert 'ordinary operations' in text and 'visible tail' in text


def test_text_rates_use_visible_words_not_hidden_terms():
    payload=('<html><body><div hidden>going concern</div><p>'+('ordinary operations '*220)+
        ' substantial doubt about ability to continue as a going concern </p></body></html>').encode()
    values,evidence=extract_signals(payload)
    assert values['text_going_concern_rate']>0
    assert values['text_going_concern_rate']==values['text_substantial_doubt_rate']
    assert values['text_covenant_breach_rate']==0
    assert evidence['word_count']>400 and evidence['matches']


def test_empty_or_access_response_cannot_be_zero_text_features():
    with pytest.raises(ValueError):extract_signals(b'')
    with pytest.raises(ValueError):extract_signals(b'<html>Request rate threshold exceeded</html>')
    with pytest.raises(ValueError):extract_signals(b'<html>short document</html>')


def test_exact_original_document_is_required_before_prediction():
    row={'cik':'0000000001','accession_number':'0000000001-20-000001@2020-12-31',
        'source_accession_number':'0000000001-20-000001','feature_filing_date':'2020-11-15',
        'prediction_date':'2020-12-31'}
    meta={'0000000001-20-000001':{'filing_date':'2020-11-15','filing_type':'10-Q','source_url':'https://www.sec.gov/example'}}
    assert document_for(row,meta)['filing_date']=='2020-11-15'
    bad=dict(row,feature_filing_date='2021-01-01')
    with pytest.raises(ValueError):document_for(bad,meta)
    meta['0000000001-20-000001']['filing_type']='10-Q/A'
    with pytest.raises(ValueError):document_for(row,meta)


def test_review_quorum_cannot_be_created_by_duplicate_names_or_empty_templates():
    from src.validation.disclosure_review_workflow import assess
    from datetime import datetime,timezone
    from src.validation.sec_disclosure_data import TARGET
    task={'unit_id':'1@2024-12-31','source_checkpoint':{'sha256':'source'},'evidence_cutoff':'2026-07-19','automated_proposed_label':0}
    now=datetime(2026,10,19,tzinfo=timezone.utc)
    response={'unit_id':task['unit_id'],'target':TARGET,'packet_sha256':'packet','checkpoint_sha256':'source',
        'reviewer_id':'Alice','reviewed_at':'2026-10-18T12:00:00+00:00','approved_label':0,
        'independent_review_attestation':True,'checked_primary_sources':True,
        'full_horizon_disclosure_coverage_checked':True,'notes':'Reviewed complete source interval.'}
    assert not assess({task['unit_id']:task},[],'packet',now=now)['independent_reviews_complete']
    with pytest.raises(ValueError,match='Duplicate'):
        assess({task['unit_id']:task},[response,dict(response,reviewer_id=' ALICE ')],'packet',now=now)
    approved=assess({task['unit_id']:task},[response,dict(response,reviewer_id='Bob')],'packet',now=now)
    assert approved['independent_reviews_complete'] and not approved['production_approval']
    bad=dict(response,reviewer_id='')
    with pytest.raises(ValueError):assess({task['unit_id']:task},[bad],'packet',now=now)


def test_reviewer_label_disagreement_requests_revision_not_dataset_mutation():
    from src.validation.disclosure_review_workflow import assess
    from datetime import datetime,timezone
    from src.validation.sec_disclosure_data import TARGET
    task={'unit_id':'u','source_checkpoint':{'sha256':'s'},'evidence_cutoff':'2026-07-19','automated_proposed_label':1}
    record={'unit_id':'u','target':TARGET,'packet_sha256':'p','checkpoint_sha256':'s',
        'reviewer_id':'one','reviewed_at':'2026-10-18T12:00:00+00:00','approved_label':0,
        'independent_review_attestation':True,'checked_primary_sources':True,
        'full_horizon_disclosure_coverage_checked':True,'notes':'Source correction requested.'}
    result=assess({'u':task},[record,dict(record,reviewer_id='two')],'p',now=datetime(2026,10,19,tzinfo=timezone.utc))
    assert result['label_revision_required_units']==['u'] and not result['independent_reviews_complete']
    assert task['automated_proposed_label']==1


def test_weight_concentration_is_visible_and_weights_stay_unchanged():
    import pandas as pd
    from src.validation.development_release_preflight import weight_support
    from src.validation.sec_disclosure_data import TARGET
    frame = pd.DataFrame({'unit_id':['a','b','c'], 'cik':['a','b','c'],
                          TARGET:[1,1,0], 'design_weight':[20.,1.,1.]})
    before = frame.copy(deep=True)
    support = weight_support(frame)
    assert support['positive']['effective_sample_size'] == pytest.approx(441/401)
    assert support['positive']['largest_company_weight_fraction'] == pytest.approx(20/21)
    pd.testing.assert_frame_equal(frame, before)


def test_release_preflight_cannot_pass_missing_review_or_threshold():
    from src.validation.development_release_preflight import release_blockers
    assert len(release_blockers({'development_feasible':False}, {})) == 2
    assert len(release_blockers({'development_feasible':True}, {})) == 1


def test_constant_baseline_does_not_use_held_fold_labels():
    import numpy as np
    from src.validation.text_model_development import crossfit_baseline
    y = np.array([0,1,0,1]); w = np.ones(4); folds = np.array([0,0,1,1])
    original = crossfit_baseline(y,w,folds)
    y[:2] = 1
    changed = crossfit_baseline(y,w,folds)
    np.testing.assert_array_equal(original[:2], changed[:2])
    assert changed[2] != original[2]


def test_development_union_includes_competing_reservations():
    from src.validation.recent_development_expansion import calibration_union_probability as pi
    # Original background can enter development in the first sampling stage,
    # or after escaping both subsequent final/training reservations.
    assert pi(.1,.5,.1,.1,.1) == pytest.approx(.1229)
    assert pi(.1,.5,0,0,1) == pytest.approx(.95)
    assert pi(.1,.5,1,0,1) == pytest.approx(.05)
    assert pi(1,.9,0,0,1) == pytest.approx(.9)
    with pytest.raises(ValueError):pi(.1,.5,-.1,0,1)


def test_parent_named_as_owner_is_not_petitioning_issuer():
    from src.validation.issuer_identity_admission import corroborate_identity, direct_issuer_subject
    text=('On September 19, 2023 (the Petition Date), UpHealth Holdings, Inc., '
          'a wholly-owned direct subsidiary of UpHealth, Inc. (the Company), filed a voluntary '
          'petition for relief under Chapter 11. Neither the Company nor any other subsidiary filed.')
    events=corroborate_identity(text,filed='2023-09-19',names=['UpHealth, Inc.'])
    assert events and not any(e['issuer_explicitly_included'] for e in events)
    assert direct_issuer_subject(' Parent, Inc. (the Company), ',['Parent, Inc.'])
    assert direct_issuer_subject(' the Company and its subsidiaries ',['Parent, Inc.'])
    assert not direct_issuer_subject(" Parent, Inc.'s subsidiaries ",['Parent, Inc.'])


def test_unconfirmed_old_positive_becomes_unknown_never_negative():
    from src.validation.issuer_identity_admission import apply
    from src.validation.sec_disclosure_data import TARGET
    f=pd.DataFrame({'cik':['1'],'unit_id':['u'],TARGET:[1],
                    'prediction_date':['2022-12-31'],'horizon_end':['2023-12-31'],
                    'label_available_date':['2024-07-18']})
    changed,notes=apply(f,{'1':[]})
    assert pd.isna(changed.iloc[0][TARGET]) and notes[0]['new_label'] is None
    assert f.iloc[0][TARGET]==1


def test_issuer_legal_alias_punctuation_preserves_true_cases():
    from src.validation.issuer_identity_admission_v2 import direct_issuer_subject
    assert direct_issuer_subject('Lannett Company, Inc. ("LCI") and subsidiaries',['Lannett Co Inc'])
    assert direct_issuer_subject('after considering all strategic alternatives, Infinity Pharmaceuticals, Inc. (the "Company")',['Infinity Pharmaceuticals Inc'])
    assert direct_issuer_subject('PARTS iD, Inc., a Delaware corporation (the Company) and its subsidiary',['PARTS ID Inc'])
    assert not direct_issuer_subject('TopPop LLC, a wholly-owned subsidiary of Iconic Brands, Inc. (the Company)',['Iconic Brands Inc'])
    assert not direct_issuer_subject('UpHealth Holdings, Inc., a subsidiary of UpHealth, Inc. (the Company)',['UpHealth Inc'])
    assert not direct_issuer_subject('Parent, Inc. Holdings LLC',['Parent Inc'])
    assert not direct_issuer_subject("Parent, Inc.'s subsidiaries",['Parent Inc'])
