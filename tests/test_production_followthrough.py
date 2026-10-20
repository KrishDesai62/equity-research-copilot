"""Production target separation and outcome-only final audits."""
from hashlib import sha256
import json

import pandas as pd
import pytest

from src.agents_full.risk_agent import RiskAgentTool
from src.ml.artifacts import save_model
from src.ml.release import assess
from src.ml.train_distress import train_logistic_baseline
from src.validation.final_outcome_audit import COLUMNS, outcome_rows


def setup_model(tmp_path, target):
    frame=pd.DataFrame({'x':[-3.,-2.,-1.,1.,2.,3.],'label':[0,0,0,1,1,1]})
    model=train_logistic_baseline(frame,['x'],'label')
    path=tmp_path/'distress.joblib'
    save_model(model,['x'],path,trained_through='2020-01-01',target=target)
    evidence={'test_metrics':{'precision':.8,'recall':.8},'test_companies':100,
              'independent_positive_events':30,'independent_test':True,
              'representative_population':True,'validated_calibration':True,
              'preprocessing_fit_on':'training rows only'}
    policy=json.loads(open('config/release_policy.json').read())
    release={'status':'approved','calibrated':True,'threshold':.5,'released_on':'2020-02-01',
             'artifact_sha256':sha256(path.read_bytes()).hexdigest(),'evidence':evidence,'assessment':assess(evidence,policy)}
    path.with_suffix('.release.json').write_text(json.dumps(release))
    return pd.DataFrame({'ticker':['TEST'],'cik':['0000000001'],'filing_date':['2020-03-01'],
              'latest_input_filing_date':['2020-03-01'],'accession_number':['0000000001-20-000001'],
              'source_url':['https://www.sec.gov/source'],'x':[2.]})


def test_accounting_enforcement_model_cannot_be_served_as_bankruptcy(tmp_path):
    features=setup_model(tmp_path,'target_aaer_enforcement_flag')
    with pytest.raises(ValueError,match='Unsupported prediction target for distress'):
        RiskAgentTool(tmp_path).get_risk_profile('TEST','2020-04-01',features_df=features)


def test_disclosure_target_is_exposed_with_its_scope(tmp_path):
    features=setup_model(tmp_path,'target_sec_disclosed_petition_1yr')
    result=RiskAgentTool(tmp_path).get_risk_profile('TEST','2020-04-01',features_df=features)
    assert result['prediction_targets']['distress']=='target_sec_disclosed_petition_1yr'
    assert 'not exhaustive court' in result['target_definitions']['distress']
    assert result['distress_probability'] is not None


def test_missing_input_schema_abstains_without_serving_probability(tmp_path):
    features=setup_model(tmp_path,'target_bankruptcy_next_1yr').drop(columns='x')
    result=RiskAgentTool(tmp_path).get_risk_profile('TEST','2020-04-01',features_df=features)
    assert result['distress_probability'] is None
    assert result['unavailable_reasons']['distress']=='feature_schema_mismatch'


def test_outcome_audit_excludes_heldout_model_features(tmp_path,monkeypatch):
    path=tmp_path/'final.csv'
    frame=pd.DataFrame([{c:('0000000001' if c=='cik' else 'value') for c in COLUMNS}])
    frame['cash_to_assets']=999
    frame.to_csv(path,index=False)
    result=outcome_rows(path)
    assert set(result.columns)==set(COLUMNS) and 'cash_to_assets' not in result


def test_deliberation_preface_does_not_hide_direct_issuer_or_admit_owner():
    from src.validation.issuer_identity_admission_v3 import corroborate_identity
    prefix=('On March 14, 2025, after considering its financial and operational condition '
            'and strategic alternatives, including financing and a reorganization under chapter 11, ')
    text=prefix+'Example Inc. (the Company) and its subsidiaries each filed a voluntary petition under Chapter 7.'
    events=corroborate_identity(text,filed='2025-03-14',names=['Example Inc'])
    assert events and events[0]['issuer_explicitly_included'] and events[0]['chapter']=='7'
    text=prefix+'Sub LLC, a wholly-owned subsidiary of Example Inc. (the Company), filed a voluntary petition under Chapter 7.'
    events=corroborate_identity(text,filed='2025-03-14',names=['Example Inc'])
    assert events and not any(e['issuer_explicitly_included'] for e in events)


def test_context_signals_separate_audit_boilerplate_negation_and_warning():
    from src.validation.contextual_filing_signals import signals
    ordinary='ordinary operations '*220
    warning,_=signals(ordinary+' There is substantial doubt about our ability to continue as a going concern.')
    boilerplate,_=signals(ordinary+' Our responsibility is to evaluate whether conditions raise substantial doubt about ability to continue as a going concern.')
    negated,_=signals(ordinary+' There is no substantial doubt about our ability to continue as a going concern.')
    assert warning['context_going_concern_rate']>0
    assert boilerplate['context_going_concern_rate']==0
    assert boilerplate['context_audit_boilerplate_rate']>0
    assert negated['context_going_concern_rate']==0
    assert negated['context_negated_distress_rate']>0


def test_review_of_old_packet_cannot_satisfy_current_outcome_gate():
    from src.validation.current_production_readiness import assess
    model={'selected':{'threshold':None},'development_feasible':False}
    outcomes={'positive_companies':20,'resolved_companies':335}
    review={'packet_sha256':'old','independent_reviews_complete':True}
    blockers=assess(model,outcomes,review,'current')
    assert 'independent_review_of_current_packet' in blockers
    assert 'development_precision_recall_floor' in blockers
    assert 'reserved_outcome_support' not in blockers
    assert 'untouched_final_performance' in blockers


def test_market_notices_and_industry_exclude_future_filings():
    from src.validation.market_filing_signals import values_at
    rows=[{'form':'8-K','date':'2022-12-01','items':'3.01'},
          {'form':'8-K','date':'2023-01-02','items':'3.01,4.01'}]
    annual=pd.DataFrame({'filed':['2022-02-01','2023-02-01'],'sic':[2834,7372]})
    result=values_at(rows,annual,'2022-12-31')
    assert result['market_delisting_90d']==1 and result['market_auditor_change_365d']==0
    assert result['industry_division_2']==1 and result['industry_division_7']==0
    rows[0]['items']=''
    assert pd.isna(values_at(rows,annual,'2022-12-31')['market_delisting_90d'])


def test_insufficient_cash_statement_is_a_warning_not_negated_distress():
    from src.validation.contextual_filing_signals import signals
    from src.validation.market_filing_signals import corrected_liquidity
    text='ordinary operations '*220+' We do not have sufficient cash to fund the next twelve months.'
    original,_=signals(text)
    fixed=corrected_liquidity(text,original)
    assert fixed['context_liquidity_shortfall_rate']>0
    assert fixed['context_negated_distress_rate']==0


def test_audited_source_bytes_are_rechecked_not_just_the_checkpoint(tmp_path):
    from src.validation.current_production_readiness import verify_source_proofs
    path=tmp_path/'filing.html';path.write_bytes(b'original')
    proof={'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}
    assert verify_source_proofs([proof,proof])==1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source bytes changed'):
        verify_source_proofs([proof])
