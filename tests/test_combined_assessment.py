import pytest
from src.agents_full.combined_assessment import combine

COMPANY={'ticker':'TEST','cik':'123','name':'Test'}
SOURCES=[{'id':1,'text':'Liquidity depends on cash flow and financing.'},{'id':2,'text':'Debt maturities require repayment.'}]

def profile():
    return {'ticker':'TEST','as_of':'2026-10-24','status':'research_only','production_approval':False,'score':.9,'threshold':.5,
            'inputs':{'feature_filing_date':'2026-08-01','values':{'current_ratio':1.2},'observed_features':52,'missing_features':['industry'],'metadata_complete':True}}

def test_unreleased_score_never_becomes_combined_decision():
    result=combine(COMPANY,'2026-10-24',profile(),SOURCES)
    assert result['model']['score'] is None
    assert result['model']['screening_signal']=='unavailable'
    assert result['recommendation']=='ABSTAIN'
    assert result['themes'][0]['source_ids']==[1]
    assert result['themes'][1]['source_ids']==[2]
    assert result['themes'][0]['inputs'][0]['value']==1.2

def test_released_signal_requires_valid_threshold_and_still_abstains_on_trades():
    risk=profile();risk.update(status='validated_score',production_approval=True)
    result=combine(COMPANY,'2026-10-24',risk,SOURCES)
    assert result['model']['screening_signal']=='threshold_exceeded'
    assert result['status']=='screening_available'
    assert result['recommendation']=='ABSTAIN'
    risk['threshold']=None
    assert combine(COMPANY,'2026-10-24',risk,SOURCES)['model']['screening_signal']=='unavailable'

def test_context_cannot_mix_issuers_or_dates():
    for patch in [{'ticker':'OTHER'},{'as_of':'2026-10-23'},{'inputs':{'feature_filing_date':'2026-10-25'}}]:
        with pytest.raises(ValueError):combine(COMPANY,'2026-10-24',dict(profile(),**patch),SOURCES)
