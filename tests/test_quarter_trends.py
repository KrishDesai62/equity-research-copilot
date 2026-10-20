import numpy as np
from src.ml.quarter_trends import at_snapshot


def snapshot():
    return {'feature_filing_date':'2023-05-15','prediction_date':'2023-05-31','fiscal_period_end':'2023-03-31',
       'Cash':10.,'Assets':100.,'Liabilities':80.,'AssetsCurrent':50.,'LiabilitiesCurrent':40.,'RetainedEarningsAccumulatedDeficit':-20.}


def prior(**changes):
    return {**snapshot(),'filed':'2023-03-01','end':'2022-12-31','accession':'old','Cash':20.,**changes}


def test_future_filings_cannot_change_earlier_trends():
    row=snapshot();a,proof=at_snapshot(row,[prior()]);b,other=at_snapshot(row,[prior(),prior(filed='2023-06-01',Cash=999.)])
    assert a==b and proof==other
    assert a['q_cash_change_to_prior_assets']==-.1


def test_missing_or_ambiguous_history_remains_unknown():
    a,p=at_snapshot(snapshot(),[])
    assert np.isnan(a['q_cash_change_to_prior_assets']) and p is None
    a,p=at_snapshot(snapshot(),[prior(),prior(Cash=999.)])
    assert np.isnan(a['q_cash_change_to_prior_assets']) and p is None


def test_unreconciled_accession_cannot_supply_prior_balance():
    from src.ml.quarter_trends import balances
    fact={'accn':'original','filed':'2023-03-01','form':'10-K','end':'2022-12-31','val':100.}
    payload={'facts':{'us-gaap':{'Assets':{'units':{'USD':[fact,{**fact,'accn':'amended','val':999.}]}}}}}
    metadata={'original':{'filing_date':'2023-03-01','filing_type':'10-K'}}
    result=balances(payload,metadata)
    assert len(result)==1 and result[0]['Assets']==100.


def test_invalid_future_snapshot_is_rejected():
    import pytest
    row={**snapshot(),'feature_filing_date':'2023-06-01'}
    with pytest.raises(ValueError,match='point-in-time'):at_snapshot(row,[prior()])
