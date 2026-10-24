import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ml.detection_models import FEATURES
from src.web.model_inputs import assemble_features, financial_snapshot, imported_inputs
from src.web.research_service import ResearchService
from src.agents_full.synthesis_agent import synthesize_research
from src.web.company_imports import CompanyImports


def financial_rows():
    from src.validation.sec_disclosure_data import FEATURES as FINANCIAL
    row={key:1. for key in FINANCIAL}
    row.update(cik='0000000123',ticker='TEST',filing_date='2026-07-01',latest_input_filing_date='2026-07-01',
               accession_number='0000000123-26-000001',Assets=100.,Liabilities=60.,Cash=10.,
               NetIncomeLoss_ttm=8.,OperatingCashFlow_ttm=12.)
    return pd.DataFrame([row])


def test_feature_adapter_preserves_schema_dates_and_missing_industry():
    history=financial_rows()
    filings=[dict(accession_number='0000000123-26-000001',filing_date='2026-07-01',filing_type='10-Q',source_url='https://www.sec.gov/source')]
    metadata=[dict(form='8-K',date='2026-09-30',items='2.04',accession='prior'),dict(form='8-K',date='2026-10-01',items='2.04',accession='same-day')]
    document=('<html><body>'+('The company has cash resources and operating revenues. '*100)+'</body></html>').encode()
    result=assemble_features(history,filings,metadata,'2026-10-01',document)
    assert result['ordered_features']==FEATURES and len(result['values'])==63
    assert result['values']['feature_age_days']==92
    assert result['values']['material_default_items_90d']==1
    assert result['values']['accruals_to_assets']==pytest.approx(-.04)
    assert result['values']['net_debt_proxy_to_assets']==.5
    assert all(result['values'][f'industry_division_{i}'] is None for i in range(10))
    incomplete=assemble_features(history,filings,metadata,'2026-10-01',document,metadata_complete=False)
    assert incomplete['values']['material_default_items_90d'] is None
    assert incomplete['values']['market_delisting_90d'] is None


def test_snapshot_rejects_same_day_and_future_inputs():
    with pytest.raises(ValueError,match='No reconciled'):
        financial_snapshot(financial_rows(),'2026-07-01')
    frame=financial_rows();frame['latest_input_filing_date']='2026-07-02'
    with pytest.raises(ValueError,match='published after'):
        financial_snapshot(frame,'2026-10-01')


class Imports:
    def __init__(self,root):self.root=root
    def payload(self,ticker):return {'company':dict(ticker=ticker,cik='123',name='Test Company')}


def research(tmp_path,monkeypatch,enabled):
    service=ResearchService(Imports(tmp_path),allow_research=enabled,config=dict(protected_evaluation_dates=['2024-12-31','2025-04-01']))
    monkeypatch.setattr(service,'model_status',lambda:dict(release_status='not_approved',trained_through='2024-12-17',threshold=None))
    vector=dict.fromkeys(FEATURES,1.)
    features=dict(ticker='TEST',as_of='2026-10-01',feature_filing_date='2026-07-01',source_url='https://www.sec.gov/source',ordered_features=FEATURES,values=vector,observed_features=63,missing_features=[],financial_missing_features=[],metadata_complete=True,eligible=True,ineligible_reasons=[])
    monkeypatch.setattr('src.web.research_service.imported_inputs',lambda *args:features)
    return service,features


def test_public_api_prepares_inputs_without_scoring_unreleased_model(tmp_path,monkeypatch):
    service,features=research(tmp_path,monkeypatch,False)
    monkeypatch.setattr(service,'_load',lambda:pytest.fail('Unreleased public inference must not execute'))
    result=service.risk('TEST','2026-10-01')
    assert result['status']=='research_only' and result['score'] is None and result['alert'] is None
    assert result['inputs']['observed_features']==63


def test_opt_in_research_score_keeps_class_order_and_has_no_alert(tmp_path,monkeypatch):
    service,features=research(tmp_path,monkeypatch,True)
    class Model:
        classes_=[1,0]
        def predict_proba(self,X):
            assert list(X.columns)==FEATURES
            return np.array([[.8,.2]])
    monkeypatch.setattr(service,'_load',lambda:dict(feature_cols=FEATURES,model=Model()))
    result=service.risk('TEST','2026-10-01')
    assert result['score']==.8 and result['status']=='unvalidated_research_score'
    assert result['threshold'] is None and result['alert'] is None and not result['production_approval']


def test_protected_final_dates_and_bad_inputs_never_score(tmp_path,monkeypatch):
    service,features=research(tmp_path,monkeypatch,True)
    monkeypatch.setattr(service,'_load',lambda:pytest.fail('Must not predict'))
    assert service.risk('TEST','2024-12-31')['score'] is None
    assert service.risk('TEST','2025-04-01')['score'] is None
    features.update(eligible=False,ineligible_reasons=['Stale inputs'])
    assert service.risk('TEST','2026-10-01')['reasons']==['Stale inputs']


def test_memo_deduplicates_sources_and_refuses_wrong_issuer():
    company=dict(ticker='TEST',name='Test',cik='123')
    source=dict(text='A cited excerpt',filing_date='2026-01-01',cik='123',accession_number='accession',source_url='https://www.sec.gov/source')
    evidence=dict(citations=[source],evidence_status='available')
    result=synthesize_research(company,'2026-10-01',evidence,evidence,{})
    assert len(result['citations'])==1 and result['recommendation']=='ABSTAIN'
    source['cik']='456'
    with pytest.raises(ValueError):synthesize_research(company,'2026-10-01',evidence,evidence,{})
