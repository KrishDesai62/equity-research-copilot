"""Build the frozen research model's ordered input from verified public sources."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.ml.financial_history import extract_history
from src.ml.detection_models import FEATURES
from src.validation.sec_disclosure_data import FEATURES as FINANCIAL
from src.validation.sec_filing_signals import signals_at, FEATURES as FILING
from src.validation.filing_text_signals import extract_signals, visible_text
from src.validation.contextual_filing_signals import signals
from src.validation.market_filing_signals import corrected_liquidity, values_at, FEATURES as MARKET
from src.web.company_imports import normalize_filings, finite


def financial_snapshot(history, anchor):
    public = history.loc[history.filing_date.lt(anchor)].sort_values(['filing_date','accession_number'])
    if public.empty:
        raise ValueError('No reconciled financial statement was public before the requested date.')
    row = public.iloc[-1].to_dict()
    if pd.Timestamp(row['latest_input_filing_date']) > pd.Timestamp(row['filing_date']):
        raise ValueError('Financial features contain inputs published after their source filing.')
    age = (pd.Timestamp(anchor)-pd.Timestamp(row['filing_date'])).days
    assets, cash = row.get('Assets',np.nan),row.get('Cash',np.nan)
    row.update(feature_age_days=age,log_total_assets=float(np.log1p(assets)) if assets>0 else np.nan,
        accruals_to_assets=(row.get('NetIncomeLoss_ttm',np.nan)-row.get('OperatingCashFlow_ttm',np.nan))/assets if assets>0 else np.nan,
        net_debt_proxy_to_assets=(row.get('Liabilities',np.nan)-cash)/assets if assets>0 else np.nan,
        operating_cash_burn_to_cash=max(-row.get('OperatingCashFlow_ttm',np.nan),0)/cash if cash>0 else np.nan)
    return row


def metadata_rows(payload):
    block=payload.get('filings',{}).get('recent',payload)
    n=len(block.get('accessionNumber',[]))
    if any(len(block.get(k,[]))!=n for k in ['form','filingDate']):
        raise ValueError('Malformed filing metadata')
    items=block.get('items',['']*n)
    if len(items)!=n:items=['']*n
    return [dict(accession=block['accessionNumber'][i],form=block['form'][i],date=block['filingDate'][i],items=items[i] or '') for i in range(n)]


def assemble_features(history, public_filings, metadata, anchor, document, *, metadata_complete=True, annual=None):
    row=financial_snapshot(history,anchor)
    source=next((r for r in public_filings if r['accession_number']==row['accession_number']),None)
    if not source or source['filing_date']!=row['filing_date'] or source['filing_type'] not in {'10-K','10-Q'}:
        raise ValueError('The exact original financial filing could not be reconciled.')
    text_values,text_proof=extract_signals(document)
    text=visible_text(document)
    context_values,context_proof=signals(text)
    context_values.update(corrected_liquidity(text,context_values))
    event_values=signals_at(metadata,anchor)
    # A current SIC code is not substituted for missing historically published SIC.
    if annual is None:annual=pd.DataFrame(columns=['filed','sic'])
    market=values_at(metadata,annual,anchor)
    if not metadata_complete:
        for name in FILING:event_values[name]=np.nan
        for name in MARKET:
            if not name.startswith('industry_'):market[name]=np.nan
    values={**row,**event_values,**text_values,**context_values,**market}
    vector={name:finite(values.get(name)) for name in FEATURES}
    missing=[name for name,value in vector.items() if value is None]
    financial_missing=[name for name in FINANCIAL if vector[name] is None]
    reasons=[]
    if row['feature_age_days']>180:reasons.append('Financial statement is more than 180 days old.')
    if len(financial_missing)/len(FINANCIAL)>.25:reasons.append('More than 25% of financial model inputs are unavailable.')
    if len(missing)/len(FEATURES)>.25:reasons.append('More than 25% of all model inputs are unavailable.')
    return dict(ticker=row['ticker'],cik=row['cik'],as_of=anchor,
        feature_filing_date=row['filing_date'],accession_number=row['accession_number'],
        source_url=source['source_url'],ordered_features=list(FEATURES),values=vector,
        observed_features=len(FEATURES)-len(missing),missing_features=missing,
        financial_missing_features=financial_missing,metadata_complete=metadata_complete,
        eligible=not reasons,ineligible_reasons=reasons,
        text_proof=text_proof,context_proof=context_proof)


def imported_inputs(imports,ticker,anchor):
    pointer=imports.root/'companies'/f'{ticker}.json'
    if pointer.exists():
        version=imports.root/'versions'/json.loads(pointer.read_text())['version']
        manifest=json.loads((version/'manifest.json').read_text())
    else:
        company=next((c for c in imports.catalog()['companies'] if c['ticker']==ticker),None)
        if company is None:raise ValueError('Company is not covered.')
        source_manifest=Path('data/processed/research/source_manifest.json')
        if not source_manifest.exists():raise ValueError('Original SEC feature sources are not installed on this server.')
        cik=company['cik']
        sources=[s for s in json.loads(source_manifest.read_text()) if
            re.search(r'/submissions/CIK'+cik+r'(?:-submissions-\d+)?\.json$',s.get('url','')) or
            s.get('url','').endswith('/companyfacts/CIK'+cik+'.json') or
            '/Archives/edgar/data/'+str(int(cik))+'/' in s.get('url','')]
        manifest=dict(cik=cik,sources=sources)
    cik=manifest['cik'];payloads=[];proofs=[];facts=None;documents={}
    for source in manifest['sources']:
        path=Path(source['path'])
        if sha256(path.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('An imported SEC source changed after acquisition.')
        url=source.get('url','')
        if '/submissions/' in url:
            payload=json.loads(path.read_text())
            if 'cik' in payload and str(payload['cik']).zfill(10)!=cik:
                raise ValueError('Filing metadata issuer mismatch.')
            payloads.append((url,payload))
        elif '/companyfacts/' in url:
            facts=json.loads(path.read_text())
        elif '/Archives/' in url:
            documents[url]=path
        proofs.append(source)
    if facts is None or str(facts.get('cik')).zfill(10)!=cik:
        raise ValueError('Reconciled SEC company-facts data is unavailable.')
    rows={};metadata={};needed=set();loaded=set()
    start=(pd.Timestamp(anchor)-pd.Timedelta(days=365)).date().isoformat()
    for url,payload in payloads:
        loaded.add(url.rsplit('/',1)[-1])
        for archived in payload.get('filings',{}).get('files',[]):
            if archived.get('filingFrom','0000')<anchor and archived.get('filingTo','9999')>=start:
                needed.add(archived['name'])
        block=payload.get('filings',{}).get('recent',payload)
        history_start=(pd.Timestamp(anchor)-pd.DateOffset(years=5)).date().isoformat()
        for row in normalize_filings(block,cik,ticker,anchor,start=history_start):
            key=row['accession_number']
            if key in rows and rows[key]!=row:raise ValueError('Conflicting public filing metadata.')
            rows[key]=row
        for row in metadata_rows(payload):
            key=row['accession']
            if key in metadata and metadata[key]!=row:raise ValueError('Conflicting public filing metadata.')
            metadata[key]=row
    rows=list(rows.values())
    history,_,_=extract_history(facts,ticker,public_filings=rows)
    selected=financial_snapshot(history,anchor)
    filing=next(r for r in rows if r['accession_number']==selected['accession_number'])
    if filing['source_url'] not in documents:
        raise ValueError('The exact financial source report is not indexed for this date. Import a more suitable reporting period.')
    result=assemble_features(history,rows,list(metadata.values()),anchor,
        documents[filing['source_url']].read_bytes(),metadata_complete=needed<=loaded)
    result['source_proofs']=[dict(path=p['path'],sha256=p['sha256'],url=p.get('url')) for p in proofs]
    return result
