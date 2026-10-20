"""Quarterly balance changes from reconciled original, point-in-time filings."""
import numpy as np
import pandas as pd
from src.ml.financial_history import divide

TAGS={'Assets':['Assets'],'Cash':['CashAndCashEquivalentsAtCarryingValue','Cash'],
      'Liabilities':['Liabilities'],'AssetsCurrent':['AssetsCurrent'],'LiabilitiesCurrent':['LiabilitiesCurrent'],
      'RetainedEarningsAccumulatedDeficit':['RetainedEarningsAccumulatedDeficit']}
FEATURES=['q_cash_change_to_prior_assets','q_liabilities_change_to_prior_assets','q_assets_growth',
          'q_current_ratio_change','q_cash_ratio_change','q_retained_change_to_prior_assets',
          'q_period_gap_days','q_prior_public_age_days','q_current_feature_age_days']


def balances(payload,metadata):
    gaap=payload.get('facts',{}).get('us-gaap',{});allowed={(a,r['filing_date'],r['filing_type']) for a,r in metadata.items()}
    assets=gaap.get('Assets',{}).get('units',{}).get('USD',[])
    groups={}
    for fact in assets:
        if not fact.get('start') and (fact.get('accn'),fact.get('filed'),fact.get('form')) in allowed:
            groups.setdefault((fact['accn'],fact['filed'],fact['form']),[]).append(fact)
    indexes={}
    for tags in TAGS.values():
        for tag in tags:
            index={}
            for fact in gaap.get(tag,{}).get('units',{}).get('USD',[]):
                if not fact.get('start'):index.setdefault((fact.get('accn'),fact.get('filed'),fact.get('end')),[]).append(fact.get('val'))
            indexes[tag]=index
    result=[]
    for (accn,filed,form),facts in groups.items():
        end=max(f['end'] for f in facts)
        if end>filed:continue
        record={'accession':accn,'filed':filed,'end':end}
        for field,tags in TAGS.items():
            value=np.nan
            for tag in tags:
                candidates=indexes[tag].get((accn,filed,end),[])
                unique=set(candidates)
                if len(unique)>1:break
                if len(unique)==1:
                    value=float(next(iter(unique)));break
            record[field]=value
        result.append(record)
    return result


def at_snapshot(row,history):
    result={f:np.nan for f in FEATURES}
    filed=row.get('feature_filing_date')
    if not isinstance(filed,str):filed=row.get('source_filing_date')
    anchor=row['prediction_date'];end=row['fiscal_period_end']
    if not isinstance(filed,str) or filed>=anchor or end>filed:raise ValueError('Invalid point-in-time snapshot')
    result['q_current_feature_age_days']=(pd.Timestamp(anchor)-pd.Timestamp(filed)).days
    eligible=[h for h in history if h['filed']<=filed and h['end']<end and 60<=(pd.Timestamp(end)-pd.Timestamp(h['end'])).days<=130]
    if not eligible:return result,None
    latest=max((h['end'],h['filed']) for h in eligible)
    candidates=[h for h in eligible if (h['end'],h['filed'])==latest]
    first=candidates[0]
    for other in candidates[1:]:
        if any(not (pd.isna(first[k]) and pd.isna(other[k])) and first[k]!=other[k] for k in TAGS):return result,None
    prior=first
    if not np.isfinite(prior['Assets']) or prior['Assets']<=0:return result,None
    result.update(q_cash_change_to_prior_assets=divide(row['Cash']-prior['Cash'],prior['Assets']),
       q_liabilities_change_to_prior_assets=divide(row['Liabilities']-prior['Liabilities'],prior['Assets']),
       q_assets_growth=divide(row['Assets']-prior['Assets'],prior['Assets']),
       q_current_ratio_change=divide(row['AssetsCurrent'],row['LiabilitiesCurrent'])-divide(prior['AssetsCurrent'],prior['LiabilitiesCurrent']),
       q_cash_ratio_change=divide(row['Cash'],row['Assets'])-divide(prior['Cash'],prior['Assets']),
       q_retained_change_to_prior_assets=divide(row['RetainedEarningsAccumulatedDeficit']-prior['RetainedEarningsAccumulatedDeficit'],prior['Assets']),
       q_period_gap_days=(pd.Timestamp(end)-pd.Timestamp(prior['end'])).days,
       q_prior_public_age_days=(pd.Timestamp(filed)-pd.Timestamp(prior['filed'])).days)
    return result,{'accession':prior['accession'],'filing_date':prior['filed'],'fiscal_period_end':prior['end']}
