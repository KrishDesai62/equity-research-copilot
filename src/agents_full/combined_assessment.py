"""Connect a gated model signal to dated evidence without inventing attribution."""
import math
import re

THEMES=[
    ('Liquidity and cash runway', ['current_ratio','cash_to_assets','cashflow_to_assets','operating_cash_burn_to_cash'], r'\b(liquidity|cash|funding|working capital)\b'),
    ('Debt and repayment', ['liabilities_to_assets','net_debt_proxy_to_assets','interest_coverage'], r'\b(debt|borrowings|maturities|maturity|covenant|interest)\b'),
    ('Profitability and operating performance', ['operating_margin','ebit_to_ta','cashflow_margin'], r'\b(profit|profitability|operating income|operating loss|margin)\b'),
    ('Growth and business outlook', ['revenue_growth_yoy','sales_to_ta'], r'\b(revenue|sales|growth|competition|demand)\b'),
]


def combine(company, as_of, risk, citations):
    if risk.get('ticker',company['ticker'])!=company['ticker'] or risk.get('as_of',as_of)!=as_of:
        raise ValueError('Model and agent evidence must use the same company and date')
    inputs=risk.get('inputs') or {}
    if inputs.get('feature_filing_date','1900-01-01')>as_of:
        raise ValueError('Model source postdates research cutoff')
    score=risk.get('score');threshold=risk.get('threshold')
    approved=risk.get('production_approval') is True and risk.get('status')=='validated_score'
    valid_score=isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and 0<=score<=1
    usable=approved and valid_score
    valid_threshold=isinstance(threshold,(int,float)) and not isinstance(threshold,bool) and math.isfinite(threshold) and 0<=threshold<=1
    screening=usable and valid_threshold
    model=dict(status='released' if usable else 'withheld',score=score if usable else None,
               threshold=threshold if screening else None,
               screening_signal=('threshold_exceeded' if score>=threshold else 'below_threshold') if screening else 'unavailable',
               target=risk.get('model',{}).get('target_definition','Issuer Chapter 7/11 petition disclosed in SEC filings within one year.'),
               feature_filing_date=inputs.get('feature_filing_date'),
               observed_inputs=inputs.get('observed_features'),
               missing_inputs=len(inputs.get('missing_features',[])) if inputs else None,
               reasons=risk.get('reasons',[]))
    themes=[]
    for title,features,pattern in THEMES:
        related=[c['id'] for c in citations if re.search(pattern,c.get('text',c.get('excerpt','')),re.I)]
        observations=[{'feature':name,'value':inputs.get('values',{}).get(name)} for name in features]
        themes.append(dict(title=title,inputs=observations,source_ids=related,
                           status='inputs_and_evidence' if related and any(v['value'] is not None for v in observations) else 'partial_coverage'))
    gaps=[]
    if not screening:gaps.append('A released model score and operating threshold are required for model-based screening.')
    if not citations:gaps.append('No verified agent evidence is available for this company and cutoff.')
    if not inputs:gaps.append('Refresh SEC data to prepare dated financial inputs where available.')
    if inputs.get('missing_features'):gaps.append(f"{len(inputs['missing_features'])} model inputs are unavailable.")
    if not inputs.get('metadata_complete',True):gaps.append('Filing-history coverage is incomplete.')
    return dict(status='screening_available' if screening and citations else 'evidence_review_only',
                model=model,themes=themes,gaps=gaps,
                next_steps=['Review financing terms and repayment dates in the linked filings.',
                            'Compare growth evidence with liquidity and operating cash flow.',
                            'Review valuation, investment horizon and portfolio exposure separately.'],
                recommendation='ABSTAIN',
                explanation='Financial inputs and related passages are organized by topic. These links are not model feature attributions, and this report does not issue a buy/sell recommendation.')
