"""Combine dated bull/bear evidence into an extractive memo with source references."""
from datetime import date


def synthesize_research(company, as_of, bull, bear, risk, liquidity=None):
    date.fromisoformat(as_of)
    citations, seen = [], {}
    sections = []
    sources=[('Growth and opportunity evidence', bull), ('Risks and constraints', bear)]
    if liquidity is not None:sources.append(('Liquidity, debt and cash runway',liquidity))
    for title, result in sources:
        references = []
        for source in result.get('citations', []):
            if source['filing_date'] > as_of or str(source['cik']).zfill(10) != str(company['cik']).zfill(10):
                raise ValueError('Research source is outside the requested issuer or date')
            key = (source['accession_number'], source['text'])
            if key not in seen:
                seen[key] = len(citations)+1
                citations.append(dict(source, id=seen[key]))
            references.append(seen[key])
        sections.append(dict(title=title, source_ids=references, status=result['evidence_status']))
    from src.agents_full.combined_assessment import combine
    assessment=combine(company,as_of,risk,citations)
    return dict(combined_assessment=assessment,ticker=company['ticker'],company_name=company['name'],as_of=as_of,
                mode='extractive_research_memo',sections=sections,citations=citations,risk=risk,
                recommendation='ABSTAIN',
                limitation='These are candidate research excerpts, not validated directional theses or investment recommendations.')
