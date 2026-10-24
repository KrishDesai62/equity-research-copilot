"""Deterministic query cleanup and topic relevance for filing excerpts."""
import re

STOP = set('a an the and or to of in on for with is are was were do does what how why about say says describe describes disclosed disclose filing filings company companies their its from by at as it this that can has have be tell me please inc corporation corp'.split())
GROUPS = {
    'risk': {'risk','risks','uncertainty','uncertainties'},
    'competition': {'competition','competitive','competitor','competitors'},
    'liquidity': {'liquidity','cash','funding','debt'},
    'bankruptcy': {'bankruptcy','bankrupt','insolvency'},
    'revenue': {'revenue','revenues','sales'},
}

def terms(query):
    return list(dict.fromkeys(t for t in re.findall(r'[a-z0-9]+',query.lower()) if t not in STOP))[:64]

def expanded_query(query):
    words=terms(query)
    for group in GROUPS.values():
        if group.intersection(words): words=list(dict.fromkeys(words+sorted(group)))
    return ' '.join(words)

def rank_evidence(rows, question):
    words=terms(question)
    # Broad risk requests should return the issuer's risk section, not accounting
    # passages that happen to contain the company name or a generic question word.
    risk_request=bool(set(words)&GROUPS['risk'])
    topical=set(words)-GROUPS['risk']
    generic_risk=risk_request and len(topical)<=1
    ranked=[]
    seen=set()
    for index,row in enumerate(rows):
        text=row.get('text',''); tokens=set(terms(text))
        section=str(row.get('section','')).lower()
        risk_section=section in {'risk_factors','risk factors','item_1a'}
        matched=sum(bool(({word}|next((g for g in GROUPS.values() if word in g),set())) & tokens) for word in words)
        if generic_risk and not risk_section: continue
        if words and not matched and not (generic_risk and risk_section): continue
        key=row.get('metadata',{}).get('chunk_id') or text
        if key in seen:continue
        seen.add(key)
        score=matched/max(len(words),1)+(2 if risk_request and risk_section else 0)
        ranked.append((score,index,row))
    return [row for _,_,row in sorted(ranked,key=lambda r:(-r[0],r[1]))]
