from src.agents.qa_agent import answer_question
from src.rag.query_relevance import terms


def row(text, section, chunk):
    return {'text':text,'section':section,'ticker':'AAPL','filing_date':'2025-10-31',
            'metadata':{'source_url':'https://www.sec.gov/Archives/test.htm','accession_number':'test','chunk_id':chunk,'cik':'0000320193'}}


def test_broad_risks_excludes_accounting_noise():
    rows=[row('Apple describes accounting estimates and disaggregation of expenses.','management_discussion','noise'),
          row('Competition and supply disruptions present risks to the business.','risk_factors','risk')]
    result=answer_question('What risks does Apple describe?','AAPL',as_of='2026-01-01',retriever=lambda *a,**k:rows)
    assert [c['chunk_id'] for c in result['citations']]==['risk']


def test_no_matching_risk_section_abstains():
    result=answer_question('What risks does Apple describe?','AAPL',as_of='2026-01-01',retriever=lambda *a,**k:[row('Apple accounting estimates.','management_discussion','noise')])
    assert result['status']=='unavailable'


def test_specific_topic_keeps_relevant_nonrisk_section_and_dates():
    rows=[row('Liquidity and cash requirements increased.','management_discussion','cash'),row('There is intense competition.','risk_factors','other')]
    result=answer_question('Explain liquidity','AAPL',as_of='2026-01-01',retriever=lambda *a,**k:rows)
    assert [c['chunk_id'] for c in result['citations']]==['cash']
    assert not answer_question('Explain liquidity','AAPL',as_of='2024-01-01',retriever=lambda *a,**k:rows)['citations']


def test_stopwords_are_not_search_terms():
    assert terms('What does the company describe about liquidity?')==['liquidity']


def test_relevant_text_after_long_preamble_is_not_discarded():
    preamble=' '.join('preface'+str(i) for i in range(100))
    result=answer_question('Explain cybersecurity incidents','AAPL',as_of='2026-01-01',retriever=lambda *a,**k:[row(preamble+' Cybersecurity incidents can damage reputation.','risk_factors','late-topic')])
    assert [c['chunk_id'] for c in result['citations']]==['late-topic']
