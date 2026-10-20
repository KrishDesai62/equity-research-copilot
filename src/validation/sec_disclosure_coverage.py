"""Primary-document review shared by the new historical development expansion."""
import re

from src.data_layer.filing_changes import normalize_filing
from src.validation.adjudicate_public_events import corroborate
from src.validation.download import fetch


def review_events(metadata, company, sec_name, *, earliest_anchor, raw):
    assertions, unresolved, reviews, sources = [], [], [], []
    seen = set()
    for filing in sorted(metadata, key=lambda f: (f['filing_date'], f['accession_number'])):
        if filing['filing_type'] not in {'8-K', '8-K/A'}:
            continue
        key = filing['accession_number']
        if key in seen:
            continue
        seen.add(key)
        items = set(re.findall(r'\d+\.\d{2}', filing.get('items') or ''))
        if '1.03' not in items and not (filing['filing_date'] > earliest_anchor and (not items or filing['filing_type'] == '8-K/A')):
            continue
        path = raw / 'documents' / f"{company['cik']}_{key}.html"
        proof = fetch(filing['source_url'], path)
        sources.append(proof)
        text = path.read_text(errors='replace')
        plain = normalize_filing(text)
        item103 = '1.03' in items or bool(re.search(r'\bItem\s*1\s*\.\s*03\b', plain, re.I))
        found = corroborate(text, filed=filing['filing_date'], names=[sec_name, company['company_name']]) if item103 else []
        for assertion in found:
            assertion.update(filing_date=filing['filing_date'], source_url=filing['source_url'], source_sha256=proof['sha256'])
        assertions.extend(found)
        ambiguous = (item103 and (not found or not all(a['issuer_explicitly_included'] for a in found))) or (
            not item103 and bool(re.search(r'bankrupt|chapter\s*(?:7|11)\b|receivership', plain, re.I)))
        if ambiguous:
            unresolved.append({'accession_number': key, 'filing_date': filing['filing_date']})
        reviews.append({'accession_number': key, 'filing_date': filing['filing_date'],
            'item_103': item103, 'assertions': found, 'unresolved': ambiguous, 'source_sha256': proof['sha256']})
    return assertions, unresolved, reviews, sources
