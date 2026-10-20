"""Conservative primary-source corroboration, separate from first-case labels."""
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import write_json
from src.validation.historical_coverage import load_reviewed_registry

INPUT = Path('data/processed/validation/public_event_checks/checkpoints')
OUTPUT = Path('data/processed/validation/outcome_adjudication')
MONTHS = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = MONTHS + r'\s+\d{1,2},?\s+20\d{2}'
STATEMENT = re.compile(r'\bon\s+(?P<date>' + DATE + r')\b(?P<subject>.{0,650}?)\b(?P<verb>filed|commenced)\s+(?P<object>(?:(?:a|the|their|its|joint|voluntary|involuntary)\s+){0,5}(?:petitions?\b|(?:chapter\s+(?:7|11|15|9)\s+)?(?:cases|proceedings)\b))', re.I)
OTHER_ACTION = re.compile(r'\b(?:entered|received|amended|agreed|approved|notified|issued|announced|would|might|expects?|intend|not)\b', re.I)


def canonical(text):
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    synonyms = {'corporation': 'corp', 'incorporated': 'inc', 'company': 'co', 'limited': 'ltd'}
    return ' '.join(synonyms.get(word, word) for word in text.split())


def parent_in_subject(subject, names):
    # Remove only an initial Petition Date parenthetical, never a substantive party.
    subject = re.sub(r'^\s*(?:\([^)]{0,100}\)\s*)?[,;:]?\s*', '', subject).strip()
    if re.match(r'^(?:the\s+)?(?:Company|Registrant)\s*(?:,|and\b|\(|$)', subject, re.I):
        return True
    if re.search(r'\b(?:together|collectively)\s+with\s+(?:the\s+)?Company\b', subject, re.I):
        return True
    normalized = ' ' + canonical(subject) + ' '
    return any(len(canonical(name)) > 5 and ' ' + canonical(name) + ' ' in normalized for name in names if name)


def corroborate(text, *, filed, names):
    text = re.sub(r'\s+', ' ', normalize_filing(text))
    matches = []
    for anchor in re.finditer(r'\bon\s+' + DATE, text, re.I):
        match = STATEMENT.match(text, anchor.start())
        if match is None:
            continue
        subject = match.group('subject')
        if OTHER_ACTION.search(subject) or re.search(DATE, subject, re.I):
            continue
        nearby = text[match.start('verb'):match.end() + 240]
        chapter = re.search(r'chapter\s+(7|11|15|9)\b', nearby, re.I)
        if chapter is None or chapter.group(1) not in {'7', '11'} or re.search(r'\binvoluntary\b', nearby, re.I):
            continue
        day = pd.Timestamp(match.group('date'))
        if not pd.Timestamp(filed) - pd.Timedelta(days=30) <= day <= pd.Timestamp(filed):
            continue
        matches.append({'petition_date': day.date().isoformat(), 'chapter': chapter.group(1),
            'issuer_explicitly_included': parent_in_subject(subject, names),
            'statement_sha256': sha256(text[match.start():match.end() + 240].encode()).hexdigest(),
            'statement_character_start': match.start(), 'statement_character_end': match.end() + 240})
    return matches


def run():
    registry = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    rows, proofs = [], []
    for marker in sorted(INPUT.glob('*.json')):
        record = json.loads(marker.read_text())
        assertions = []
        for filing in record['primary_item_103_filings']:
            source = filing['source']
            payload = Path(source['path']).read_bytes()
            if sha256(payload).hexdigest() != source['sha256']:
                raise ValueError('Primary source changed after acquisition')
            checked = corroborate(payload.decode('utf-8'), filed=filing['filing_date'],
                names=[record['publisher_company_name'], record['sec_company_name']])
            for assertion in checked:
                assertion.update(source_url=filing['source_url'], source_sha256=source['sha256'],
                    source_path=source['path'], filing_date=filing['filing_date'])
            assertions.extend(checked)
        parent_dates = sorted({(a['petition_date'], a['chapter']) for a in assertions if a['issuer_explicitly_included']})
        if len(parent_dates) == 1:
            day, chapter = parent_dates[0]
            status = 'issuer_event_source_corroborated'
        else:
            day, chapter = None, None
            status = 'ambiguous_dates' if len(parent_dates) > 1 else 'issuer_or_petition_statement_unresolved'
        prior = registry.get(record['cik'])
        rows.append({'cik': record['cik'], 'company_name': record['publisher_company_name'],
            'publisher_date': record['publisher_date'], 'petition_date': day, 'chapter': chapter,
            'status': status, 'earlier_brd_case': prior if prior and prior < (day or record['publisher_date']) else None,
            'first_case_review_complete': False, 'training_label_admitted': False, 'final_test_eligible': False})
        proofs.append({'cik': record['cik'], 'publisher_date': record['publisher_date'], 'assertions': assertions})
    frame = pd.DataFrame(rows)
    # Distinct later filings are documented as repeats; they are not silently
    # relabeled as first bankruptcies. Absence of an earlier event is not proof.
    observed = frame.dropna(subset=['petition_date']).groupby('cik').petition_date.min().to_dict()
    frame['earlier_corroborated_event'] = [observed.get(r.cik) if observed.get(r.cik) and pd.notna(r.petition_date) and observed[r.cik] < r.petition_date else None for r in frame.itertuples()]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / 'event_decisions.csv', index=False)
    write_json(OUTPUT / 'primary_statement_evidence.json', proofs)
    report = {'event_rows': len(frame), 'source_corroborated_event_rows': int(frame.status.eq('issuer_event_source_corroborated').sum()),
        'source_corroborated_companies': int(frame.loc[frame.status.eq('issuer_event_source_corroborated')].cik.nunique()),
        'earlier_brd_case_rows': int(frame.earlier_brd_case.notna().sum()),
        'later_repeat_rows': int(frame.earlier_corroborated_event.notna().sum()),
        'unresolved_rows': int(frame.status.ne('issuer_event_source_corroborated').sum()),
        'new_first_case_labels_admitted': 0, 'new_negative_labels_admitted': 0, 'production_ready': False,
        'note': 'Issuer event corroboration does not establish complete prior-case coverage or negative outcomes. All admissions remain blocked pending their separate evidence.',
        'decisions_sha256': sha256((OUTPUT / 'event_decisions.csv').read_bytes()).hexdigest(),
        'evidence_sha256': sha256((OUTPUT / 'primary_statement_evidence.json').read_bytes()).hexdigest()}
    write_json(OUTPUT / 'report.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
