"""Versioned, uniform primary-evidence corrections; never invents negative labels."""
import json
from pathlib import Path
import re

import pandas as pd

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import write_json
from src.validation.adjudicate_public_events import DATE, OTHER_ACTION, parent_in_subject
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import amendments, TARGET
from src.validation.sec_disclosure_repair import outcome_submission_rows

CONFIG = Path('config/sec_disclosure_expansion/admission_v2.json')
ROOT = Path('data/processed/validation/sec_disclosure_admission_v2')
STATEMENT = re.compile(r'\bon\s+(?P<date>' + DATE + r')\b(?P<subject>.{0,1500}?)\b(?P<verb>filed|commenced)\s+(?P<object>(?:(?:a|the|their|its|joint|voluntary|involuntary)\s+){0,5}(?:petitions?\b|(?:chapter\s+(?:7|11|15|9)\s+)?(?:cases?|proceedings?)\b))', re.I)


def corroborate_v2(text, *, filed, names, maximum_lag=200):
    text = re.sub(r'\s+', ' ', normalize_filing(text))
    result = []
    for anchor in re.finditer(r'\bon\s+' + DATE, text, re.I):
        match = STATEMENT.match(text, anchor.start())
        if match is None or OTHER_ACTION.search(match.group('subject')) or re.search(DATE, match.group('subject'), re.I):
            continue
        nearby = text[match.start('verb'):match.end() + 240]
        chapter = re.search(r'chapter\s+(7|11|15|9)\b', nearby, re.I)
        if not chapter or chapter.group(1) not in {'7', '11'} or re.search(r'\binvoluntary\b', nearby, re.I):
            continue
        day = pd.Timestamp(match.group('date'))
        if not pd.Timestamp(filed) - pd.Timedelta(days=maximum_lag) <= day <= pd.Timestamp(filed):
            continue
        end = min(len(text), match.end() + 240)
        from hashlib import sha256
        result.append({'petition_date': day.date().isoformat(), 'chapter': chapter.group(1),
            'issuer_explicitly_included': parent_in_subject(match.group('subject'), names),
            'statement_start': match.start(), 'statement_end': end,
            'statement_sha256': sha256(text[match.start():end].encode()).hexdigest()})
    return result


def company_proofs(cik, checkpoint, maximum_lag):
    record = json.loads(checkpoint.read_text())
    if record['status'] != 'complete':
        return [], []
    names, metadata = [], {}
    for source in record['sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError('Primary input changed')
        if '/submissions/' not in source['path']:
            continue
        payload = json.loads(Path(source['path']).read_text())
        if 'name' in payload:
            names.append(payload['name'])
            names.extend(n['name'] for n in payload.get('formerNames', []))
        block = payload.get('filings', {}).get('recent', payload)
        for filing in outcome_submission_rows(block, cik, 'CIK' + cik, '2010-01-01', '2026-07-19'):
            metadata[filing['accession_number']] = filing
    company = {'cik': cik, 'ticker': 'CIK' + cik}
    for filing in amendments(record['sources'], company, '2010-01-01', '2026-07-19'):
        metadata[filing['accession_number']] = filing
    if 'row' in record:
        names.append(record['row']['company_name'])
    elif record.get('rows'):
        names.extend(r.get('company_name', '') for r in record['rows'])
    events, proofs = [], []
    for source in record['sources']:
        matched = re.search(r'_(\d{10}-\d{2}-\d{6})\.html$', source['path'])
        if matched is None or matched.group(1) not in metadata:
            continue
        filing = metadata[matched.group(1)]
        text = Path(source['path']).read_text(errors='replace')
        if filing['filing_type'] not in {'8-K', '8-K/A'} or ('1.03' not in (filing.get('items') or '') and not re.search(r'\bItem\s*1\s*\.\s*03\b', normalize_filing(text), re.I)):
            continue
        found = corroborate_v2(text, filed=filing['filing_date'], names=names, maximum_lag=maximum_lag)
        for event in found:
            event.update(filing_date=filing['filing_date'], source_sha256=source['sha256'], source_url=source['url'])
        events.extend(found)
        if found:
            proofs.append({'cik': cik, 'source': source, 'assertions': found})
    return events, proofs


def apply_evidence(frame, events):
    result = frame.copy()
    changes = []
    for index, row in result.iterrows():
        cutoff = row['label_available_date']
        eligible = [e for e in events.get(row.cik, []) if e['issuer_explicitly_included'] and e['filing_date'] <= cutoff]
        before = [e for e in eligible if e['petition_date'] <= row.prediction_date]
        positives = [e for e in eligible if row.prediction_date < e['petition_date'] <= row.horizon_end]
        original = row[TARGET]
        label = original
        reason = None
        if before:
            label, reason = None, 'primary_prior_event_censored'
        elif positives:
            if pd.notna(original) and original == 0:
                label, reason = None, 'contradictory_negative_requires_review'
            elif pd.isna(original):
                label, reason = 1, 'explicit_primary_statement_v2'
        if reason and (pd.isna(original) != pd.isna(label) or (pd.notna(original) and pd.notna(label) and original != label)):
            result.at[index, TARGET] = label
            result.at[index, 'label_reason'] = reason
            changes.append({'unit_id': row.unit_id, 'cik': row.cik,
                'old_label': int(original) if pd.notna(original) else None,
                'new_label': int(label) if pd.notna(label) else None, 'reason': reason})
    return result, changes


def run():
    config = json.loads(CONFIG.read_text())
    paths = {'training': Path('data/processed/validation/sec_disclosure_training_v2/dataset.csv'),
        'calibration': Path('data/processed/validation/sec_disclosure_v1/dataset.csv'),
        'final': Path('data/processed/validation/sec_disclosure_v2/combined_final_dataset.csv')}
    ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'report.json'
    source_hashes = {k: digest(p) for k, p in paths.items()}
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved['source_dataset_hashes'] != source_hashes or saved['implementation_sha256'] != digest(__file__) or saved['config_sha256'] != digest(CONFIG):
            raise ValueError('Frozen admission source/config changed')
        for role, expected in saved['output_hashes'].items():
            if digest(ROOT / (role + '.csv')) != expected:
                raise ValueError('Frozen admission output changed')
        if digest(ROOT / 'evidence.json') != saved['evidence_sha256']:
            raise ValueError('Frozen admission proof file changed')
        for checkpoint in saved['checkpoint_inputs'].values():
            if digest(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('Frozen admission checkpoint changed')
        for proof in json.loads((ROOT / 'evidence.json').read_text()):
            if digest(proof['source']['path']) != proof['source']['sha256']:
                raise ValueError('Frozen admission evidence changed')
        return saved
    frames = {k: pd.read_csv(p, dtype={'cik': str}) for k, p in paths.items()}
    frames['calibration'] = frames['calibration'].loc[frames['calibration'].role.eq('calibration')].copy()
    markers = {}
    for directory in ['sec_disclosure_training_v2', 'sec_disclosure_v1', 'sec_disclosure_v2']:
        for marker in (Path('data/processed/validation') / directory / 'checkpoints').glob('*.json'):
            markers[marker.stem] = marker
    events, proofs, checkpoints = {}, [], {}
    for cik in sorted(set().union(*(set(f.cik) for f in frames.values()))):
        marker = markers[cik]
        events[cik], evidence = company_proofs(cik, marker, config['maximum_statement_lag_days'])
        proofs.extend(evidence)
        checkpoints[cik] = {'path': str(marker), 'sha256': digest(marker)}
    changes, counts, output_hashes = {}, {}, {}
    for role, frame in frames.items():
        updated, changes[role] = apply_evidence(frame, events)
        updated.to_csv(ROOT / (role + '.csv'), index=False)
        output_hashes[role] = digest(ROOT / (role + '.csv'))
        counts[role] = {'rows': len(updated), 'positive_companies': int(updated.loc[updated[TARGET].eq(1), 'cik'].nunique()),
            'positive_rows': int(updated[TARGET].eq(1).sum()), 'negative_rows': int(updated[TARGET].eq(0).sum()),
            'unknown_rows': int(updated[TARGET].isna().sum())}
    write_json(ROOT / 'evidence.json', proofs)
    report = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'source_dataset_hashes': source_hashes, 'output_hashes': output_hashes,
        'evidence_sha256': digest(ROOT / 'evidence.json'), 'checkpoint_inputs': checkpoints,
        'counts': counts, 'changes': changes, 'new_negative_labels': 0,
        'independent_human_review': False, 'final_predictions_created': False, 'production_ready': False}
    write_json(report_path, report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
