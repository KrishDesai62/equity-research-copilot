"""Snapshot public SEC Item 1.03 search with pagination and source proofs.

Search results identify candidates, not bankruptcy labels or negative coverage.
"""
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode

import pandas as pd
import requests

from src.data_layer.edgar_ingest import limiter, sec_headers
from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.local_workflow import local_only

CONFIG = Path('config/sec_disclosure_expansion/discovery_v1.json')
RAW = Path('data/raw/sec_event_inventory_v1')
ROOT = Path('data/processed/validation/sec_event_inventory_v1')


def get_page(config, offset):
    params = {'forms': config['forms'], 'items': config['item'], 'startdt': config['start'],
        'enddt': config['end'], 'dateRange': 'custom', 'from': offset, 'size': config['page_size']}
    url = config['endpoint'] + '?' + urlencode(params)
    path = RAW / f'page_{offset:06d}.json'
    if not path.exists():
        for attempt in range(3):
            try:
                limiter.wait()
                response = requests.get(url, headers=sec_headers(), timeout=(10, 45))
                response.raise_for_status()
                payload = response.json()
                write_json(path, payload)
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return json.loads(path.read_text()), {'url': url, 'path': str(path), 'sha256': digest(path)}


def validate_page(payload, config, offset):
    if payload.get('timed_out') or payload.get('_shards', {}).get('failed', 0):
        raise ValueError('Incomplete SEC search response')
    total = payload['hits']['total']
    if total['relation'] != 'eq':
        raise ValueError('Truncated SEC search; narrower predefined date slices required')
    query = payload['query']
    if query['from'] != offset or query['size'] != config['page_size']:
        raise ValueError('SEC ignored pagination request')
    filters = query['query']['bool']['filter']
    if {'terms': {'items': [config['item']]}} not in filters or {'terms': {'root_forms': ['8-K']}} not in filters:
        raise ValueError('SEC ignored item/root-form filters')
    expected_range = {'range': {'file_date': {'gte': config['start'], 'lte': config['end']}}}
    if expected_range not in filters:
        raise ValueError('SEC ignored date filters')
    return int(total['value'])


def candidates_from_hits(hits, config):
    rows = []
    for hit in hits:
        source = hit['_source']
        accession = source['adsh']
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession) or not hit['_id'].startswith(accession + ':'):
            raise ValueError('Invalid SEC accession')
        if source['form'] not in {'8-K', '8-K/A'} or config['item'] not in source['items']:
            raise ValueError('Unrelated filing returned by SEC item search')
        if not config['start'] <= source['file_date'] <= config['end']:
            raise ValueError('SEC returned filing outside discovery window')
        for cik in source['ciks']:
            if not re.fullmatch(r'\d{1,10}', cik):
                raise ValueError('Invalid SEC CIK')
            rows.append({'cik': cik.zfill(10), 'accession_number': accession,
                'filing_date': source['file_date'], 'form': source['form'],
                'display_names': '; '.join(source['display_names']), 'search_hit_id': hit['_id'],
                'label_admitted': False})
    return rows


def run():
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'report.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved['config_sha256'] != digest(CONFIG) or saved['implementation_sha256'] != digest(__file__) or saved['candidates_sha256'] != digest(ROOT / 'candidates.csv'):
            raise ValueError('Frozen SEC discovery changed')
        for source in saved['sources']:
            if digest(source['path']) != source['sha256']:
                raise ValueError('Frozen SEC discovery source changed')
        return saved
    first, proof = get_page(config, 0)
    total = validate_page(first, config, 0)
    hits, sources = list(first['hits']['hits']), [proof]
    for offset in range(config['page_size'], total, config['page_size']):
        payload, proof = get_page(config, offset)
        if validate_page(payload, config, offset) != total:
            raise ValueError('SEC inventory changed while paginating')
        hits.extend(payload['hits']['hits'])
        sources.append(proof)
    if len(hits) != total or len({h['_id'] for h in hits}) != total:
        raise ValueError('Missing/duplicate SEC search hits')
    rows = candidates_from_hits(hits, config)
    frame = pd.DataFrame(rows, columns=['cik', 'accession_number', 'filing_date', 'form', 'display_names', 'search_hit_id', 'label_admitted'])
    frame.to_csv(ROOT / 'candidates.csv', index=False)
    report = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'search_hits': total, 'unique_company_candidates': int(frame.cik.nunique()), 'sources': sources,
        'candidates_sha256': digest(ROOT / 'candidates.csv'), 'outcome_labels_admitted': 0,
        'limits': config['limits']}
    write_json(report_path, report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(run(), indent=2))
