"""Primary SEC event discovery across fixed historical training windows."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.local_workflow import local_only
from src.validation import sec_event_inventory as search

CONFIG = Path('config/development_20261019/historical_inventory.json')
ROOT = Path('data/processed/validation/historical_sec_inventory_20261019')
RAW = Path('data/raw/historical_sec_inventory_20261019')


def run():
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'report.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved['config_sha256'] != digest(CONFIG) or saved['implementation_sha256'] != digest(__file__) or saved['candidates_sha256'] != digest(ROOT / 'candidates.csv'):
            raise ValueError('Frozen historical discovery changed')
        for source in saved['sources']:
            if digest(source['path']) != source['sha256']:
                raise ValueError('Frozen historical discovery bytes changed')
        return saved
    rows, sources, windows = [], [], []
    for year in config['years']:
        query = {**config, 'start': f'{year}-01-01', 'end': min(f'{year}-12-31', config['last_date'])}
        print('Historical SEC discovery', year, flush=True)
        with patch.object(search, 'RAW', RAW / str(year)):
            payload, proof = search.get_page(query, 0)
            total = search.validate_page(payload, query, 0)
            hits = list(payload['hits']['hits'])
            sources.append(proof)
            for offset in range(config['page_size'], total, config['page_size']):
                payload, proof = search.get_page(query, offset)
                if search.validate_page(payload, query, offset) != total:
                    raise ValueError('SEC result count changed mid-window')
                hits.extend(payload['hits']['hits'])
                sources.append(proof)
        if len(hits) != total or len({h['_id'] for h in hits}) != total:
            raise ValueError('Incomplete/duplicate historical search pages')
        rows.extend(search.candidates_from_hits(hits, query))
        windows.append({'year': year, 'hits': total})
    frame = pd.DataFrame(rows).drop_duplicates(['cik', 'accession_number'])
    frame.to_csv(ROOT / 'candidates.csv', index=False)
    result = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'search_helper_sha256': digest(search.__file__), 'candidates_sha256': digest(ROOT / 'candidates.csv'),
        'windows': windows, 'filing_company_pairs': len(frame), 'candidate_companies': int(frame.cik.nunique()),
        'sources': sources, 'outcomes_admitted': 0, 'final_predictions_created': False}
    write_json(report_path, result)
    return result


if __name__ == '__main__':
    with local_only():
        print(json.dumps(run(), indent=2))
