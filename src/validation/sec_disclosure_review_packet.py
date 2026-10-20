"""Export evidence for actual independent review, without inventing approvals."""
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json, write_jsonl
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_admission_v3 import ROOT as ADMISSION, run as verify_admission
from src.validation.sec_disclosure_data import TARGET

ROOT = Path('data/processed/validation/sec_disclosure_review_v3')


def run():
    admission = verify_admission()
    frame = pd.read_csv(ADMISSION / 'final.csv', dtype={'cik': str})
    proofs = json.loads((ADMISSION / 'evidence.json').read_text())
    v2 = json.loads(Path('data/processed/validation/sec_disclosure_admission_v2/report.json').read_text())
    rows, positive_index = [], []
    for row in frame.loc[frame[TARGET].notna()].to_dict('records'):
        assertions = []
        for proof in proofs:
            if proof['cik'] != row['cik']:
                continue
            for assertion in proof['assertions']:
                if assertion['issuer_explicitly_included'] and row['prediction_date'] < assertion['petition_date'] <= row['horizon_end'] and assertion['filing_date'] <= row['label_available_date']:
                    assertions.append({'source': proof['source'], 'assertion': assertion})
        label = int(row[TARGET])
        if label == 1 and not assertions:
            raise ValueError('Positive review unit lacks primary issuer proof')
        task = {'unit_id': row['unit_id'], 'cik': row['cik'], 'company_name': row.get('company_name'),
            'prediction_date': row['prediction_date'], 'horizon_end': row['horizon_end'],
            'evidence_cutoff': row['label_available_date'], 'automated_proposed_label': label,
            'target': TARGET, 'scope': 'SEC-disclosed issuer Chapter 7/11 petition, not exhaustive court events',
            'positive_statement_evidence': assertions, 'source_checkpoint': v2['checkpoint_inputs'][row['cik']],
            'review_status': 'pending_independent_review', 'reviewers': [], 'approved_label': None,
            'model_predictions_included': False}
        rows.append(task)
        if label:
            positive_index.append({'cik': row['cik'], 'company_name': row.get('company_name'),
                'petition_dates': ';'.join(sorted({a['assertion']['petition_date'] for a in assertions})),
                'primary_source_urls': ';'.join(sorted({a['source']['url'] for a in assertions}))})
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'pending_final_reviews.jsonl'
    # Preserve completed reviews: review responses belong in a separate file;
    # regenerate this deterministic evidence packet only if its source matches.
    expected = {'admission_report_sha256': digest(ADMISSION / 'report.json'),
        'implementation_sha256': digest(__file__)}
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved[k] != v for k, v in expected.items()) or saved['packet_sha256'] != digest(output) or saved['positive_index_sha256'] != digest(ROOT / 'positive_sources.csv'):
            raise ValueError('Frozen review packet changed; review responses must be separate')
        return saved
    write_jsonl(output, rows)
    pd.DataFrame(positive_index).to_csv(ROOT / 'positive_sources.csv', index=False)
    result = {**expected, 'review_units': len(rows), 'positive_units': len(positive_index),
        'negative_units': len(rows) - len(positive_index), 'packet_sha256': digest(output),
        'positive_index_sha256': digest(ROOT / 'positive_sources.csv'), 'independent_reviews_completed': 0,
        'unknown_units_not_admitted': admission['counts']['final']['unknown_rows'],
        'production_approval': False, 'model_predictions_included': False}
    write_json(manifest_path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
