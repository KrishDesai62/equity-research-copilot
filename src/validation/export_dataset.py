"""Export the verified V3 cohort without mixing tasks or recycling final labels."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from src.validation.cohort_v3 import FEATURES, TARGET, read_parts, summary
from src.validation.cohort_v3_data import ROOT, digest
from src.data_layer.research_dataset import write_json


def export(destination=Path('data/processed/datasets/bankruptcy_v3')):
    destination = Path(destination)
    # Verify the consumed final experiment, model, dataset and saved evidence.
    result = summary()
    _, manifest, parts = read_parts()
    source_hash = manifest['dataset_sha256']
    if destination.exists():
        saved = json.loads((destination / 'manifest.json').read_text())
        if saved['source_dataset_sha256'] != source_hash:
            raise ValueError('Existing export belongs to a different source dataset')
        for name, expected in saved['file_sha256'].items():
            if digest(destination / name) != expected:
                raise ValueError(f'Export integrity failure: {name}')
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix='.bankruptcy-v3-', dir=destination.parent))
    try:
        metadata = ['unit_id', 'cik', 'prediction_date', 'feature_filing_date', 'source_url',
                    'inclusion_probability', 'design_weight']
        counts = {}
        for role, frame in parts.items():
            columns = metadata + FEATURES + ([TARGET] if role != 'final' else [])
            # Final labels are not made available as training inputs. These firms
            # are still exposed, even though this export withholds their labels.
            frame[columns].to_csv(staged / f'{role}.csv', index=False)
            counts[role] = {'rows': len(frame), 'companies': int(frame.cik.nunique()),
                            'contains_target': role != 'final'}
        files = {p.name: digest(p) for p in staged.glob('*.csv')}
        catalog = {'version': 1, 'dataset': 'bankruptcy_v3', 'source_dataset_sha256': source_hash,
            'source_path': str(ROOT / 'dataset.csv'), 'features': FEATURES, 'target': TARGET,
            'partitions': counts, 'file_sha256': files, 'final_test_status': 'already_consumed',
            'production_ready': result['production_ready'],
            'usage': 'Research only. Train on training.csv; calibrate/select on calibration.csv. Never train on final.csv or treat its companies as a new holdout.',
            'weight_column': 'design_weight',
            'limits': 'Historical large-public-company source scope through 2022; sampling weights do not correct missing coverage. Unknown outcomes excluded.'}
        write_json(staged / 'manifest.json', catalog)
        staged.rename(destination)
        return catalog
    finally:
        if staged.exists():
            shutil.rmtree(staged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/processed/datasets/bankruptcy_v3'))
    print(json.dumps(export(parser.parse_args().output), indent=2))
