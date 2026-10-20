import pandas as pd
import pytest

from src.validation import export_dataset as module


def test_export_preserves_weights_withholds_final_labels_and_detects_tampering(tmp_path, monkeypatch):
    row = {'unit_id': '1@2012', 'cik': '0000000001', 'prediction_date': '2012-12-31',
           'feature_filing_date': '2012-11-01', 'source_url': 'https://www.sec.gov/Archives/example',
           'inclusion_probability': .1, 'design_weight': 10., 'x': .2, module.TARGET: 0}
    parts = {role: pd.DataFrame([row]) for role in ['training', 'calibration', 'final']}
    monkeypatch.setattr(module, 'summary', lambda: {'production_ready': False})
    monkeypatch.setattr(module, 'read_parts', lambda: ({}, {'dataset_sha256': 'frozen'}, parts))
    monkeypatch.setattr(module, 'FEATURES', ['x'])
    output = tmp_path / 'bundle'
    result = module.export(output)
    assert result['final_test_status'] == 'already_consumed'
    assert not result['production_ready']
    assert module.TARGET in pd.read_csv(output / 'training.csv').columns
    assert module.TARGET not in pd.read_csv(output / 'final.csv').columns
    assert pd.read_csv(output / 'training.csv').design_weight.tolist() == [10.]
    assert module.export(output) == result
    (output / 'training.csv').write_text('tampered')
    with pytest.raises(ValueError, match='Export integrity failure'):
        module.export(output)
