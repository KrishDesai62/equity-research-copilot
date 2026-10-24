import hashlib
import json

import pytest

from src.validation.rapid_review import build, sample_units


def test_sampling_is_reproducible_and_order_independent():
    units = [str(n) for n in range(315)]
    result = sample_units(units, 60, 'fixed')
    assert result == sample_units(list(reversed(units)), 60, 'fixed')
    assert len(set(result)) == 60
    assert result != sample_units(units, 60, 'other')
    with pytest.raises(ValueError):
        sample_units(units, 0, 'fixed')


def test_missing_evidence_is_escalated_without_approval(tmp_path):
    packet = tmp_path / 'packet'
    packet.mkdir()
    task = dict(unit_id='1@2024-12-31', cik='1', company_name='<issuer>',
                prediction_date='2024-12-31', horizon_end='2025-12-31', evidence_cutoff='2026-07-19',
                automated_proposed_label=0, source_checkpoint={'path':str(tmp_path/'missing'), 'sha256':'bad'})
    data = json.dumps(task) + '\n'
    (packet/'pending_final_reviews.jsonl').write_text(data)
    (packet/'manifest.json').write_text(json.dumps({'packet_sha256':hashlib.sha256(data.encode()).hexdigest()}))
    output = tmp_path/'output'
    result = build(packet, output, sample_size=1)
    row = json.loads((output/'queue.jsonl').read_text())
    assert row['selected_for_human_review']
    assert 'checkpoint missing or hash mismatch' in row['automated_check_exceptions']
    assert result['labels_approved_by_sampling'] == 0
    assert result['production_approval'] is False
    assert '&lt;issuer&gt;' in (output/'review.html').read_text()
    (packet/'pending_final_reviews.jsonl').write_text(data + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        build(packet, output)
