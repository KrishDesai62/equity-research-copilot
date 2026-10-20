from pathlib import Path
import pytest
from src.validation.cohort_v3_data import digest
from src.validation.freeze_local_research import verify_files
from src.validation.negative_coverage_audit import documents_for


def test_frozen_evidence_detects_tampering(tmp_path):
    path=tmp_path/'evidence';path.write_text('original')
    proofs={str(path):digest(path)};verify_files(proofs)
    path.write_text('changed')
    with pytest.raises(ValueError,match='changed'):verify_files(proofs)


def test_document_inventory_rejects_mismatched_issuer(tmp_path):
    import json
    folder=tmp_path/'submissions';folder.mkdir();path=folder/'issuer.json'
    path.write_text(json.dumps({'cik':'2','filings':{'recent':{}}}))
    record={'sources':[{'path':str(path),'sha256':digest(path)}]}
    with pytest.raises(ValueError,match='issuer mismatch'):
        documents_for(record,'0000000001','2023-01-01','2024-01-01')


@pytest.mark.skipif(not Path('data/processed/validation/broad_candidate_review_20261019/report.json').is_file(), reason="Requires locally generated research evidence, excluded from Git")
def test_broader_search_candidates_never_enter_training_as_approved_labels():
    import json
    from pathlib import Path
    root=Path('data/processed/validation/broad_candidate_review_20261019')
    report=json.loads((root/'report.json').read_text())
    proposals=json.loads((root/'proposals.json').read_text())
    assert report['labels_admitted']==0 and report['independent_reviews_completed']==0
    assert report['completed_documents']==len(proposals)
    for record in proposals:
        assert record['approved_label'] is None and record['label_admitted'] is False
        assert record['model_predictions_included'] is False


@pytest.mark.skipif(not Path('data/processed/validation/reconciled_coverage_20261019/report.json').is_file(), reason="Requires locally generated research evidence, excluded from Git")
def test_reconciled_queue_excludes_reports_with_recorded_prior_reviews():
    import json
    from pathlib import Path
    root=Path('data/processed/validation/reconciled_coverage_20261019')
    report=json.loads((root/'report.json').read_text())
    queue=json.loads((root/'remaining_current_reports.json').read_text())
    assert report['checkpoint_inventory_reports']==report['reports_with_other_recorded_body_review']+len(queue)
    assert len(queue)==report['reports_without_a_body_review_in_scanned_ledgers']
