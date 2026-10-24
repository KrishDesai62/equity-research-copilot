import pytest
from src.validation.sampled_negative_coverage import scan


def test_screen_flags_context_without_approving_label():
    result=scan(('<html><body>'+('ordinary report text '*20)+' Our customer filed a chapter 11 petition.</body></html>').encode())
    assert result['keyword_matches'] >= 2
    assert 'customer' in result['excerpts'][0]
    assert result['automated_negative_approval'] is False


def test_screen_rejects_block_page():
    with pytest.raises(ValueError):
        scan(b'<html>Access denied</html>')


def test_inventory_reports_missing_declared_history(tmp_path):
    import json
    from src.validation.cohort_v3_data import digest
    from src.validation.sampled_negative_coverage import inventory
    source=tmp_path/'submissions'/'CIK0000000001.json'
    source.parent.mkdir()
    source.write_text(json.dumps({'cik':1,'filings':{'recent':{'accessionNumber':[],'form':[]},'files':[{'name':'older.json','filingFrom':'2025-01-01','filingTo':'2025-12-31'}]}}))
    checkpoint=tmp_path/'checkpoint.json'
    checkpoint.write_text(json.dumps({'sources':[{'path':str(source),'sha256':digest(source)}],'reporting_followup_complete':True}))
    task={'cik':'0000000001','prediction_date':'2024-12-31','evidence_cutoff':'2026-07-19','source_checkpoint':{'path':str(checkpoint),'sha256':digest(checkpoint)}}
    rows,missing,followup=inventory(task)
    assert rows==[]
    assert missing==['older.json']
    assert followup is True
    source.write_text('{}')
    with pytest.raises(ValueError,match='Submission hash mismatch'):
        inventory(task)
