from src.validation.coverage_completion import index_rows, screen_submission, decode_uu_pdf_line
from src.validation.sampled_negative_coverage import scoped_submission_rows


def test_coverage_normalization_retains_amendments_and_foreign_reports():
    forms = ['8-K/A', '10-K/A', '10-Q/A', '6-K', '20-F', '40-F/A', '4']
    block = dict(form=forms, accessionNumber=[f'0000000123-25-{i:06d}' for i in range(7)],
                 filingDate=['2025-01-01']*7, primaryDocument=['report.htm']*7)
    rows = scoped_submission_rows(block, '0000000123', '2024-12-31', '2026-07-19')
    assert [row['filing_type'] for row in rows] == forms[:-1]


def test_index_filters_horizon_and_forms():
    text = '\n'.join(['123|A|8-K|2025-01-01|edgar/data/123/0000000123-25-000001.txt',
                      '123|A|4|2025-01-01|other.txt',
                      '123|A|10-K|2026-07-20|late.txt'])
    rows = list(index_rows(text))
    assert len(rows) == 1
    assert rows[0]['cik'] == '0000000123'
    assert rows[0]['accession'] == '0000000123-25-000001'


def test_exhibits_keep_all_matches_and_surface_unreadable_formats():
    payload = b'<DOCUMENT><TYPE>EX-99.1\n<FILENAME>press.htm\n<TEXT><p>Chapter 11 petition</p></TEXT></DOCUMENT><DOCUMENT><TYPE>EX-10\n<FILENAME>contract.pdf\n<TEXT>binary</TEXT></DOCUMENT>'
    rows = screen_submission(payload)
    assert rows[0]['keyword_matches'] == 2
    assert len(rows[0]['excerpts']) == 2
    assert rows[1]['status'] == 'unreadable_text_requires_review'


def test_primary_amendments_and_machine_readable_exhibits_are_screened():
    payload = b'<DOCUMENT><TYPE>10-K/A\n<FILENAME>amend.htm\n<TEXT><p>Annual amendment</p></TEXT></DOCUMENT><DOCUMENT><TYPE>EX-101.LAB\n<FILENAME>labels.xml\n<TEXT><root><label>Bankruptcy</label></root></TEXT></DOCUMENT>'
    rows = screen_submission(payload)
    assert rows[0]['scope'] == 'primary_report'
    assert rows[0]['keyword_matches'] == 0
    assert rows[1]['scope'] == 'exhibit'
    assert rows[1]['keyword_matches'] == 1
    assert len(rows[1]['content_sha256']) == 64


def test_pdf_encoding_honors_declared_length_with_nonzero_padding():
    assert decode_uu_pdf_line(b'$3T8-"@$!') == b'OF\r\n'
