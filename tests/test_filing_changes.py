from dataclasses import replace
import json

import pytest

from src.data_layer.filing_changes import (
    Filing, compare_filings, detect_latest_changes, extract_sections, main,
    normalize_filing, render_markdown, select_filing_pair,
)


PREVIOUS = Filing("1", "10-K", "2024-02-01", "0000000001-24-000001", "annual.htm")
CURRENT = Filing("1", "10-K", "2025-02-01", "0000000001-25-000001", "annual.htm")
RISK = "Our operations depend on a limited number of suppliers in several international markets."
MDA = "Revenue increased by 10 percent during the year as demand grew across our principal markets."


def document(risk=RISK, mda=MDA, html=False):
    lines = ["Item 1A. Risk Factors", risk, "Item 1B. Unresolved Staff Comments", "None.",
             "Item 7. Management’s Discussion and Analysis", mda, "Item 7A. Market Risk", "Other disclosures."]
    return "<html><body>" + "".join(f"<p>{line}</p>" for line in lines) + "</body></html>" if html else "\n".join(lines)


def test_added_removed_and_modified_disclosures_have_citations():
    before = document(RISK + " We face currency exposure. Our warehouse lease will expire.")
    after = document(RISK + " We face higher currency exposure. We breached a debt covenant.")
    report = compare_filings(PREVIOUS, CURRENT, before, after)
    assert report["summary"]["changed_sections"] == 1
    risk, mda = report["sections"]
    assert risk["status"] == "changed"
    assert mda["status"] == "unchanged"
    assert "debt covenant" in risk["changes"][0]["current_text"]
    assert "warehouse lease" in risk["changes"][0]["previous_text"]
    assert risk["current_source"] == CURRENT.url
    assert report["current"]["sha256"] != report["previous"]["sha256"]
    assert "debt covenant" in render_markdown(report)
    json.dumps(report)


def test_formatting_and_hidden_xbrl_do_not_trigger_change():
    html = document(html=True).replace("Our operations", "Our    operations")
    html = html.replace("<body>", '<body><ix:hidden><p>Secret data</p></ix:hidden><script>fake</script><div style="display: none">hidden</div>')
    report = compare_filings(PREVIOUS, CURRENT, document(), html)
    assert report["summary"]["changed_sections"] == 0
    assert report["summary"]["unavailable_sections"] == 0
    assert "Secret data" not in normalize_filing(html)


@pytest.mark.parametrize("old_risk,new_risk,kind", [
    (RISK, RISK + " We may be unable to refinance our outstanding debt.", "added"),
    (RISK + " We may be unable to refinance our outstanding debt.", RISK, "removed"),
])
def test_standalone_added_and_removed_sentences(old_risk, new_risk, kind):
    report = compare_filings(PREVIOUS, CURRENT, document(old_risk), document(new_risk))
    change = report["sections"][0]["changes"][0]
    assert change["kind"] == kind
    assert bool(change["current_text"]) == (kind == "added")
    assert bool(change["previous_text"]) == (kind == "removed")


def test_missing_section_is_unavailable_not_removed():
    report = compare_filings(PREVIOUS, CURRENT, document(), "We could not parse this filing")
    assert report["summary"]["unavailable_sections"] == 2
    assert report["summary"]["change_blocks"] == 0


def test_missing_end_boundary_is_unavailable():
    assert not extract_sections("Item 1A. Risk Factors\n" + RISK, "10-K")


def test_table_of_contents_is_not_selected_over_body():
    toc = "Item 1A. Risk Factors 10\nItem 1B. Staff Comments 20\nItem 7. Management's Discussion 30\nItem 7A. Market Risk 40\n"
    sections = extract_sections(toc + document(), "10-K")
    assert RISK in sections["risk_factors"]
    assert MDA in sections["management_discussion"]


def test_item_title_split_over_html_blocks():
    html = document(html=True).replace("Item 1A. Risk Factors", "Item 1A.</p><p>Risk Factors")
    assert RISK in extract_sections(html, "10-K")["risk_factors"]


def test_quarterly_sections_and_numeric_changes():
    before = document().replace("Item 7.", "Item 2.").replace("Item 7A.", "Item 3.")
    after = before.replace("10 percent", "20 percent")
    report = compare_filings(replace(PREVIOUS, filing_type="10-Q"), replace(CURRENT, filing_type="10-Q"), before, after)
    assert report["sections"][1]["status"] == "changed"


@pytest.mark.parametrize("previous,current", [
    (PREVIOUS, replace(CURRENT, cik="2")),
    (PREVIOUS, replace(CURRENT, filing_type="10-Q")),
    (CURRENT, PREVIOUS), (CURRENT, CURRENT),
])
def test_reject_incomparable_pairs(previous, current):
    with pytest.raises(ValueError):
        compare_filings(previous, current, document(), document())


def record(filing):
    return {key: getattr(filing, key) for key in
            ("filing_type", "filing_date", "accession_number", "primary_document")}


def test_select_pair_filters_future_filings_and_amendments():
    future = replace(CURRENT, filing_date="2026-02-01", accession_number="0000000001-26-000001")
    amendment = {**record(CURRENT), "filing_type": "10-K/A"}
    filings = [record(future), amendment, record(CURRENT), record(PREVIOUS), record(CURRENT)]
    assert select_filing_pair("1", filings, as_of="2025-03-01") == (PREVIOUS, CURRENT)
    with pytest.raises(ValueError, match="At least two"):
        select_filing_pair("1", filings, as_of="2024-03-01")


def test_same_day_pair_requires_manual_review():
    other = replace(CURRENT, accession_number="0000000001-25-000002")
    with pytest.raises(ValueError, match="same-day"):
        select_filing_pair("1", [record(CURRENT), record(other)])


def test_invalid_metadata_and_amendments_are_rejected():
    for fields in ({"cik": "../1"}, {"primary_document": "../annual.htm"},
                   {"accession_number": "bad"}, {"filing_type": "10-K/A"}):
        with pytest.raises(ValueError):
            replace(CURRENT, **fields)


def test_downloads_only_selected_pair(monkeypatch):
    from src.data_layer import edgar_ingest
    monkeypatch.setattr(edgar_ingest, "list_filings", lambda *args: [record(CURRENT), record(PREVIOUS)])
    downloads = []
    def download(cik, accession, primary_document):
        downloads.append(accession)
        return document()
    monkeypatch.setattr(edgar_ingest, "download_filing", download)
    report = detect_latest_changes("1")
    assert downloads == [PREVIOUS.accession_number, CURRENT.accession_number]
    assert report["summary"]["changed_sections"] == 0


def test_offline_cli_writes_json_and_markdown(tmp_path, monkeypatch):
    old, new = tmp_path / "old.html", tmp_path / "new.html"
    old.write_text(document(html=True))
    new.write_text(document(RISK + " Our financing arrangements may limit future borrowing.", html=True))
    output = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["filing_changes", "--cik", "1", "--previous-file", str(old),
        "--current-file", str(new), "--previous-date", PREVIOUS.filing_date,
        "--current-date", CURRENT.filing_date, "--previous-accession", PREVIOUS.accession_number,
        "--current-accession", CURRENT.accession_number, "--previous-document", "annual.htm",
        "--current-document", "annual.htm", "--output", str(output)])
    main()
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["summary"]["changed_sections"] == 1
    assert "financing arrangements" in output.with_suffix(".md").read_text()
