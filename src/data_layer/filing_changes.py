"""Auditable disclosure diffs. No LLM, embeddings, or trained model required.

Compare like forms: annual against annual, quarterly against quarterly. A textual
change is a review signal, not a determination of financial materiality.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import date
from difflib import SequenceMatcher, unified_diff
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import quote


@dataclass(frozen=True)
class Filing:
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    primary_document: str

    def __post_init__(self):
        cik = str(self.cik)
        if not cik.isdigit() or len(cik) > 10:
            raise ValueError("CIK must contain at most 10 digits")
        object.__setattr__(self, "cik", cik.zfill(10))
        if self.filing_type not in {"10-K", "10-Q"}:
            raise ValueError("Only original 10-K and 10-Q filings are supported; amendments need separate review")
        date.fromisoformat(self.filing_date)
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", self.accession_number):
            raise ValueError("Invalid SEC accession number")
        if not self.primary_document or any(c in self.primary_document for c in ("/", "\\")) or ".." in self.primary_document:
            raise ValueError("primary_document must be a document filename")

    @property
    def url(self):
        accession = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{accession}/{quote(self.primary_document)}"


class _VisibleText(HTMLParser):
    """Preserve block boundaries; exclude scripts, hidden XBRL and hidden nodes."""
    blocks = {"p", "div", "br", "li", "tr", "table", "h1", "h2", "h3", "h4", "hr"}
    voids = {"br", "hr", "img", "input", "meta", "link", "wbr", "area", "base", "embed", "source", "col"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        style = re.sub(r"\s+", "", attributes.get("style", "")).lower()
        hidden = (any(value for _, value in self.stack) or
                  tag in {"script", "style", "head", "ix:hidden"} or
                  "hidden" in attributes or "display:none" in style or "visibility:hidden" in style)
        if tag not in self.voids:
            self.stack.append((tag, hidden))
        if not hidden and tag in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        hidden = any(value for _, value in self.stack)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if not hidden and tag in self.blocks:
            self.parts.append("\n")

    def handle_data(self, data):
        if not any(value for _, value in self.stack):
            self.parts.append(data)


def normalize_filing(text):
    """Ignore layout whitespace and common typographic differences, retain numbers."""
    if re.search(r"<(?:html|div|p|table|body|ix:|span)\b", text, re.I):
        parser = _VisibleText()
        parser.feed(text)
        text = "".join(parser.parts)
    text = text.translate(str.maketrans({"\u00a0": " ", "\u200b": "", "\u00ad": "",
                                        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}))
    return "\n".join(line for raw in text.splitlines() if (line := " ".join(raw.split())))


_HEADINGS = re.compile(r"(?im)^\s*(?:PART\s+(?:I|II|III|IV)\s+)?ITEM\s+(\d{1,2}[A-C]?)\s*[.\-:\u2014]?\s*(.*)$")


def extract_sections(text, filing_type):
    """Extract bounded Risk Factors and MD&A sections, preferring body over TOC.

    Missing boundaries are reported as unavailable rather than taking the rest
    of the document and inventing additions/removals.
    """
    text = normalize_filing(text)
    headings = list(_HEADINGS.finditer(text))
    targets = {"risk_factors": ("1A", r"risk\s+factors"),
               "management_discussion": ("7" if filing_type == "10-K" else "2", r"management.{0,6}s?\s+discussion")}
    result = {}
    for section, (item, title) in targets.items():
        candidates = []
        for i, heading in enumerate(headings):
            if heading.group(1).upper() != item:
                continue
            # SEC headings can split the item number and section title over lines.
            header_tail = text[heading.start():heading.end() + 200]
            if not re.search(title, header_tail, re.I):
                continue
            if i + 1 == len(headings):
                continue
            body = text[heading.start():headings[i + 1].start()].strip()
            if len(body.split()) >= 12:
                candidates.append(body)
        if candidates:
            result[section] = max(candidates, key=len)
    return result


def _sentences(text):
    # Flatten layout first: changing HTML paragraph wrapping alone is not a disclosure change.
    text = " ".join(text.split())
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+(?=[A-Z\d\"'])", text) if part.strip()]


def compare_filings(previous, current, previous_text, current_text):
    """Return JSON-serializable excerpts and source citations for each changed section."""
    if previous.cik != current.cik or previous.filing_type != current.filing_type:
        raise ValueError("Compare the same company and filing type")
    if previous.filing_date >= current.filing_date:
        raise ValueError("Previous filing must precede current filing by filing date")
    before = extract_sections(previous_text, previous.filing_type)
    after = extract_sections(current_text, current.filing_type)
    sections = []
    for name in ("risk_factors", "management_discussion"):
        base = {"section": name, "previous_source": previous.url, "current_source": current.url}
        if name not in before or name not in after:
            sections.append({**base, "status": "unavailable", "changes": [],
                             "reason": "A bounded section could not be extracted from one or both filings"})
            continue
        old, new = _sentences(before[name]), _sentences(after[name])
        matcher = SequenceMatcher(None, old, new, autojunk=False)
        changes = []
        for tag, a, b, c, d in matcher.get_opcodes():
            if tag == "equal":
                continue
            removed, added = " ".join(old[a:b]), " ".join(new[c:d])
            changes.append({"kind": {"insert": "added", "delete": "removed", "replace": "modified"}[tag],
                            "previous_text": removed, "current_text": added,
                            "previous_sentence_range": [a + 1, b] if a < b else None,
                            "current_sentence_range": [c + 1, d] if c < d else None})
        sections.append({**base, "status": "changed" if changes else "unchanged",
                         "text_similarity": round(matcher.ratio(), 4), "changes": changes,
                         "unified_diff": "\n".join(unified_diff(old, new, fromfile=previous.accession_number,
                                                                tofile=current.accession_number, lineterm=""))})
    return {"schema_version": 1, "cik": current.cik, "filing_type": current.filing_type,
            "previous": {**asdict(previous), "source_url": previous.url, "sha256": sha256(previous_text.encode()).hexdigest()},
            "current": {**asdict(current), "source_url": current.url, "sha256": sha256(current_text.encode()).hexdigest()},
            "summary": {"changed_sections": sum(s["status"] == "changed" for s in sections),
                        "unavailable_sections": sum(s["status"] == "unavailable" for s in sections),
                        "change_blocks": sum(len(s["changes"]) for s in sections)},
            "note": "Textual differences require analyst review; similarity is not a financial risk score.",
            "sections": sections}


def select_filing_pair(cik, filings, *, filing_type="10-K", as_of=None):
    """Select the two latest original same-form filings available at the cutoff."""
    cutoff = date.fromisoformat(as_of) if as_of else date.max
    eligible = {}
    for record in filings:
        if record["filing_type"] != filing_type or date.fromisoformat(record["filing_date"]) > cutoff:
            continue
        filing = Filing(cik=str(cik), **{key: record[key] for key in
                        ("filing_type", "filing_date", "accession_number", "primary_document")})
        eligible[filing.accession_number] = filing
    ordered = sorted(eligible.values(), key=lambda f: (f.filing_date, f.accession_number))
    if len(ordered) < 2:
        raise ValueError("At least two comparable filings are required before the as-of date")
    previous, current = ordered[-2:]
    if previous.filing_date == current.filing_date:
        raise ValueError("Multiple same-day filings need explicit manual selection")
    return previous, current


def detect_latest_changes(cik, *, filing_type="10-K", as_of=None):
    """Use existing SEC ingestion and its local/S3 cache, downloading only the pair."""
    from src.data_layer.edgar_ingest import list_filings, download_filing
    previous, current = select_filing_pair(cik, list_filings(cik, [filing_type]),
                                          filing_type=filing_type, as_of=as_of)
    texts = [download_filing(f.cik, f.accession_number, f.primary_document) for f in (previous, current)]
    return compare_filings(previous, current, *texts)


def render_markdown(report):
    """Readable report with exact excerpts from both cited filings."""
    lines = [f"# Filing changes: CIK {report['cik']} ({report['filing_type']})", "",
             f"Previous: [{report['previous']['filing_date']}]({report['previous']['source_url']})",
             f"Current: [{report['current']['filing_date']}]({report['current']['source_url']})", "", report["note"]]
    def quote_text(text):
        return "\n".join("> " + line for line in text.splitlines())
    for section in report["sections"]:
        lines.extend(["", f"## {section['section'].replace('_', ' ').title()}: {section['status']}", ""])
        if section["status"] == "unavailable":
            lines.append(section["reason"])
        for index, change in enumerate(section["changes"], 1):
            lines.extend([f"### Change {index}: {change['kind']}", ""])
            for label, field, url in (("Previous", "previous_text", section["previous_source"]),
                                      ("Current", "current_text", section["current_source"])):
                if change[field]:
                    lines.extend([f"[{label} filing]({url}):", "", quote_text(change[field]), ""])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", required=True)
    parser.add_argument("--form", choices=["10-K", "10-Q"], default="10-K")
    parser.add_argument("--as-of", help="YYYY-MM-DD; excludes later filings")
    parser.add_argument("--previous-file", type=Path)
    parser.add_argument("--current-file", type=Path)
    parser.add_argument("--previous-date")
    parser.add_argument("--current-date")
    parser.add_argument("--previous-accession")
    parser.add_argument("--current-accession")
    parser.add_argument("--previous-document")
    parser.add_argument("--current-document")
    parser.add_argument("--output", type=Path, default=Path("data/processed/filing_changes/report"))
    args = parser.parse_args()
    if args.previous_file or args.current_file:
        required = (args.previous_file, args.current_file, args.previous_date, args.current_date,
                    args.previous_accession, args.current_accession, args.previous_document, args.current_document)
        if not all(required):
            parser.error("Offline comparison requires both files, dates, accessions and document filenames")
        previous = Filing(args.cik, args.form, args.previous_date, args.previous_accession, args.previous_document)
        current = Filing(args.cik, args.form, args.current_date, args.current_accession, args.current_document)
        if args.as_of and date.fromisoformat(current.filing_date) > date.fromisoformat(args.as_of):
            parser.error("Current filing is after --as-of")
        report = compare_filings(previous, current, args.previous_file.read_text(encoding="utf-8"),
                                 args.current_file.read_text(encoding="utf-8"))
    else:
        report = detect_latest_changes(args.cik, filing_type=args.form, as_of=args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(args.output) + ".json"), Path(str(args.output) + ".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"{report['summary']['changed_sections']} changed sections; "
          f"{report['summary']['unavailable_sections']} unavailable. Reports: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
