"""Independently cross-check Company Facts against original inline-XBRL filings."""
import argparse
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

import pandas as pd


class InlineFacts(HTMLParser):
    """Conservative reader for dot-decimal facts and non-dimensional contexts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.contexts, self.facts = {}, []
        self.context = self.date_field = self.fact = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "xbrli:context":
            self.context = {"id": attrs["id"], "dimensional": False}
        elif self.context is not None:
            if tag in {"xbrldi:explicitmember", "xbrldi:typedmember"}:
                self.context["dimensional"] = True
            if tag in {"xbrli:instant", "xbrli:startdate", "xbrli:enddate"}:
                self.date_field = tag.split(":")[1]
                self.context[self.date_field] = ""
        if tag == "ix:nonfraction":
            self.fact = {**attrs, "text": ""}

    def handle_data(self, data):
        if self.context is not None and self.date_field:
            self.context[self.date_field] += data
        if self.fact is not None:
            self.fact["text"] += data

    def handle_endtag(self, tag):
        if tag == "xbrli:context" and self.context:
            self.contexts[self.context["id"]] = self.context
            self.context = None
        if tag in {"xbrli:instant", "xbrli:startdate", "xbrli:enddate"}:
            self.date_field = None
        if tag == "ix:nonfraction" and self.fact:
            fact = self.fact
            self.fact = None
            transform = fact.get("format", "").split(":")[-1]
            if fact.get("continuedat") or transform not in {"", "num-dot-decimal", "numdotdecimal"}:
                return
            try:
                text = fact["text"].replace(",", "").replace("\u00a0", "").strip()
                fact["value"] = float(Decimal(text) * (Decimal(10) ** int(fact.get("scale", 0))))
                if fact.get("sign") == "-":
                    fact["value"] *= -1
                self.facts.append(fact)
            except (ValueError, InvalidOperation):
                return


def audit_comparison(report_path, provenance_path="data/processed/validation/sec_pilot/fact_provenance.csv", *, destination=None):
    report = json.loads(Path(report_path).read_text())
    provenance = pd.read_csv(provenance_path, dtype={"cik": str})
    if "accession_number" not in provenance and "accn" in provenance:
        provenance = provenance.rename(columns={"accn": "accession_number"})
    checks, sources = [], []
    for side in ("previous", "current"):
        filing = report[side]
        raw_path = Path("data/raw/filings") / filing["cik"] / (filing["accession_number"] + ".txt")
        text = raw_path.read_text()
        parser = InlineFacts()
        parser.feed(text)
        sources.append({"source_url": filing["source_url"], "raw_sha256": sha256(text.encode()).hexdigest()})
        cells = provenance[provenance.accession_number == filing["accession_number"]]
        for _, cell in cells.iterrows():
            values = set()
            for fact in parser.facts:
                context = parser.contexts.get(fact.get("contextref"), {})
                end = context.get("instant", context.get("enddate"))
                if (fact.get("name") != "us-gaap:" + cell.source_tag or context.get("dimensional")
                        or end != cell.end or fact.get("unitref", "").lower() != "usd"):
                    continue
                if pd.notna(cell.get("start")) and context.get("startdate") != cell.start:
                    continue
                values.add(fact["value"])
            checks.append({"accession_number": filing["accession_number"], "field": cell.field,
                           "companyfacts_value": cell.val, "original_filing_values": sorted(values),
                           "matches": values == {cell.val}})
    result = {"method": "Independent parsing of original non-dimensional inline-XBRL USD values",
              "sources": sources, "checks": checks, "all_match": bool(checks) and all(c["matches"] for c in checks)}
    output = Path(destination) if destination else Path(report_path).with_name(Path(report_path).stem + "_financial_audit.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"{len(checks)} financial checks; all match: {result['all_match']}; {output}")
    if not result["all_match"]:
        raise ValueError("Financial audit has missing or mismatching original facts")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    audit_comparison(args.report)
