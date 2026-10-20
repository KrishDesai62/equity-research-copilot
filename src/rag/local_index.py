"""Offline SQLite full-text filing retrieval; BM25 is lexical relevance only."""
import argparse
from datetime import date
import json
from pathlib import Path
import re
import sqlite3

DEFAULT_PATH = "data/processed/research/filings.sqlite"


def build_index(source="data/processed/research/retrieval_chunks.jsonl", destination=DEFAULT_PATH):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    count = 0
    try:
        with sqlite3.connect(temporary) as connection:
            connection.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, ticker TEXT, filing_type TEXT, filing_date TEXT, section_title TEXT, payload TEXT)")
            connection.execute("CREATE VIRTUAL TABLE passages USING fts5(text)")
            for line in source.open(encoding="utf-8"):
                row = json.loads(line)
                date.fromisoformat(row["filing_date"])
                if not row["text"].strip() or not row.get("source_url") or not row.get("accession_number"):
                    raise ValueError("Chunks require text and source citations")
                count += 1
                connection.execute("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (count, row["chunk_id"], row["ticker"], row["filing_type"], row["filing_date"], row["section"], json.dumps(row)))
                connection.execute("INSERT INTO passages(rowid, text) VALUES (?, ?)", (count, row["text"]))
            if not count:
                raise ValueError("Cannot publish an empty filing index")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"chunks": count, "index": str(destination), "retrieval_method": "SQLite FTS5 BM25 lexical baseline"}


class LocalIndex:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    def retrieve(self, query, *, ticker=None, filing_type=None, date_range=None,
                 section_title=None, has_table=None, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))[:64]
        if not terms:
            return []
        clauses, values = ["passages MATCH ?"], [" OR ".join('"' + t + '"' for t in terms)]
        for key, value in (("ticker", ticker), ("filing_type", filing_type), ("section_title", section_title)):
            if value is not None:
                clauses.append(f"m.{key} = ?")
                values.append(value)
        if date_range:
            start, end = [date.fromisoformat(str(d)[:10]).isoformat() for d in date_range]
            if start > end:
                raise ValueError("Date range starts after it ends")
            clauses.append("m.filing_date BETWEEN ? AND ?")
            values.extend([start, end])
        if has_table is True:
            return []
        sql = "SELECT m.payload, bm25(passages) FROM passages JOIN metadata m ON passages.rowid=m.id WHERE " + " AND ".join(clauses) + " ORDER BY bm25(passages),m.chunk_id LIMIT ?"
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(sql, values + [k]).fetchall()
        result = []
        for payload, rank in rows:
            row = json.loads(payload)
            result.append({"text": row["text"], "ticker": row["ticker"], "filing_type": row["filing_type"],
                "filing_date": row["filing_date"], "section": row["section"], "metadata": row,
                "similarity_score": None, "lexical_rank": rank, "retrieval_method": "sqlite_fts5_bm25"})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/research/retrieval_chunks.jsonl")
    parser.add_argument("--destination", default=DEFAULT_PATH)
    print(json.dumps(build_index(**vars(parser.parse_args())), indent=2))
