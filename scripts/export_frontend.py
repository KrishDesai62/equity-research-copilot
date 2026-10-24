"""Export a public-data-only static research snapshot; never export outcomes or models."""
import csv
import json
import math
from pathlib import Path

SOURCE = Path('data/processed/research')
DEST = Path('frontend/data')
FIELDS = ['Revenues_ttm', 'Revenues_quarter', 'OperatingIncomeLoss_ttm', 'OperatingCashFlow_ttm',
          'NetIncomeLoss_ttm', 'Assets', 'Liabilities', 'Cash', 'current_ratio',
          'liabilities_to_assets', 'revenue_growth_yoy', 'operating_margin']

def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def export():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'companies').mkdir(exist_ok=True)
    read = lambda name: [json.loads(line) for line in (SOURCE/name).read_text().splitlines() if line.strip()]
    companies, chunks, documents = read('companies.jsonl'), read('retrieval_chunks.jsonl'), read('documents.jsonl')
    with (SOURCE/'financial_features.csv').open() as handle:
        financial = list(csv.DictReader(handle))
    catalog = []
    for company in companies:
        ticker = company['research_ticker']
        passages = [r for r in chunks if r['ticker']==ticker]
        filings = [r for r in documents if r.get('ticker')==ticker]
        series = {}
        for row in sorted((r for r in financial if r['ticker']==ticker), key=lambda r:r['filing_date']):
            if row['filing_date'] > '2026-10-23':
                continue
            if number(row.get('Revenues_ttm')) is None:
                continue
            series[row['fiscal_period_end']] = dict(period=row['fiscal_period_end'],filed=row['filing_date'],
                source_url=row.get('source_url',''), **{f:number(row.get(f)) for f in FIELDS})
        history = sorted(series.values(),key=lambda r:r['period'])[-12:]
        entry = dict(ticker=ticker,name=company['current_sec_name'],cik=company['cik'],
                     passage_count=len(passages),filing_count=len(filings),latest=history[-1] if history else None)
        catalog.append(entry)
        safe_filings = [{k:r.get(k) for k in ['accession_number','filing_date','filing_type','source_url','primary_document']} for r in filings]
        payload = dict(company=entry,history=history,filings=safe_filings,
                       passages=[{k:r.get(k) for k in ['chunk_id','text','section','filing_date','filing_type','source_url','accession_number']} for r in passages])
        (DEST/'companies'/f'{ticker}.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
    catalog.sort(key=lambda c:c['ticker'])
    result = dict(snapshot_date='2026-10-16',export_date='2026-10-23',companies=catalog,
                  totals=dict(companies=len(catalog),filings=len(documents),passages=len(chunks)),
                  mode='static_public_sec_snapshot')
    (DEST/'catalog.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False))
    print(f"Exported {len(catalog)} companies, {len(documents)} filings and {len(chunks)} passages to {DEST}")

if __name__=='__main__':
    export()
