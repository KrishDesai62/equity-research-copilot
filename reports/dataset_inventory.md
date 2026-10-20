# Dataset inventory and changes

Subsequent production-support work adds three more primary SEC bankruptcy event
records (WW International, 23andMe, Tupperware), pending first-case/prediction-unit
review, and a separate seven-feature development extension over 2,439 research
rows. Disclosure features are available on 137 rows. These additions do not alter
the sealed V3 dataset or add reviewed recent negatives. See
[the six-requirement implementation status](production_work_status.md).

The datasets below are already present locally. This update packages the verified
V3 bankruptcy cohort into reusable CSVs and documents the existing sources; it
adds no newly downloaded source or newly reviewed bankruptcy outcome. Counts
come from the saved local manifests and files. Different tables overlap and
must not be added together as a count of independent training examples.

| Data added | Source | Saved amount | Purpose / label status |
|---|---|---|---|
| Historical US bankruptcy cohort V3 | SEC quarterly archives, Companyfacts and submission histories; LoPucki BRD outcomes | 915 requested companies; 1,334 snapshots; 542 eligible and 792 unknown/outside-scope | Main one-year Chapter 7/11 research task. Unknowns excluded from fitting. |
| Earlier development panel V2 | SEC 2012/2014/2016 Q1 issuer samples and financial histories; BRD coverage review | 435 requested companies; 2,667 annual rows; 1,356 eligible under its frozen rules | Earlier experiment, retained separately with its exposed evaluation. |
| Broader financial research snapshot | SEC EDGAR / Companyfacts | 46 companies; 2,439 financial rows; 53,475 filing-index entries, through 2026-10-16 | Financial analysis and cited research. 88 positive one-year bankruptcy rows; 2,351 unknown. Filing entries are metadata, not 53,475 downloaded documents. |
| Recent quarterly financial source | SEC 2026 Q3 Financial Statement Data Set | 331 consolidated USD facts from 24 matching filings/companies | Additional financial evidence; supplies no bankruptcy outcomes by itself. |
| Bankruptcy registry | LoPucki BRD, release dated 2023-01-12 | 1,218 historical cases in imported source | Reviewed identity/coverage interval for V3 is 2010–2022. Entire registry is not automatically suitable model training data. |
| Recent cited bankruptcy cases | SEC filings for Big Lots, Spirit Airlines and Nikola | Three source-checked 2024–2025 case additions | Extends positive research evidence; does not establish complete recent negative coverage. |
| Filing text and changes | SEC 10-K / 10-Q / 8-K documents | 138 downloaded documents, 5,132 retrieval chunks, 46 filing pairs | Cited Q&A, risk-factor/MD&A retrieval and filing changes. Automated differences have no reviewed materiality label. |
| Accounting disclosures | SEC 8-K and AAER sources | 66 event candidates; six confirmed non-reliance disclosures; one cited AAER event | Candidates, non-reliance and enforcement remain distinct. Non-reliance is not proof of fraud; unknown negatives stay unknown. |
| Insider transactions | SEC Form 4 XML | 64 sampled transactions | Preserves transaction codes, shares, prices and publication dates. Awards/exercises are distinct from purchases and sales. |
| Financial numerical Q&A | Pinned FinQA development split | 883 questions with expected answers, tables, evidence and programs | Separate reasoning benchmark. Not merged with bankruptcy labels or presented as untouched final testing. |
| Polish bankruptcy benchmark | UCI Polish Companies Bankruptcy, `5year.arff` | 5,910 raw rows; 120 duplicate feature rows removed, leaving 5,790 | Separate diagnostic benchmark with 64 features. Missing company/date identifiers prevent verified company/time separation. |

Source links are recorded with individual downloads. Main source entry points:
[SEC financial statement archives](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets),
[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[LoPucki BRD](https://lopucki.law.ufl.edu/download_cases_table.php),
[FinQA](https://github.com/czyssrs/FinQA), and
[UCI benchmark source](https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip).
The research snapshot's `source_manifest.json` records actual URLs and hashes;
V3 also seals raw sources and per-company checkpoints. These are provenance links,
not a claim that their current online contents have been refreshed in this update.

## Changes made to the bankruptcy dataset

- Joined financial records using exact CIKs, original filing accessions and public
  dates; withheld unmatched, conflicting and future disclosures.
- Reconciled quarter flows into trailing-year quantities and added liquidity,
  leverage, cash flow, margins, annual changes, log assets and financial-input age.
- Kept prediction dates separate from older input filing dates. Inputs must be
  strictly earlier than prediction and at most 180 days old.
- Required a complete year of qualifying follow-up for negative labels. Confirmed
  bankruptcies are observed at their actual event date; a missing annual filing
  after bankruptcy cannot erase a known positive. Older frozen experiments retain
  their original data; a separate revision records the coverage correction.
- Resolved the Toys R Us date disagreement using the adjudicated court filing date,
  2017-09-19. Other unresolved identities or source gaps do not become healthy labels.
- V3 includes all eligible-frame registry case companies plus a 10% background
  sample, with inverse inclusion weights on every snapshot. These weights correct
  sampling, not missing sources or incomplete legal-entity coverage.
- Separated companies and prediction years between training, calibration and final
  evaluation; preprocessing is fitted only on training rows. Prior exposed companies
  are excluded from the V3 acquisition frame.
- Preserved separate bankruptcy, accounting, insider, retrieval and Q&A tables.
  They are different tasks, with different labels and evaluation requirements.

## Newly implemented export

Run locally, with no AWS or downloads:

```bash
.venv/bin/python -m src.validation.export_dataset
```

Output: `data/processed/datasets/bankruptcy_v3/`.

| File | Rows | Companies | Contents |
|---|---:|---:|---|
| `training.csv` | 180 | 81 | 18 financial features, source/date metadata, design weights, one-year target |
| `calibration.csv` | 124 | 75 | Same schema for separate calibration/model selection |
| `final.csv` | 238 | 75 | Features/metadata/weights; target omitted from training export |
| `manifest.json` | — | — | Source dataset hash, output checksums, feature order, target, partition counts and usage restrictions |

The final companies were **already evaluated**. Omitting their labels here does
not make them a fresh holdout. They cannot be recycled for training or independent
validation. The original sealed dataset and evaluation are preserved; repeated
exports verify hashes instead of overwriting data. Unknown/outside-scope rows
remain available in the original dataset for audit and are excluded from the export.

## Remaining limitation

This update adds packaging and traceability, not new independent outcome evidence.
The selected V3 model remains research only: five final bankruptcy companies,
no threshold meeting both precision/recall requirements, and unvalidated recent
population calibration. The most recent financial inputs reach 2026; reviewed
V3 outcome coverage ends in 2022. No labeled news-to-buy/sell or stock-return
benchmark has been added. See [the measured results](cohort_v3_results.md) and
[the full V3 data card](cohort_v3_data_card.md).


Latest public-outcome expansion: October 2021 training anchors increased eligible snapshots to 3,912 and positive-company coverage to 95. A separate April 2025 reservation has one positive, 83 disclosure negatives and 18 unknowns, with no predictions or independent approvals. See [sources, sampling, evidence dates and model comparison](outcome_expansion_20261019.md).
