# Connected equity research dataset

See [the local workflow update](local_completion.md) for the full LoPucki import,
the offline cited Q&A implementation, inspected accounting disclosures and
current limitations. Earlier sections describe the initial dataset build.

The builder in `src/data_layer/research_dataset.py` connects filing research and
financial analysis with bankruptcy outcomes, accounting-event discovery, insider
transactions, and question-answering evaluation. Bankruptcy remains a core task.

The research universe contains 43 companies: the previous 36-company sample,
including 21 cited bankruptcy cases, plus Amazon, Alphabet, Meta, Nvidia,
Johnson & Johnson, Exxon Mobil, and Kraft Heinz. It is a purposive research sample, not a
historical S&P 500 membership list. The seven added companies do not constitute a
new validated holdout and have no automatically assumed healthy labels.

## Run

```bash
.venv/bin/python -m src.data_layer.research_dataset
```

Uses the real SEC contact configured in `.env`. Public SEC requests share the
existing rate limiter. Raw responses are cached locally, with source URLs and
SHA256 checksums exported in the manifest. Downloads do not deploy AWS resources.
The existing AWS monitor continues to use its own explicitly configured watchlist.

Default window: 2010-01-01 through 2024-12-31. The end date comes from
`config/research_universe.json`. Dates describe public filing availability, not
fiscal period ends. Company Facts extracts original, same-accession disclosures;
historical feature calculation excludes inputs published after each row.

The SEC submissions importer reads both recent and date-overlapping archived
history. The filing index includes original 10-K, 10-Q, 8-K and Form 4 filings;
amendments require separate handling. Download limits default to two annual
filings, one quarterly filing and one Form 4 per company. For known bankrupt
companies these samples precede the cited bankruptcy date. The financial feature
table retains its wider history, with post-bankruptcy outcomes unavailable.

```bash
.venv/bin/python -m src.data_layer.research_dataset \
  --annual-documents 2 --quarterly-documents 2 --insider-documents 5
```

This generates quarterly as well as annual change pairs when sufficient filings
are available. Document pairs with missing extraction boundaries remain visibly
unavailable in their reports. A failed download records a partial build and
causes a nonzero CLI exit; rerunning reuses successful cached downloads.

## Outputs: `data/processed/research/`

| File | Purpose |
|---|---|
| `companies.jsonl` | CIK registry; research ticker, current SEC name, cited bankruptcy date and coverage status |
| `filings.jsonl` | Historical filing index with accession, public date, form, source URL and acceptance timestamp when present |
| `financial_features.csv` | As-filed financial history and ratios, with separate 1-year and 3-year bankruptcy targets |
| `fact_provenance.csv` | Source XBRL tags, units, accession numbers, public dates and values |
| `financial_rows_withheld.csv` | Company Facts rows without an exact match to an original SEC submission and public date; excluded from the research feature table |
| `documents.jsonl` | Downloaded document paths, hashes and available/missing sections |
| `retrieval_chunks.jsonl` | Risk Factors and MD&A chunks with stable IDs and source citations; ready for a future index-loading step |
| `filing_pairs.jsonl`, `changes/` | Matched filings and exact textual change reports |
| `accounting_event_candidates.jsonl` | Separate 8-K Item 4.02 non-reliance and Item 1.03 bankruptcy/receivership review candidates |
| `aaer_events.jsonl` | Cited issuer enforcement events, initially Kraft Heinz AAER 4248 |
| `insider_transactions.jsonl` | Sampled non-derivative transactions, preserving codes, missing prices and public availability dates |
| `qa_review.jsonl` | Filing-change questions awaiting human answers and evidence annotation |
| `finqa_development.jsonl` | Separate pinned FinQA development benchmark with original tables, expected answers, evidence and programs |
| `outcome_coverage_review.csv` | Company-level bankruptcy, AAER and restatement coverage review worksheet |
| `source_manifest.json`, `summary.json` | Provenance, counts, extraction conflicts, failures and limitations |

Join SEC records using CIK and accession number. A research ticker is a label,
not a validated historical ticker mapping. FinQA identities are not automatically
matched to CIKs. Its original development split is retained and must not be
presented as a previously untouched final test.

## What counts as a label

Bankruptcy events retain the existing cited dates. A positive target requires an
event within the prediction horizon and observation window. Missing events do
**not** imply negative outcomes. Negative bankruptcy labels remain unknown until
complete follow-up has been independently reviewed. A row is training-eligible
only when both requested bankruptcy horizons satisfy the existing label rules.

Item 4.02 is non-reliance, not proof of fraud. Item 1.03 includes receivership,
not just Chapter 11. Both remain review candidates. The initial curated AAER
event is Kraft Heinz release 4248, dated September 3, 2021, backed by its SEC
order and the issuer's CIK. Its target is a future enforcement release within
three calendar years of a filing; the label becomes available on the release
date. This is an enforcement outcome, not a blanket proven-fraud label or a
claim that each preceding filing was misstated. All AAER negatives and
restatement targets remain unknown without reviewed coverage. One AAER issuer
is insufficient for training a validated accounting model.

The coverage worksheet is a review artifact, not an automatic approval channel:
editing it does not silently enable production training. Similarly, generated
diffs do not become golden Q&A answers. Review entries are preserved across
rebuilds, and newly encountered companies and pairs are appended.

## Validation and remaining work

Tests cover archived SEC histories, date filtering, malformed metadata, separate
event types, bankruptcy retention, unknown outcomes, Form 4 issuer matching,
rendered-versus-raw XML, transaction-code distinctions, review preservation and
connected output construction. Run:

```bash
.venv/bin/python -m pytest tests/test_research_dataset.py -q
```

No model is retrained or promoted by this builder. It does not claim an accuracy
improvement. RAG embeddings/index loading, the Q&A agent implementation, reviewed
golden answers and complete outcome labels remain separate work. Historical
returns, delisting returns and verified index membership are still needed before
portfolio backtesting. News snippets are not substituted for a historical news
archive with point-in-time publication metadata.

The live reconciliation found three Company Facts accessions categorized as
original forms whose submissions are amendments (JCP, ROST, DLTR), plus one FTR
accession with conflicting public dates. These four rows are quarantined from
this research table. Earlier model datasets need the same submissions cross-check
before relying on their reported performance; they are not silently retrained here.

Source reconciliation now happens **before** any TTM or trend calculation, so
excluded raw inputs cannot contaminate later rows. Current model input and
outcome-coverage status is in [model_readiness.md](model_readiness.md). Run
`.venv/bin/python -m src.validation.model_readiness --evaluate` after completing
coverage review to compare all three model types on development folds. An
unavailable accuracy stays null; the exposed final holdout is not rescored.

Sources: [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[SEC AAER](https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases),
[FinQA authors' repository](https://github.com/czyssrs/FinQA).
FinQA source README and license are downloaded alongside its pinned development data.
