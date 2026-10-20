# Public-source work completed — October 18, 2026

## Real development run

Ran ten comparisons: five classifiers on the original 18 features and the same
five with three extra financial features (accruals, net-liability proxy and cash
burn). Used the same 180 training rows and 124 company-separated calibration rows,
with design weights and company-fold calibration. Final rows were excluded before
label conversion; no completed final cohort was scored again.

The extended balanced logistic model had the lowest development log loss:
**0.058667**, versus **0.058683** for the original model. This change is negligible.
Its PR-AUC was slightly worse, **0.1851 versus 0.1916**, and no operating threshold
met both fixed precision/recall floors. Both neural variants converged with the
predefined 2,000-iteration budget. These are reused-development results, not new
independent validation or evidence of improved production detection.

Saved candidate, all comparison metrics and development predictions:
`data/processed/validation/development_refinement_20261018/`.
Re-running `.venv/bin/python -m src.validation.refine_development` verifies/reuses
this result. The selected candidate remains outside the serving registry.

## Larger public dataset audited

Downloaded only the training file of
[OpenFundex, published by Luke Danielson / ttchopper](https://huggingface.co/datasets/ttchopper/openfundex),
under its declared CC-BY-4.0 license. Pinned revision:
`a12f20990b4540ecc3db0e0294e985b7c81f58e9`. The raw file hash is checked against
upstream LFS metadata. Its validation, test and live files remain unopened.
The optional Parquet reader is pinned in `requirements-public-data.lock`.

Local audit results:

| Check | Result |
|---|---:|
| Training-file rows | 216,023 |
| Companies | 11,727 |
| Rows with filing dates after the card's stated 2019 training cutoff | 4,852 |
| Fiscal periods after filing dates | 13 |
| Known publisher events inside our filing-date horizon without a positive label | 251 |
| Positive publisher labels outside its known event's filing-date horizon | 154 |
| BRD positive-window rows without an upstream positive | 177 |

Some disagreements can reflect different target/date semantics, not necessarily
publisher mistakes. They still prevent dropping these labels directly into this
project's filing-date prediction task. The publisher-linked code repository also
returned HTTP 404 during the audit. An upstream zero has not been treated as an
independently reviewed healthy company. The file remains **quarantined**, and no
new negative labels were created.

Audit: `data/processed/validation/openfundex_audit/report.json`.
Command: `.venv/bin/python -m src.validation.openfundex_audit`.

## Primary-source event review pass

The audited training file produced **151 alias rows representing 119 unique
company/date/chapter events** for 2023–2025. The initial aggregate incorrectly
counted alias rows as separate event checks; execution now deduplicates the event
key while preserving the original queue and its hash.

- 118 unique events had nearby primary Item 1.03 filings.
- 103 yielded a single date candidate with the corrected date/action parser.
- 69 single-date candidates differed from the publisher date.
- Stronger issuer-subject checks corroborated **83 issuer events**; **36 remain
  unresolved**. Candidate dates and issuer-correlated events are different counts.

Regression tests prevent a restructuring-agreement date from being attached to a
later petition verb and ensure a second dated sentence is still examined. Chapter
15, involuntary and prospective statements are rejected. Subsidiary-only statements
do not establish an issuer petition. Primary proofs preserve document hashes and
statement offsets. None of this alone establishes complete court outcome coverage.

The candidate IDs remain recorded in
`config/evidence/openfundex_candidate_exposures.json`. The original strict
outcome-blind reservation workflow continues to exclude them. The separate SEC
disclosure study explicitly uses a **proxy-assisted case/background sampling design**
with nonzero role probabilities and inverse joint sampling/role weights. It is not
represented as an outcome-blind population sample. Source review exposure is
recorded separately from model fitting/selection exposure.

The review index includes company names, source links, candidate dates,
earlier recorded BRD cases and exposure flags:
`data/processed/validation/public_event_checks/review_index.csv`.
Per-event checkpoints and aggregate counts are in the same directory.
Command: `.venv/bin/python -m src.validation.verify_public_events`.
Cached reruns verify saved primary-source hashes. Errors cause a nonzero CLI exit.

## Brief source-to-dataset summary

| Source | What the project does with it |
|---|---|
| SEC EDGAR financial archives, Companyfacts and original filings | Reconstruct financial ratios and publication dates; extract cited text and filing changes. |
| LoPucki BRD | Historical bankruptcy outcomes and restricted large-public-company coverage review through 2022. |
| Recent SEC Item 1.03 filings | Verify event evidence and petition-date candidates; no automatic healthy labels. |
| OpenFundex | Quarantined candidate source; audit labels/dates and locate primary evidence before admission. |
| FinQA | Separate numerical Q&A benchmark; not bankruptcy training data. |
| UCI Polish bankruptcy dataset | Separate diagnostic classifier benchmark; not US-market validation. |

Unknown outcomes remain unknown. Company/time partitions, source hashes, sampling
weights and full-year negative follow-up remain required. The new evidence pass
has not supplied complete recent negative coverage or a new independent final
result. AWS remains deferred. No production release was created.

Verification: **192 tests passed**, compilation, CloudFormation lint and whitespace
checks passed. One existing small neural-network test fixture emits a convergence
warning. Passing these checks does not establish predictive accuracy.

## Subsequent scoped cohort acquisition

The subsequent [SEC-disclosure cohort report](sec_disclosure_v1.md) supersedes the
"no new negative labels" status **for that separate, explicitly scoped target**.
It records the actual 498-company pass, 29 positive and 230 disclosure-negative
snapshots, 65 unknowns, 174 exclusions and five fitted models. These are not
complete court-bankruptcy negatives. No production artifact was promoted, and
the underpowered new final cohort remains unscored.
