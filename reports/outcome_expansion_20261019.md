# Public outcome expansion — completed development comparison

Expanded the cached public-data pipeline, acquired additional SEC evidence, constructed new historical examples, reserved disjoint later companies, prepared source-only review packets, and ran two fixed model refits locally. No AWS or paid service was needed. Neither refit improved on the existing model, which was retained. No production model or threshold was released.

## Data added and sources

Sources are original SEC financial-statement bulk archives, EDGAR submissions and company facts, and primary issuer bankruptcy disclosures. Discovery used SEC Item 1.03 search results; search hits are sampling proxies, never bankruptcy labels by themselves. Raw URLs, source bytes and SHA-256 hashes are saved in acquisition checkpoints. The new annual frame came from the **2025 Q1 SEC financial-statement archive**, restricted to original nonfinancial annual reports published before April 1, 2025.

Added **October 31, 2021 training snapshots**, with outcomes through October 31, 2022 and the unchanged 200-day reporting window through May 19, 2023. This precedes the May 31, 2023 development forecast. Older snapshots are retained; company inclusion probabilities are updated under the union sampling design. The added wave selected 207 additional companies with fixed sampling, not model-driven selection.

| Training evidence | Previous | Expanded |
| --- | ---: | ---: |
| Eligible snapshots | 3,005 | 3,912 |
| Positive companies | 91 | 95 |
| Disclosure-negative rows | 2,913 | 3,815 |
| Unknown rows excluded | 899 | 1,137 |

There are **127 companies newly represented as disclosure negatives**; 68 reported negative operating cash flow before a forecast. These provide survivor examples beyond healthy-company contrasts. They are not proof of absence of every bankruptcy court case. Four companies newly contribute positive examples: CIKs 0001169245, 0001579823, 0001587221 and 0001659183. Names in historical sampling frames may differ from later legal names; primary statements and exact CIKs are preserved for review.

The target remains an issuer Chapter 7/11 petition explicitly disclosed in SEC filings within one year, under the stated observation window. Unknown follow-up, conflicting identities or unresolved disclosures are excluded rather than silently called negative. Acquisition completed without recorded errors. Features use the same 63-column schema; 3,005 previous input rows were checked against immutable inputs and reused, while new exact pre-forecast filings supplied remaining features. No future filing was intentionally admitted as a forecast feature.

## New reservation, still unscored

Before fitting, reserved **136 companies from a conditional eligible population of 1,424**, excluding all 3,913 previously selected company IDs and the expanded training selection. The forecast is April 1, 2025, the horizon ends April 1, 2026, and its reporting window ended **October 18, 2026**.

Acquisition produced 102 usable-input rows: **one source-corroborated positive, 83 disclosure negatives and 18 unknowns**. Other selected companies lacked admissible financial inputs. The positive is Twin Hospitality Group, with a primary issuer disclosure of a January 26, 2026 Chapter 11 petition. It has not received independent human approval. No model predictions were generated for the reservation.

The positive-support gate **fails**: one positive is far below the unchanged minimum of 20 companies. This is additional reserved evidence, not a replacement passing final test. The original December 2024 final cohort remains unscored and unconsumed.

## Same-development model comparison

Both refits use the expanded training table and fixed HGB settings, compared to the unchanged reference on the **same 167 resolved development companies**. Five fixed company partitions, nested calibration and held-company alert thresholds are unchanged. The fresh reservation is never read by the feature-fitting or model-comparison modules.

| Candidate | Median weighted log loss (lower is better) | Median PR-AUC | Weighted alert coverage range |
| --- | ---: | ---: | ---: |
| frozen_reference | 0.07258 | 0.507 | 100.0%–100.0% |
| native_balanced | 0.07491 | 0.504 | 29.2%–100.0% |
| native_company_balanced | 0.07719 | 0.400 | 0.0%–66.8% |

The original model was selected again. Neither refit achieved consistent 50% precision and recall, and some fitting folds could not select a qualifying threshold, so they abstained. No model benefit is claimed from collecting more data alone. The comparison reused development companies and cannot establish independent generalization.

## Review packets and reproduction

Prepared without model predictions or invented approvals:

- New positive training cases: `data/processed/validation/expanded_training_reviews_20261019/pending_reviews.jsonl` — four tasks, zero independent approvals.
- New reservation labels: `data/processed/validation/fresh_reservation_reviews_20261019/pending_reviews.jsonl` — 84 tasks, zero independent approvals.
- Unknown reservation outcomes: `data/processed/validation/fresh_reservation_reviews_20261019/unresolved.json` — 18 excluded units.

The packets include source URLs, source/checkpoint hashes, petition assertions and coverage checkpoints. A blank response template is provided; exporting it does not approve labels.

```bash
# Resume the complete expanded-training workflow; no reserved-company scoring:
.venv/bin/python -m src.validation.outcome_expansion_run
# Recheck the reserved outcome evidence and export review tasks:
.venv/bin/python -m src.validation.fresh_reservation_reviews
.venv/bin/python -m src.validation.expanded_training_review_packet
# Original operational-reference diagnostic:
.venv/bin/python -m src.validation.local_reliability_status
```

Uncached acquisition needs public internet access; saved replays use cached sources. Do not run concurrent copies of the same acquisition command. Frozen sampling/config/source checks stop on changes instead of silently migrating an experiment.

Validation: **282 tests passed**, with the existing toy neural-network convergence warning. All new source/model seals matched, the reference predictions reproduced across all five partitions, company reservations were disjoint, and the original final-evaluation marker remained absent. Predictive reliability is still unproven; independent outcome review and adequate untouched positive support remain unresolved.
