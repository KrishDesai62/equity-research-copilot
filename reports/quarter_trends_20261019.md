# Quarterly trends and false-positive audit

Completed a local cached-source audit and a fixed, point-in-time feature experiment. No network downloads or AWS services were needed. Existing model/data files and final reservations were preserved.

The unstable alert partition contains five disclosure-negative companies: Novo Integrated Sciences, Crown Electrokinetics, Acreage Holdings, Helbiz and EzFill. Each has design weight ten and negative operating cash flow. Cached checkpoints record completed reporting follow-up and no unresolved bankruptcy-disclosure accessions. The audit established no contrary issuer-petition evidence and therefore did not relabel them. This is not an independent human review or a court-record clearance.

The source-linked packet is `data/processed/validation/quarter_trends_20261019/false_positive_audit.json`. These are development companies; none was moved into training to make the false positives disappear.

Added nine features: quarter-to-quarter cash, liabilities, assets, current ratio, cash/assets and retained-earnings changes, plus period spacing and public-data age. Prior balances must come from reconciled original filings, be published no later than the current input filing, and represent a fiscal period 60–130 days earlier. Missing or conflicting prior balances remain unknown. Existing annual trends were preserved rather than duplicated. Quarter cash changes are available in 2,951/3,005 training snapshots and 157/167 development companies.

Two fixed refits use the same HGB settings as the reference, with balanced and company-balanced training weights. Models use the same source tables, prediction target, design weights and five company partitions. Calibration is nested; alert thresholds are selected using other companies only. Configuration and implementation hashes were sealed before comparison.

| Candidate | Median weighted log loss | Median PR-AUC | Held-company alert precision range | Held-company alert recall range |
| --- | ---: | ---: | ---: | ---: |
| frozen_reference | 0.07258 | 0.507 | 26.5%–65.5% | 52.9%–55.9% |
| trend_balanced | 0.07095 | 0.449 | 25.4%–47.4% | 50.0%–58.8% |
| trend_company_balanced | 0.07640 | 0.394 | 20.0%–42.9% | 33.3%–58.3% |

The balanced trend refit reduced median probability loss by **2.2%**, but worsened alert precision and PR-AUC. It is saved separately as probability research, not as a replacement alert model. Neither refit satisfies both 50% precision and recall across partitions. The prior alert model and serving behavior remain unchanged. No threshold is released, no model is approved and no final predictions were generated.

These are reused-development results, not fresh independent validation. All five partitions overlap in company membership. Repeated tuning cannot establish reliability. Independent label review, additional informative training outcomes and an untouched evaluation remain outstanding.

Reproduce the immutable experiment and inspect the local reliability status:

```bash
.venv/bin/python -m src.validation.quarter_trend_data
.venv/bin/python -m src.validation.quarter_trend_models
.venv/bin/python -m src.validation.local_reliability_status
```

Verification: the frozen reference predictions reproduced across all five partitions; saved implementation/config/data/model hashes matched; final evaluation remains unstarted. Tests protect future-data exclusion, original-filing reconciliation, missing/conflicting history and snapshot timing. Full suite: **277 passed**, with one existing toy neural-network convergence warning.
