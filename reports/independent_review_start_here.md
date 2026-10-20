# Start here: human source review

You may perform one source review yourself. As the project author, do not describe your own review as external independent validation. Ask a second person who did not generate the labels or tune the model to reach a separate conclusion from primary sources. An accounting/finance student, researcher or analyst who understands issuer identity and financial filings can help; this project does not require a paid certification.

The two-agreeing-reviewer rule is a requirement of the current project validator (`src/validation/current_outcome_reviews.py`), not a universal rule or proof of reviewer independence. Do not create two reviewer names for the same person. Do not attest to checks you have not performed. The validator records attestations; it cannot authenticate people or certify legal conclusions.

Review labels from SEC evidence, not whether model predictions look right. Keep the frozen model unchanged and do not use final-cohort review findings to tune features or thresholds.

## Begin with the 20 proposed positive final-cohort labels

Each source below is a proposed bankruptcy example, not a human-approved outcome. Read the source and record the petition date yourself before comparing it with the automated assertion. These tasks all forecast from **2024-12-31** through **2025-12-31**, with evidence admitted through **2026-07-19**.

For each case:

1. Confirm the exact reporting issuer and CIK. A subsidiary, customer or counterparty bankruptcy alone does not establish this issuer's label.
2. Find an actual Chapter 7/11 petition statement. A warning, negotiation, delisting or intention to file is insufficient.
3. Record the petition date, not merely the SEC filing date or emergence date. Check that it is after the forecast date and on or before the horizon end.
4. Check that the source filing was public by the evidence cutoff. Save its URL, relevant section and your own short evidence notes.
5. Record `1` only if the target is supported; otherwise mark uncertain or flag a correction. Reading a prepared assertion is not a substitute for reading the source.

| Company | CIK | Primary source to start with |
| --- | --- | --- |
| ACCELERATE DIAGNOSTICS, INC | 0000727207 | [Filing 1](https://www.sec.gov/Archives/edgar/data/727207/000110465925046104/tm2514448d1_8k.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/727207/000110465925079855/tm2523695d1_8k.htm) |
| GLOBAL CLEAN ENERGY HOLDINGS, INC. | 0000748790 | [Filing 1](https://www.sec.gov/Archives/edgar/data/748790/000121390025033420/ea0238127-8k_global.htm) |
| ZYNEX INC | 0000846475 | [Filing 1](https://www.sec.gov/Archives/edgar/data/846475/000110465925121390/zyxi-20251215x8k.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/846475/000110465925122563/zyxi-20251217x8k.htm); [Filing 3](https://www.sec.gov/Archives/edgar/data/846475/000110465926004122/zyxi-20260114x8k.htm); [Filing 4](https://www.sec.gov/Archives/edgar/data/846475/000110465926009241/zyxi-20260129x8k.htm); [Filing 5](https://www.sec.gov/Archives/edgar/data/846475/000110465926013411/zyxi-20260207x8k.htm); [Filing 6](https://www.sec.gov/Archives/edgar/data/846475/000110465926032723/zyxi-20260319x8k.htm) |
| WOLFSPEED, INC. | 0000895419 | [Filing 1](https://www.sec.gov/Archives/edgar/data/895419/000119312525153145/d755867d8k.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/895419/000119312525199310/d36059d8k.htm); [Filing 3](https://www.sec.gov/Archives/edgar/data/895419/000119312525224251/d22768d8k.htm) |
| DZS INC. | 0001101680 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1101680/000110168025000021/dzsi-20250314.htm) |
| CUTERA INC | 0001162461 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1162461/000119312525046217/d878024d8k.htm) |
| MARIN SOFTWARE INC | 0001389002 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1389002/000095017025093994/mrin-20250630.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/1389002/000095017025102108/mrin-20250729.htm); [Filing 3](https://www.sec.gov/Archives/edgar/data/1389002/000095017025113236/mrin-20250829.htm) |
| TPI COMPOSITES, INC | 0001455684 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1455684/000117184325005450/f8k_081525.htm) |
| LOOP MEDIA, INC. | 0001643988 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1643988/000149315225018533/form8-k.htm) |
| CANOO INC. | 0001750153 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1750153/000162828025001918/goev-20250117.htm) |
| LUMINAR TECHNOLOGIES, INC./DE | 0001758057 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1758057/000114036125045494/ef20061234_8k.htm) |
| SUNNOVA ENERGY INTERNATIONAL INC. | 0001772695 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1772695/000177269525000105/nova-20250608.htm) |
| WW INTERNATIONAL, INC. | 0000105319 | [Filing 1](https://www.sec.gov/Archives/edgar/data/105319/000119312525114307/d934787d8k.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/105319/000119312525142383/d943798d8k.htm) |
| DANIMER SCIENTIFIC, INC. | 0001779020 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1779020/000095017025041367/dnmr-20250318.htm) |
| 23ANDME HOLDING CO. | 0001804591 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1804591/000119312525060817/d933650d8k.htm) |
| SONDER HOLDINGS INC. | 0001819395 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1819395/000162828025052329/son-20251107.htm) |
| BENSON HILL, INC. | 0001830210 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1830210/000183021025000016/bhil-20250319.htm); [Filing 2](https://www.sec.gov/Archives/edgar/data/1830210/000110465925052703/tm2515920d1_8k.htm); [Filing 3](https://www.sec.gov/Archives/edgar/data/1830210/000110465925086777/tm2525102d1_8k.htm) |
| WAG! GROUP CO. | 0001842356 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1842356/000184235625000098/pet-20250829.htm) |
| OMEGA THERAPEUTICS, INC. | 0001850838 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1850838/000119312525027349/d896578d8k.htm) |
| BRIGHT GREEN CORP | 0001886799 | [Filing 1](https://www.sec.gov/Archives/edgar/data/1886799/000149315225008512/form8-k.htm) |

## The 315 proposed negative labels require more work

Open the full final-cohort packet below, then the `source_checkpoint.path` for a company. Check source identity, the complete one-year window, follow-up and relevant SEC disclosures through the cutoff. Continuing to file financial reports or having no Item 1.03 match is insufficient by itself. The broader coverage audit identified unreviewed report bodies. An absence claim needs documented search/review coverage; if that is incomplete, record uncertainty (`null`) rather than approve `0`.

The target is SEC-disclosed issuer Chapter 7/11 petitions under this protocol, not a guarantee that no court case exists anywhere. Do not approve a broader claim than the evidence supports.

Full packet: [pending_final_reviews.jsonl](../data/processed/validation/final_outcome_audit_v2_20261019/pending_final_reviews.jsonl).

Coverage backlog, with primary filing-index links and affected units: [remaining_current_reports.json](../data/processed/validation/reconciled_coverage_20261019/remaining_current_reports.json). This spans training/development/fresh-reservation negatives and is not the complete original-final review checklist.

## Record a real response in a separate file

Start from [response_template.json](../data/processed/validation/current_review_status_20261019/response_template.json). Copy it for each company into your own JSONL response file; do not edit the frozen packet, source data or template in place. Use the task's exact `unit_id`, checkpoint SHA and current packet SHA. Supply your real reviewer identifier, actual timezone-aware review timestamp, evidence notes and outcome. Set attestations true only after doing the checks. `approved_label` is `1`, `0`, or `null` for uncertain. For a `0`, complete disclosure coverage must actually have been checked.

Each person should review separately before reconciling disagreements. Two agreeing genuine reviewer responses are required for a label to pass the current validator. Conflicting labels require investigation; they must not be forced into agreement. Source corrections need a new versioned evidence packet rather than edits to this frozen one.

After recording real responses, validate them with:

```bash
.venv/bin/python -m src.validation.current_outcome_reviews --reviews /absolute/path/to/reviews.jsonl
```

This checks response format and evidence references. It does not prove reviewer independence, source correctness, production readiness or authorize final-test scoring. Even completed label review does not fix the model's existing precision/recall instability.
