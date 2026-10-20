# Local research workflow and LoPucki integration

**October 17 update:** the source/eligibility review now supplies historical
negative labels, the Toys R Us date is adjudicated, and a new unscored cohort is
reserved. See [evaluation preparation](evaluation_preparation.md). The remaining
gates below describe the earlier state before that review.

Run the connected local workflow:

```bash
.venv/bin/python -m src.validation.finish_local
```

For a fresh checkout, add `--download` to fetch missing public LoPucki and SEC
accounting source files. SEC access uses the existing contact header in `.env`.
Raw downloads, model diagnostics and local indexes remain gitignored.

## Implemented and exercised locally

- Imported the freely linked **full** Florida-UCLA-LoPucki Bankruptcy Research
  Database: 1,218 cases; 965 Chapter 11 case records have usable pre-bankruptcy
  CIKs. These are case counts, not necessarily independent companies.
- Preserved source checksums, release identity, Chapter 7/11 distinctions,
  missing identities, refilings and explicit coverage limits.
- Joined exact CIKs to the research companies: 18 case matches. Conflicting
  dates are quarantined from bankruptcy targets rather than silently overwritten.
- Rebuilt the 43-company SEC research table with bankruptcy, enforcement and
  non-reliance disclosure targets kept separate. Unknown negative outcomes
  remain unknown.
- Built a persistent SQLite FTS5 index of 4,564 cited passages. The shared
  retrieval function uses this local lexical baseline when present; Chroma
  remains an alternative when no local index exists or a store is injected.
- Replaced the Q&A stub with an extractive, date-filtered evidence response.
  Citations contain accession numbers, source URLs, chunk IDs, dates and exact
  excerpts. It abstains if no cited evidence is available.
- Checked citation fidelity for all 43 companies, plus early-date and unknown
  company abstention. These checks do **not** measure question relevance,
  numerical reasoning accuracy or the quality of an investment recommendation.
- Inspected the five unavailable Risk Factors sections: their quarterly source
  documents contain no standalone Item 1A heading. References to annual risks
  are not relabeled as a complete quarterly risk inventory.
- Downloaded all seven Item 4.02 candidates. Six contain non-reliance
  disclosures. The ASNA candidate contains executive-retirement Item 5.02 and
  exhibits, not Item 4.02, and was rejected. Decisions and original source
  checksums are in `data/curated/accounting_review_decisions.csv`.
- Validated the AWS CloudFormation templates, local ingestion/checkpoint/retry
  tests, and packaged Lambda source integrity. No resources were deployed.
  AWS account configuration is deferred at the user's request.

The six accepted accounting disclosures can describe repeat disclosures about
the same underlying error. They are not six independent fraud cases. The legacy
`target_restatement_flag` now represents a future confirmed Item 4.02 disclosure
within three years; its availability date is the public filing date. Negative
AAER/non-reliance targets require separate coverage review.

## Important limit discovered in the actual download

The January 12, 2023 BRD protocols explicitly state that the database is updated
through **December 31, 2022**, and that no further updates are intended. An older
FAQ's monthly-update description must not be used to extend this snapshot.
The population consists of qualifying large public-company debtor groups,
including specific parent/subsidiary rules, with assets of at least $100 million
in **1980 dollars** and a qualifying 10-K within three years of filing.

It cannot establish complete bankruptcy follow-up through 2024 for arbitrary
SEC registrants. The importer therefore creates **no automatic negative labels**.
Coverage approvals citing LoPucki beyond 2022-12-31 fail validation. Unresolved
CIKs are review candidates; name similarity is not an accepted identity match.

Sources: [full-table download page](https://lopucki.law.ufl.edu/download_cases_table_terms.php),
[coverage protocols](https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf).
The published data are attributed to the Florida-UCLA-LoPucki Bankruptcy Research
Database. This project does not claim a redistribution license for its raw data.

## Remaining evidence and decisions

1. **Negative bankruptcy coverage:** the actual BRD release does not remove the
   existing label gap. Review eligibility, identity completeness and follow-up
   within its scope, or obtain a broader event source. Importing an event table
   alone is insufficient. `outcome_coverage_review.csv` remains pending.
2. **Toys R Us:** issuer and case-administrator accounts report September 18,
   2017, while BRD and the court docket list September 19. Define the target's
   date policy and adjudicate the record with both sources retained. Those
   bankruptcy labels remain quarantined. Sources: [issuer SEC filing](https://www.sec.gov/Archives/edgar/data/1005414/000119312517287834/d451484d8k.htm),
   [court docket](https://www-a.vaeb.uscourts.gov/servecal/files/megadkt/677269.html),
   [case administrator](https://case.stretto.com/toyscommittee/caseinfo).
3. **Model retraining and final accuracy:** local readiness checks run, but
   logistic regression, XGBoost and NN comparisons remain gated on reviewed
   examples of both classes. The exposed old holdout is not rescored.
4. **Answer quality:** expert question/evidence annotations and FinQA numerical
   evaluation remain necessary before claiming research-answer accuracy.
   The implemented Q&A mode returns evidence, not generative conclusions.
5. **Live AWS test:** configure an AWS profile, region and private bucket later.
   The current scheduler remains disabled; local validation is complete.

Machine-readable results are in `data/processed/validation/local_completion.json`,
`data/processed/lopucki/report.json`, `data/processed/research/research_checks.json`
and `data/processed/validation/model_readiness/report.json`. Overall readiness
remains false; no new accuracy percentage or production model is claimed.
