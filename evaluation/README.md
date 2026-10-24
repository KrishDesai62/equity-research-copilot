# Retrieval development checks

Run `.venv/bin/python -m scripts.evaluate_retrieval` from the repository root.
The fixture contains seven positive questions with relevant example passage IDs
and three abstention checks. The source examples were inspected by the assistant;
labels are **not independently human reviewed**, exhaustive, or held out.
Do not call this a validated human-labeled retrieval benchmark on a resume.

The command checks immutable corpus hashes, source/date fidelity, Hit@5 and MRR@5
against a lexical baseline and writes detailed results under ignored processed
outputs. This is separate from the frozen ML final evaluation and never runs model
inference. The seed found a real long-passage relevance-filter bug; its post-fix
score is a development regression result, not independent evidence of generalization.

To create an independent evaluation, a human reviewer should add questions and
relevance judgments without looking at the rankings, cover more issuers/forms and
failure cases, freeze that fixture, and keep it separate from tuning queries.

Bounded server check: `python -m scripts.load_check --requests 24 --concurrency 2`.
Run on the EC2 host with its protected environment. It creates temporary synthetic
accounts and cleans them up, refuses non-loopback targets, caps requests/workers,
and refuses AI-enabled configurations. It measures warm retrieval only. No filings
are imported, no model inference occurs, and no provider calls are made.
