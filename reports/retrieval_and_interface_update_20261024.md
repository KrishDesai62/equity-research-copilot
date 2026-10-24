# Ledger retrieval and interface update — October 24, 2026

Deployed to https://d3lj7roa7yk0i3.cloudfront.net on existing EC2 service.

Changes:
- Remove conversational stopwords from SQLite FTS searches.
- Expand common financial topic variants for Q&A candidate retrieval.
- Retrieve a larger candidate pool, rank by topic overlap and risk-section intent,
  deduplicate and retain existing ticker/date/citation checks.
- Broad risk questions require risk-factor passages and abstain without them.
  This is a deterministic relevance heuristic, not an independently measured
  semantic retrieval benchmark.
- Compact blue/white financial workspace, sans-serif headings, stronger text
  contrast, denser tables, clearer source labels and mobile chart sizing.
- Remove personal initials from the generic workspace profile.
- Replace stale frontend cutoff maximum with current browser-local date.
- Show distinct login/rate-limit errors in filing search.

Validation:
- 337 tests passed (one existing toy neural-network convergence warning).
- Four new relevance regressions: risk/accounting distractor, abstention,
  financial-topic relevance and date restriction, stopword cleanup.
- Desktop 1440px/mobile 390px previews: no horizontal document overflow.
- Deployed backend Apple broad-risk query: 3 citations, all risk_factors.
- Public /api/health, /api/session, /styles.css returned 200; stylesheet marker
  confirms updated assets served through CloudFront.

Predeployment affected-file backup:
/home/ubuntu/ledger-code-backups/pre-redesign-20261024.tar.gz

No model changes, training, final-outcome evaluation or model release.
Two-account isolation remains an outstanding live verification step.
