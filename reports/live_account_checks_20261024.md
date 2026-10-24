# Ledger live account checks — October 24, 2026

Environment: https://d3lj7roa7yk0i3.cloudfront.net, separate Chrome test profile.
User signed in manually; no credentials or cookie values collected in output.

Verified against deployed APIs in the authenticated browser:

- Session/profile returned 200 and the frontend displayed Sign out.
- A temporary watchlist entry and note persisted to PostgreSQL and survived a page reload.
- Session survived reload. Original saved research restored afterward (200).
- Apple cited retrieval returned 200 with three citations. Broad risk question also retrieved accounting passages: citation presence is not a retrieval-relevance score.
- Unknown ticker returned unavailable with zero citations.
- Historical cutoff returned three citations, all filed on/before 2025-01-01.
- Future cutoff rejected with 400.
- Netflix import completed: six reports, 108 passages, zero reported warnings.
- Netflix question returned three Netflix/SEC citations; all excerpt/chunk IDs matched the imported payload passages.
- Minute request quota returned 429 using invalid requests without additional SEC downloads.
- Logout removed the authenticated session. Profile, answer and import endpoints each returned 401 afterward.

Not verified live: isolation between two authenticated accounts, session persistence across browser restart, daily import quota, session expiry, recovery/backup restore, CloudWatch alert delivery or an independent security audit. Local automated isolation tests do not replace the two-account deployment check. Exact SEC page content was not independently refetched during this account test.

The separate test browser was left signed out. Imported public Netflix data remains available in the shared catalog. Temporary saved note/watchlist changes were restored.
