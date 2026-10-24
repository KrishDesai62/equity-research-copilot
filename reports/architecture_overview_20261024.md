# Interactive architecture overview — October 24, 2026

Implemented locally and tested:
- New How Ledger works navigation and #architecture route.
- Clickable two-path diagram: SEC data -> retrieval -> research agents, and
  SEC data -> financial features -> bankruptcy model; both feed combined research.
- Eight explorable nodes including the neural-network comparison and original
  ChromaDB path, accurately labeled as research implementations.
- Public configuration-only status endpoint with model-release gate, AI-provider
  and market-provider connection states. No credentials, private user data,
  inference or paid provider requests are exposed or run by the overview.
- Explicit input/output, examples and limits for each layer.
- Static-mode fallback uses unknown/unverified connection status.
- 354 tests passed, with one existing toy neural-network convergence warning.
- Browser checks: all eight nodes, selected aria-pressed state, withheld model
  status, neural-network explanation, desktop and 390px mobile layout without
  document overflow. Desktop/mobile screenshots saved under /tmp/ledger-architecture-*.
- Added a mocked successful AI draft test verifying valid source IDs, exact quote
  and withheld model context. Actual provider quality evaluation still needs keys.

Deployment blocked: two SSH connection attempts to the existing EC2 public address
18.224.246.52 timed out. Public CloudFront health remained 200. No successful
architecture deployment or remote backup is claimed. Confirm current instance
public address and IP-restricted SSH source, then deploy the six files below and
restart equity-research; run ledger-health.service afterward.

Deploy files:
frontend/index.html, frontend/app.js, frontend/architecture.js, frontend/styles.css,
src/web/server.py, src/web/architecture.py.

Model review/approval, final evaluation, external provider access and IAM activation
remain unchanged. No final evaluation was run or approval records invented.
