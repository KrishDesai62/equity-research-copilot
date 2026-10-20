"""Cache source datasets with checksums; SEC calls use the configured contact."""
from hashlib import sha256
import json
from pathlib import Path
import requests
import time
from src.data_layer.edgar_ingest import limiter, sec_headers

UCI_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"


def fetch(url, destination):
    path = Path(destination)
    if not path.exists():
        headers = sec_headers() if url.startswith(("https://data.sec.gov/", "https://www.sec.gov/")) else {"User-Agent": "EquityResearchValidation/1.0"}
        for attempt in range(3):
            try:
                limiter.wait()
                response = requests.get(url, headers=headers, timeout=(10, 45))
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
                if attempt == 2 or (status is not None and status not in {429, 500, 502, 503, 504}):
                    raise
                retry_after = exc.response.headers.get("Retry-After", "1") if status == 429 else "1"
                delay = min(float(retry_after), 30) if retry_after.isdigit() else 1
                time.sleep(delay)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return {"url": url, "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def companyfacts(cik, directory="data/raw/validation"):
    cik = str(cik).zfill(10)
    path = Path(directory) / f"CIK{cik}.json"
    provenance = fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", path)
    return json.loads(path.read_text()), provenance
