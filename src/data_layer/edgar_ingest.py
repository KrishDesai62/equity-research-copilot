import time
import logging
import requests
from pathlib import Path

# Configure production logging
logger = logging.getLogger("EdgarIngest")

# Production compliant EDGAR User-Agent (Must include app name and valid contact email)
EDGAR_USER_AGENT = "QuantFinApp quantitative_admin@domain.com"
RATE_LIMIT_PER_SEC = 8  # Safe margin below SEC's 10 req/sec limit


class RateLimiter:
    """
    Token-bucket style rate limiter enforcing strict request spacing 
    across EDGAR API calls to prevent 429 Too Many Requests errors.
    """
    def __init__(self, per_sec: float):
        self.interval = 1.0 / per_sec
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()


# Global rate limiter instance for pipeline use
limiter = RateLimiter(per_sec=RATE_LIMIT_PER_SEC)


def get_submissions(cik: str) -> dict:
    """Fetches full submission history and metadata for a given CIK from SEC EDGAR."""
    headers = {'User-Agent': EDGAR_USER_AGENT}
    # Ensure CIK is zero-padded to 10 digits as required by SEC API
    padded_cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def list_filings(cik: str, filing_types: list[str] = None) -> list[dict]:
    """Filters company submission history for target form types (e.g., 10-K, 10-Q)."""
    data = get_submissions(cik)
    recent_filings = data['filings']['recent']
    filings_keep = []
    
    num_fillings = len(recent_filings['accessionNumber'])
    for i in range(num_fillings):
        f_type = recent_filings['form'][i]
        if filing_types and f_type not in filing_types:
            continue
        filings_keep.append({
            'accession_number': recent_filings['accessionNumber'][i],
            'filing_date': recent_filings['filingDate'][i],
            'primary_document': recent_filings['primaryDocument'][i],
            'filing_type': f_type
        })
    return filings_keep


def download_filing(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Downloads raw filing text from EDGAR archives, caching it locally 
    to avoid redundant network calls.
    """
    save_path = Path(f"data/raw/filings/{cik}/{accession_number}.txt")
    if save_path.exists():
        logger.info(f"File already exists locally: {accession_number}. Skipping download.")
        with open(save_path, "r", encoding="utf-8") as f:
            return f.read()

    # Enforce rate limiting before hitting the network
    limiter.wait()
    
    acc_no_no_dashes = accession_number.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_no_dashes}/{primary_document}"
    headers = {'User-Agent': EDGAR_USER_AGENT}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    logger.info(f"Successfully downloaded and cached filing: {accession_number}")
    return response.text


if __name__ == "__main__":
    target_cik = "0000320193"  # Apple Inc. CIK example
    print(f"Fetching filings for CIK: {target_cik}")
    
    try:
        filings = list_filings(target_cik, filing_types=["10-K", "10-Q"])
        for f in filings:
            try:
                download_filing(target_cik, f['accession_number'], f['primary_document'])
            except Exception as e:
                logger.error(f"Failed to download filing {f['accession_number']}: {e}")
        print("EDGAR ingestion complete!")
    except Exception as e:
        logger.error(f"Pipeline execution failed for CIK {target_cik}: {e}")