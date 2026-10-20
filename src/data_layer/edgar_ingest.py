import os
import threading
from dotenv import load_dotenv
from src.aws.s3_cache import S3Cache
import time
import logging
import requests
from pathlib import Path

# Configure production logging
logger = logging.getLogger("EdgarIngest")

# Production compliant EDGAR User-Agent (Must include app name and valid contact email)
load_dotenv()
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "")

def sec_headers():
    if not EDGAR_USER_AGENT or "@" not in EDGAR_USER_AGENT or "example.com" in EDGAR_USER_AGENT:
        raise ValueError("Set EDGAR_USER_AGENT to your application name and real contact email")
    return {"User-Agent": EDGAR_USER_AGENT}
RATE_LIMIT_PER_SEC = 8  # Safe margin below SEC's 10 req/sec limit


class RateLimiter:
    """
    Token-bucket style rate limiter enforcing strict request spacing 
    across EDGAR API calls to prevent 429 Too Many Requests errors.
    """
    def __init__(self, per_sec: float):
        if per_sec <= 0:
            raise ValueError("Rate limit must be positive")
        self._lock = threading.Lock()
        self.interval = 1.0 / per_sec
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_call = time.monotonic()



# Global rate limiter instance for pipeline use
limiter = RateLimiter(per_sec=RATE_LIMIT_PER_SEC)


def get_submissions(cik: str) -> dict:
    """Fetches full submission history and metadata for a given CIK from SEC EDGAR."""
    headers = sec_headers()
    # Ensure CIK is zero-padded to 10 digits as required by SEC API
    padded_cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    
    limiter.wait()
    response = requests.get(url, headers=headers, timeout=60)
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


def download_filing(cik: str, accession_number: str, primary_document: str, *, cache=None, dest_dir="data/raw/filings") -> str:
    """
    Downloads raw filing text from EDGAR archives, caching it locally 
    to avoid redundant network calls.
    """
    cik = str(int(cik)).zfill(10)
    if not accession_number.replace("-", "").isdigit() or "/" in primary_document or ".." in primary_document:
        raise ValueError("Invalid filing identifiers")
    save_path = Path(dest_dir) / cik / f"{accession_number}.txt"
    cache = cache if cache is not None else S3Cache.from_env()
    key = f"filings/{cik}/{accession_number}/{primary_document}"
    if save_path.exists():
        logger.info(f"File already exists locally: {accession_number}. Skipping download.")
        if cache:
            cache.upload(key, save_path)
        with open(save_path, "r", encoding="utf-8") as f:
            return f.read()

    if cache and cache.restore(key, save_path):
        return save_path.read_text(encoding="utf-8")
    
    acc_no_no_dashes = accession_number.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_no_dashes}/{primary_document}"
    headers = sec_headers()
    
    limiter.wait()
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    if cache:
        cache.upload(key, save_path)
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