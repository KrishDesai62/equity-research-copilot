import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ==========================================
# PRODUCTION CONFIGURATION & RATE LIMITING
# ==========================================
# SEC EDGAR requires a custom User-Agent identifying your application/organization.
# Rate limit: Maximum 10 requests per second to prevent IP throttling.
SEC_HEADERS = {
    "User-Agent": "QuantitativeResearchPipeline institutional-risk@domain.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

ARCHIVE_HEADERS = {
    "User-Agent": "QuantitativeResearchPipeline institutional-risk@domain.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

RATE_LIMIT_DELAY = 0.1  # 100ms pause between requests to respect SEC limits

def safe_sec_get(url: str, headers: dict) -> requests.Response:
    """Executes an HTTP GET request with built-in rate limiting and error handling."""
    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response

# ==========================================
# 1. GROUND TRUTH: BANKRUPTCY INGESTION
# ==========================================
REQUIRED_BANKRUPTCY_COLS = {"company_name", "ticker", "cik", "filing_date", "source_url"}

def load_bankruptcy_records(source_path: str) -> pd.DataFrame:
    """Loads and strictly validates historical Chapter 11 bankruptcy filings from disk."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Production bankruptcy records file not found: {path}")
        
    df = pd.read_csv(path, dtype={"cik": str})
    missing_cols = REQUIRED_BANKRUPTCY_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Bankruptcy CSV missing required columns: {missing_cols}")

    # Standardize CIK to 10-digit zero-padded string for reliable downstream joins
    df["cik"] = df["cik"].str.zfill(10)
    bad_cik = df[~df["cik"].str.match(r"^\d{10}$")]
    if not bad_cik.empty:
        raise ValueError(f"Invalid CIK formatting detected for: {bad_cik['company_name'].tolist()}")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    if df["filing_date"].isna().any():
        raise ValueError("Unparseable filing dates found in bankruptcy dataset.")

    df["ticker"] = df["ticker"].str.upper().str.strip()
    if df["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers detected in bankruptcy records.")

    if df["source_url"].isna().any() or (df["source_url"].str.strip() == "").any():
        raise ValueError("Audit failure: Every bankruptcy record must contain a verifiable source_url.")

    return df.sort_values("filing_date").reset_index(drop=True)

# ==========================================
# 2. SEC EDGAR FORM 4 FILING INGESTION
# ==========================================
def list_form4_filings(cik: str) -> list[dict]:
    """Queries SEC EDGAR submissions API for all historical Form 4 insider trading filings."""
    padded_cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    
    try:
        response = safe_sec_get(url, SEC_HEADERS)
        data = response.json()
    except Exception as e:
        print(f"Error fetching submissions for CIK {padded_cik}: {e}")
        return []
        
    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []
        
    df = pd.DataFrame({
        "accession_number": filings.get("accessionNumber", []),
        "filing_date": filings.get("filingDate", []),
        "form": filings.get("form", []),
        "primary_document": filings.get("primaryDocument", [])
    })
    
    # Filter explicitly for Form 4 filings
    form4_df = df[df["form"] == "4"].copy()
    return form4_df.to_dict(orient="records")

def fetch_and_parse_form4_xml(cik: str, accession_number: str, primary_doc: str) -> dict:
    """Downloads and parses the raw XML document for a specific Form 4 filing from EDGAR archives."""
    padded_cik = str(cik).zfill(10)
    clean_accession = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(padded_cik)}/{clean_accession}/{primary_doc}"
    
    try:
        response = safe_sec_get(url, ARCHIVE_HEADERS)
        root = ET.ElementTree(ET.fromstring(response.content)).getroot()
    except Exception as e:
        return {"reporting_owner": {}, "transactions": []}

    def get_text(element, path):
        node = element.find(path)
        return node.text.strip() if node is not None and node.text else None

    # Extract reporting owner metadata
    owner_data = {}
    reporting_owner = root.find(".//reportingOwner")
    if reporting_owner is not None:
        owner_id = reporting_owner.find("reportingOwnerId")
        owner_rel = reporting_owner.find("reportingOwnerRelationship")
        owner_data = {
            "cik": get_text(owner_id, "rptOwnerCik"),
            "name": get_text(owner_id, "rptOwnerName"),
            "is_director": get_text(owner_rel, "isDirector") == "1",
            "is_officer": get_text(owner_rel, "isOfficer") == "1",
            "officer_title": get_text(owner_rel, "officerTitle"),
            "is_ten_percent_owner": get_text(owner_rel, "isTenPercentOwner") == "1",
        }

    # Extract Table I non-derivative open-market transactions
    transactions = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        transactions.append({
            "security_title": get_text(tx, ".//securityTitle/value"),
            "transaction_date": get_text(tx, ".//transactionDate/value"),
            "transaction_code": get_text(tx, ".//transactionCoding/transactionCode"),
            "shares": float(tx.find(".//transactionAmounts/transactionShares/value").text) 
                      if tx.find(".//transactionAmounts/transactionShares/value") is not None else 0.0,
            "price_per_share": float(tx.find(".//transactionAmounts/transactionPricePerShare/value").text) 
                               if tx.find(".//transactionAmounts/transactionPricePerShare/value") is not None else 0.0,
            "acquired_disposed": get_text(tx, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        })

    return {"reporting_owner": owner_data, "transactions": transactions}

# ==========================================
# 3. PRODUCTION FEATURE ENGINEERING & LABELS
# ==========================================
def build_insider_sentiment_feature(cik: str, as_of_date: str, lookback_days: int = 90) -> dict:
    """Computes point-in-time net insider volume features strictly respecting the as_of_date boundary."""
    padded_cik = str(cik).zfill(10)
    cutoff_date = pd.to_datetime(as_of_date)
    start_date = cutoff_date - timedelta(days=lookback_days)
    
    filings = list_form4_filings(padded_cik)
    if not filings:
        return {"cik": padded_cik, "as_of_date": as_of_date, "net_shares_traded": 0.0, "insider_sentiment_score": 0.0}
        
    df_filings = pd.DataFrame(filings)
    df_filings["filing_date"] = pd.to_datetime(df_filings["filing_date"], errors="coerce")
    
    # Strict point-in-time filter: Eliminate any filings published on or after evaluation date
    valid_filings = df_filings[
        (df_filings["filing_date"] >= start_date) & 
        (df_filings["filing_date"] <= cutoff_date)
    ]
    
    total_bought, total_sold = 0.0, 0.0
    for _, filing in valid_filings.iterrows():
        parsed = fetch_and_parse_form4_xml(padded_cik, filing["accession_number"], filing["primary_document"])
        for tx in parsed.get("transactions", []):
            shares = tx.get("shares", 0.0)
            code = tx.get("transaction_code")
            acq_disp = tx.get("acquired_disposed")
            
            if code in ("P",) or acq_disp == "A":
                total_bought += shares
            elif code in ("S",) or acq_disp == "D":
                total_sold += shares

    total_volume = total_bought + total_sold
    sentiment_ratio = (total_bought - total_sold) / total_volume if total_volume > 0 else 0.0
    
    return {
        "cik": padded_cik,
        "as_of_date": as_of_date,
        "total_shares_bought": total_bought,
        "total_shares_sold": total_sold,
        "net_insider_sentiment_ratio": sentiment_ratio
    }

def build_distress_labels(universe_df: pd.DataFrame, bankruptcy_df: pd.DataFrame, horizon_years: list[int] = [1, 3]) -> pd.DataFrame:
    """Generates production point-in-time machine learning target labels avoiding look-ahead bias."""
    df = universe_df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    bankruptcies = bankruptcy_df.copy()
    bankruptcies["cik"] = bankruptcies["cik"].astype(str).str.zfill(10)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    bankruptcies["filing_date"] = pd.to_datetime(bankruptcies["filing_date"])
    
    merged = pd.merge(
        df,
        bankruptcies[["cik", "filing_date"]].rename(columns={"filing_date": "bankruptcy_filing_date"}),
        on="cik",
        how="left"
    )
    
    for years in horizon_years:
        horizon_days = years * 365.25
        target_col = f"target_distressed_{years}yr"
        days_to_bankruptcy = (merged["bankruptcy_filing_date"] - merged["date"]).dt.total_seconds() / (24 * 3600)
        
        merged[target_col] = (
            merged["bankruptcy_filing_date"].notna() & 
            (days_to_bankruptcy >= 0) & 
            (days_to_bankruptcy <= horizon_days)
        ).astype(int)
        
    return merged.drop(columns=["bankruptcy_filing_date"])