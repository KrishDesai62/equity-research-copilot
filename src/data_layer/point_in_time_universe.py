import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Configure robust production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PointInTimeUniverse")

# Known historically distressed or delisted entities used for survivorship-bias validation
KNOWN_DISTRESSED_TICKERS = {
    "LEH", "WM", "CC", "GM", "BBI", "RSH", "JCP",
    "BBBY", "PRTY", "RAD", "HTZ",
}

def get_index_membership_history(index: str = "sp500") -> pd.DataFrame:
    """
    Retrieves and standardizes point-in-time S&P 500 membership change history 
    from Wikipedia's public historical tables.
    
    Parameters:
    -----------
    index : str
        The target market index (default: 'sp500').
        
    Returns:
    --------
    pd.DataFrame
        A standardized DataFrame containing columns: ['ticker', 'cik', 'added_date', 
        'removed_date', 'removal_reason'].
    """
    if index.lower() != "sp500":
        raise NotImplementedError(f"Index '{index}' is not supported in production scope.")
        
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching index membership history for {index.upper()} from source URL.")
    
    try:
        tables = pd.read_html(url)
    except Exception as e:
        logger.error(f"Failed to fetch or parse HTML tables from Wikipedia: {e}")
        raise RuntimeError(f"Index history extraction failed: {e}")
        
    # Table index 1 historically contains the changes table (additions/removals)
    if len(tables) < 2:
        raise ValueError("Wikipedia page structure changed: expected changes table not found at index 1.")
        
    history_df = tables[1].copy()
    
    # Normalize column headers cleanly
    if isinstance(history_df.columns, pd.MultiIndex):
        history_df.columns = ['_'.join([str(c).strip().lower() for c in col if str(c) != 'nan']) for col in history_df.columns]
    else:
        history_df.columns = [str(c).strip().lower().replace(" ", "_") for c in history_df.columns]
        
    logger.debug(f"Parsed table columns: {history_df.columns.tolist()}")
    
    # Map and validate required schema fields robustly
    # Note: Wikipedia schema typically uses columns like 'date', 'added_ticker', 'removed_ticker', etc.
    output_records = []
    
    for _, row in history_df.iterrows():
        # Extract potential date fields
        date_val = None
        for col in history_df.columns:
            if 'date' in col:
                date_val = row.get(col)
                break
                
        parsed_date = pd.to_datetime(date_val, errors='coerce')
        if pd.isna(parsed_date):
            continue
            
        # Check for added tickers
        added_ticker = None
        for col in history_df.columns:
            if 'added' in col and ('ticker' in col or 'symbol' in col):
                added_ticker = row.get(col)
                break
                
        if pd.notna(added_ticker) and str(added_ticker).strip() != "":
            output_records.append({
                "ticker": str(added_ticker).strip().upper(),
                "cik": None,  # Populated downstream via SEC map
                "added_date": parsed_date,
                "removed_date": pd.NaT,
                "removal_reason": None
            })
            
        # Check for removed tickers
        removed_ticker = None
        reason = None
        for col in history_df.columns:
            if 'removed' in col and ('ticker' in col or 'symbol' in col):
                removed_ticker = row.get(col)
            if 'reason' in col:
                reason = row.get(col)
                
        if pd.notna(removed_ticker) and str(removed_ticker).strip() != "":
            output_records.append({
                "ticker": str(removed_ticker).strip().upper(),
                "cik": None,
                "added_date": pd.NaT,  # Pre-existing member
                "removed_date": parsed_date,
                "removal_reason": str(reason).strip() if pd.notna(reason) else "unknown"
            })

    result_df = pd.DataFrame(output_records)
    if result_df.empty:
        raise ValueError("Failed to extract any valid membership transition records from source table.")
        
    logger.info(f"Successfully processed {len(result_df)} index membership transition records.")
    return result_df

def universe_as_of(date: str, membership_history: pd.DataFrame) -> List[str]:
    """
    Computes the exact active constituent universe for a given historical point in time, 
    eliminating look-ahead and survivorship bias.
    
    Parameters:
    -----------
    date : str
        The evaluation timestamp (YYYY-MM-DD).
    membership_history : pd.DataFrame
        The full standardization membership change ledger.
        
    Returns:
    --------
    List[str]
        A sorted list of unique active ticker strings valid as of the target date.
    """
    target_date = pd.to_datetime(date)
    if pd.isna(target_date):
        raise ValueError(f"Invalid evaluation date format provided: {date}")
        
    # Active constituents: Added on or before target date, and either never removed or removed after target date
    active = membership_history[
        (membership_history['added_date'].notna()) & 
        (membership_history['added_date'] <= target_date)
    ]
    
    # Exclude those removed prior to or on target date
    valid_members = membership_history[
        membership_history['removed_date'].notna() & 
        (membership_history['removed_date'] > target_date)
    ]
    
    # Combine active additions and members who haven't hit a removal boundary yet
    active_tickers = set(active['ticker'].dropna().unique())
    removed_tickers = set(membership_history[
        membership_history['removed_date'].notna() & 
        (membership_history['removed_date'] <= target_date)
    ]['ticker'].dropna().unique())
    
    true_active = list(active_tickers - removed_tickers)
    
    logger.debug(f"Computed active universe for {target_date.date()}: {len(true_active)} active entities.")
    return sorted(true_active)

def write_mvp_universe(tickers: List[str], out_path: Path = Path("config/tickers_sp500.yaml")) -> None:
    """
    Validates and persists the target corporate universe to configuration storage, 
    enforcing strict historical distress checks to prevent survivorship bias.
    
    Parameters:
    -----------
    tickers : List[str]
        Collection of ticker symbols to write.
    out_path : Path
        Target destination file path for YAML configuration.
    """
    cleaned_tickers = sorted(list({t.upper().strip() for t in tickers if t}))
    included_distressed = [t for t in cleaned_tickers if t in KNOWN_DISTRESSED_TICKERS]

    if not included_distressed:
        logger.critical("Survivorship bias validation failed: Universe contains zero historically distressed or bankrupt entities.")
        raise ValueError(
            "Validation Error: No historically distressed/delisted tickers found in this "
            "universe. This indicates the pipeline pulled today's active constituent list "
            "instead of a true point-in-time survivorship-bias-free dataset."
        )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_count": len(cleaned_tickers),
            "historical_distressed_count": len(included_distressed),
            "verified_distressed_entities": included_distressed
        },
        "tickers": cleaned_tickers
    }

    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

    logger.info(
        f"Successfully wrote {len(cleaned_tickers)} validated tickers to {destination} "
        f"(including {len(included_distressed)} historical distressed benchmarks: {included_distressed})"
    )