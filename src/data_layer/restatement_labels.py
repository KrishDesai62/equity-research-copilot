"""
Track B ground truth: SEC AAER enforcement actions + 8-K Item 4.02
restatements. Maintained as SEPARATE binary classification target columns.
"""
from pathlib import Path
import pandas as pd


def load_aaer_releases(source_path: str) -> pd.DataFrame:
    """
    Loads SEC Accounting and Auditing Enforcement Releases (AAERs) from a curated,
    version-controlled CSV (data/aaer_releases.csv) mapping company name,
    CIK, exact release date, and source reference URLs.
    
    Source endpoint tracking: https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"AAER records file not found: {path}")
    
    df = pd.read_csv(path, dtype={"cik": str})
    
    required_cols = {"company_name", "cik", "release_date", "source_url", "aaer_number"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"aaer_releases.csv missing required columns: {missing}")

    df["cik"] = df["cik"].str.zfill(10)
    bad_cik = df[~df["cik"].str.match(r"^\d{10}$")]
    if not bad_cik.empty:
        raise ValueError(f"Invalid CIK format in AAER rows: {bad_cik['company_name'].tolist()}")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    bad_dates = df[df["release_date"].isna()]
    if not bad_dates.empty:
        raise ValueError(f"Unparseable release_date in AAER rows: {bad_dates['company_name'].tolist()}")

    if df["source_url"].isna().any() or (df["source_url"].str.strip() == "").any():
        raise ValueError("Every AAER row must have a verifiable source_url citation.")

    return df.sort_values("release_date").reset_index(drop=True)


def find_item_402_filings(source_path: str) -> pd.DataFrame:
    """
    Loads curated 8-K Item 4.02 filings data (Non-reliance on previously issued 
    financial statements or a related audit report or completed interim review)
    from a version-controlled CSV (data/item_402_filings.csv).
    
    Columns expected: cik, filing_date, accession_number, source_url
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Item 4.02 filings file not found: {path}")

    df = pd.read_csv(path, dtype={"cik": str})
    
    required_cols = {"cik", "filing_date", "accession_number", "source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"item_402_filings.csv missing required columns: {missing}")

    df["cik"] = df["cik"].str.zfill(10)
    bad_cik = df[~df["cik"].str.match(r"^\d{10}$")]
    if not bad_cik.empty:
        raise ValueError(f"Invalid CIK format in Item 4.02 rows: {bad_cik['cik'].tolist()}")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    bad_dates = df[df["filing_date"].isna()]
    if not bad_dates.empty:
        raise ValueError(f"Unparseable filing_date in Item 4.02 rows: {bad_dates['accession_number'].tolist()}")

    if df["source_url"].isna().any() or (df["source_url"].str.strip() == "").any():
        raise ValueError("Every Item 4.02 row must have a verifiable source_url citation.")

    return df.sort_values("filing_date").reset_index(drop=True)


def build_manipulation_labels(
    universe_df: pd.DataFrame,
    aaer_df: pd.DataFrame,
    item402_df: pd.DataFrame,
    lookforward_window_years: float = 3.0,
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    """
    Produces one row per (cik, fiscal_quarter) with TWO explicitly separate target columns:
      - target_aaer_enforcement_flag: bool (Hard enforcement action / fraud indicator)
      - target_restatement_flag: bool (8-K Item 4.02 non-reliance restatement event)

    Enforces temporal split discipline: flags events occurring within the designated 
    lookforward window after the fiscal quarter date, while filtering out post-removal 
    rows using 'removed_date' from the point-in-time universe definitions.
    """
    if "removed_date" not in universe_df.columns:
        raise ValueError(
            "universe_df must have a 'removed_date' column (from point_in_time_universe.py) "
            "to safely track active corporate lifecycles."
        )
    if "date" not in universe_df.columns:
        raise ValueError("universe_df must have a 'date' column representing the per-quarter row date.")

    df = universe_df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["date"] = pd.to_datetime(df["date"])
    df["removed_date"] = pd.to_datetime(df["removed_date"])

    # Lifecycle validity flag
    df["valid_for_training"] = df["removed_date"].isna() | (df["date"] <= df["removed_date"])

    # Clean and standardize event tables
    aaer = aaer_df.copy()
    aaer["cik"] = aaer["cik"].astype(str).str.zfill(10)
    aaer["release_date"] = pd.to_datetime(aaer["release_date"])

    item402 = item402_df.copy()
    item402["cik"] = item402["cik"].astype(str).str.zfill(10)
    item402["filing_date"] = pd.to_datetime(item402["filing_date"])

    # Merge dataset with AAER events (using left join to preserve all company quarters)
    merged = pd.merge(
        df,
        aaer[["cik", "release_date", "aaer_number"]].rename(
            columns={"release_date": "aaer_release_date", "aaer_number": "matched_aaer_number"}
        ),
        on="cik",
        how="left",
    )

    # Merge dataset with Item 4.02 restatement events
    merged = pd.merge(
        merged,
        item402[["cik", "filing_date", "accession_number"]].rename(
            columns={"filing_date": "item402_filing_date", "accession_number": "matched_item402_accession"}
        ),
        on="cik",
        how="left",
    )

    window_days = lookforward_window_years * 365.25

    # 1. Calculate AAER Enforcement Target Flag
    days_to_aaer = (merged["aaer_release_date"] - merged["date"]).dt.total_seconds() / (24 * 3600)
    merged["target_aaer_enforcement_flag"] = (
        merged["aaer_release_date"].notna()
        & (days_to_aaer >= 0)
        & (days_to_aaer <= window_days)
    ).astype(int)

    # 2. Calculate Restatement Target Flag (Item 4.02)
    days_to_restatement = (merged["item402_filing_date"] - merged["date"]).dt.total_seconds() / (24 * 3600)
    merged["target_restatement_flag"] = (
        merged["item402_filing_date"].notna()
        & (days_to_restatement >= 0)
        & (days_to_restatement <= window_days)
    ).astype(int)

    # Clean up intermediary temporary date columns
    merged = merged.drop(columns=["aaer_release_date", "matched_aaer_number", "item402_filing_date", "matched_item402_accession"])

    if drop_invalid_rows:
        n_before = len(merged)
        merged = merged[merged["valid_for_training"]].reset_index(drop=True)
        n_dropped = n_before - len(merged)
        if n_dropped:
            print(f"Track B: Dropped {n_dropped} rows past a company's structural removal date.")

    return merged