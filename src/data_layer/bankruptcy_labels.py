"""
Track A ground truth: Chapter 11 bankruptcy filings, cross-referenced to
CIK/ticker. Public record (PACER has a small per-page fee; some free
proxies/aggregators exist — search before committing to PACER scraping).

This is the labor-intensive part of the MVP the original spec
underestimates — budget real time here, more than for the model code.
"""
import difflib
import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {"company_name", "ticker", "cik", "filing_date", "source_url"}


def load_bankruptcy_records(source_path: str) -> pd.DataFrame:
    """
    Loads Chapter 11 bankruptcy filing records from a curated,
    version-controlled CSV (data/bankruptcy_records.csv) — mapping
    company name, CIK, exact filing date, and a source citation per row.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Bankruptcy records file not found: {path}")
    df = pd.read_csv(path, dtype={"cik": str})

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"bankruptcy_records.csv missing columns: {missing_cols}")

    df["cik"] = df["cik"].str.zfill(10)
    bad_cik = df[~df["cik"].str.match(r"^\d{10}$")]
    if not bad_cik.empty:
        raise ValueError(f"Invalid CIK format in rows: {bad_cik['company_name'].tolist()}")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    bad_dates = df[df["filing_date"].isna()]
    if not bad_dates.empty:
        raise ValueError(f"Unparseable filing_date in rows: {bad_dates['company_name'].tolist()}")

    df["ticker"] = df["ticker"].str.upper().str.strip()
    dupes = df[df["ticker"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(f"Duplicate tickers found: {dupes['ticker'].unique().tolist()}")

    if df["source_url"].isna().any() or (df["source_url"].str.strip() == "").any():
        raise ValueError("Every row must have a source_url — no uncited bankruptcy dates.")

    return df.sort_values("filing_date").reset_index(drop=True)


def match_to_cik(bankruptcy_df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-checks bankruptcy_df's CIKs against universe_df and flags
    mismatches. Fuzzy name matching is only a fallback SUGGESTION for
    manual review — never auto-accepted.
    """
    if "cik" not in universe_df.columns:
        raise ValueError("universe_df must have a 'cik' column to match against.")

    universe_ciks = set(universe_df["cik"].astype(str).str.zfill(10))
    df = bankruptcy_df.copy()

    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["cik_verified"] = df["cik"].isin(universe_ciks)
    df["match_note"] = ""

    unverified = df[~df["cik_verified"]]
    if not unverified.empty:
        name_col = "company_name" if "company_name" in universe_df.columns else "ticker"
        universe_names = universe_df[name_col].astype(str).tolist()

        for idx, row in unverified.iterrows():
            close = difflib.get_close_matches(
                row["company_name"], universe_names, n=1, cutoff=0.6
            )
            if close:
                match_row = universe_df[universe_df[name_col] == close[0]].iloc[0]
                note = (
                    f"CIK {row['cik']} not found in universe. Closest name "
                    f"match: '{close[0]}' (CIK {match_row['cik']}) — "
                    f"REVIEW MANUALLY, do not auto-accept."
                )
            else:
                note = (
                    f"CIK {row['cik']} not found in universe, and no "
                    f"close name match either — likely means this company "
                    f"isn't in your point-in-time universe yet. Add it in "
                    f"point_in_time_universe.py or double-check the CIK "
                    f"in bankruptcy_records.csv."
                )
            df.loc[idx, "match_note"] = note

    return df


def build_distress_labels(
    universe_df: pd.DataFrame,
    bankruptcy_df: pd.DataFrame,
    horizon_years: list[int] = [1, 3],
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    """
    Produce one row per (cik, fiscal_quarter) with a binary label per
    horizon: did this company file Chapter 11 within `horizon_years` of
    this quarter's date?

    Also flags/drops rows that fall after a company's last known active
    date, so a company that got acquired (stopped filing for reasons
    unrelated to distress) doesn't get silently labeled "not distressed"
    for hypothetical future quarters it never actually had.

    Requires universe_df to carry a 'removed_date' column (from
    point_in_time_universe.py — null/NaT for still-active companies).
    If that column is missing, this raises rather than guessing, since
    silently skipping this check reintroduces the exact bug it exists
    to catch.

    Args:
        drop_invalid_rows: if True (default), rows past a company's
            removed_date are dropped entirely. If False, they're kept
            but flagged via the added 'valid_for_training' column, so
            you can inspect them before deciding.
    """
    if "removed_date" not in universe_df.columns:
        raise ValueError(
            "universe_df must have a 'removed_date' column (from "
            "point_in_time_universe.py) so post-removal quarters can be "
            "flagged instead of silently labeled 'not distressed'."
        )

    df = universe_df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)

    bankruptcies = bankruptcy_df.copy()
    bankruptcies["cik"] = bankruptcies["cik"].astype(str).str.zfill(10)

    if "date" not in df.columns:
        raise ValueError("universe_df must have a 'date' column (per-quarter row date).")
    df["date"] = pd.to_datetime(df["date"])
    df["removed_date"] = pd.to_datetime(df["removed_date"])
    bankruptcies["filing_date"] = pd.to_datetime(bankruptcies["filing_date"])

    # Flag rows that fall after this company left the universe (acquired,
    # delisted for non-distress reasons, etc.) — these are not valid
    # negative-label training rows.
    df["valid_for_training"] = df["removed_date"].isna() | (df["date"] <= df["removed_date"])

    merged = pd.merge(
        df,
        bankruptcies[["cik", "filing_date"]].rename(columns={"filing_date": "bankruptcy_filing_date"}),
        on="cik",
        how="left",
    )

    for years in horizon_years:
        horizon_days = years * 365.25
        target_col = f"target_distressed_{years}yr"

        days_to_bankruptcy = (
            merged["bankruptcy_filing_date"] - merged["date"]
        ).dt.total_seconds() / (24 * 3600)

        merged[target_col] = (
            merged["bankruptcy_filing_date"].notna()
            & (days_to_bankruptcy >= 0)
            & (days_to_bankruptcy <= horizon_days)
        ).astype(int)

    merged = merged.drop(columns=["bankruptcy_filing_date"])

    if drop_invalid_rows:
        n_before = len(merged)
        merged = merged[merged["valid_for_training"]].reset_index(drop=True)
        n_dropped = n_before - len(merged)
        if n_dropped:
            print(f"Dropped {n_dropped} rows past a company's removal date.")

    return merged