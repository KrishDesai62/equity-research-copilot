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


def build_distress_labels(universe_df, bankruptcy_df, horizon_years=(1, 3),
                          drop_invalid_rows=True, *, observation_end=None):
    """Construct filing-date targets with explicit outcome-coverage requirements.

    Negative labels require reviewed event coverage from coverage_start through
    observed_through (or observation_end). Missing follow-up remains unlabeled.
    Rows on/after a known first bankruptcy are excluded from this pre-event model.
    """
    required = {"cik", "filing_date", "removed_date"}
    if not required <= set(universe_df):
        raise ValueError("Universe requires cik, filing_date and removed_date; fiscal period dates are not valid anchors")
    if not horizon_years or any(not isinstance(years, int) or years <= 0 for years in horizon_years):
        raise ValueError("Horizons must be positive whole years")
    df = universe_df.copy()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    dates = pd.to_datetime(df["filing_date"], errors="raise")
    removed = pd.to_datetime(df["removed_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Public filing dates cannot be missing")
    events = bankruptcy_df.copy()
    events["cik"] = events["cik"].astype(str).str.zfill(10)
    events["filing_date"] = pd.to_datetime(events["filing_date"], errors="raise")
    if events["filing_date"].isna().any():
        raise ValueError("Bankruptcy event dates cannot be missing")
    # A citation is required before a known event can become a positive target.
    if "source_url" not in events or events["source_url"].isna().any() or events["source_url"].astype(str).str.strip().eq("").any():
        raise ValueError("Bankruptcy events require source_url citations")
    first_event = events.groupby("cik")["filing_date"].min()
    event_dates = pd.to_datetime(df["cik"].map(first_event.to_dict()))
    lifecycle = (removed.isna() | (dates <= removed)) & (event_dates.isna() | (dates < event_dates))
    if observation_end is not None:
        observed = pd.Series(pd.Timestamp(observation_end), index=df.index)
    elif "observed_through" in df:
        observed = pd.to_datetime(df["observed_through"], errors="raise")
    else:
        raise ValueError("Provide observed_through or observation_end; negative outcomes cannot be assumed")
    coverage_start = pd.to_datetime(df.get("coverage_start", pd.Series(pd.NaT, index=df.index)))
    reviewed = df.get("event_coverage_reviewed", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1"])
    for years in horizon_years:
        horizon = dates + pd.DateOffset(years=years)
        positive = lifecycle & event_dates.notna() & (event_dates <= horizon) & (event_dates <= observed)
        negative = (lifecycle & reviewed & (coverage_start <= dates) & (horizon <= observed)
                    & (removed.isna() | (horizon <= removed)) & (event_dates.isna() | (event_dates > horizon)))
        target = f"target_distressed_{years}yr"
        df[target] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df.loc[positive, target] = 1
        df.loc[negative, target] = 0
        availability = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        availability.loc[positive] = event_dates[positive]
        availability.loc[negative] = horizon[negative]
        df[f"label_available_date_{years}yr"] = availability
    df["valid_for_training"] = lifecycle & df[[f"target_distressed_{years}yr" for years in horizon_years]].notna().all(axis=1)
    return df.loc[df.valid_for_training].reset_index(drop=True) if drop_invalid_rows else df
