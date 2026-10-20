from src.aws.s3_cache import S3Cache
from src.data_layer.edgar_ingest import sec_headers, limiter
import logging
import zipfile
from pathlib import Path
from typing import List, Optional
import pandas as pd
import requests

# Configure robust production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SECBulkXBRLIngest")

# SEC DERA Bulk Data URLs and Headers
SEC_HEADERS = {
    "User-Agent": "QuantitativeResearchPipeline institutional-risk@domain.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# Essential XBRL tags mapped for Altman Z-Score and Beneish M-Score models
TARGET_XBRL_TAGS = {
    "Assets",
    "Liabilities",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "NetIncomeLoss",
    "AccountsReceivableNetCurrent",
    "StockholdersEquity",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "RetainedEarningsAccumulatedDeficit",
    "OperatingIncomeLoss",
    "SalesRevenueNet",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "CostOfRevenue",
    "PropertyPlantAndEquipmentNet",
    "DepreciationDepletionAndAmortization",
    "Depreciation",
    "SellingGeneralAndAdministrativeExpense",
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "NetCashProvidedByUsedInOperatingActivities",
}


def download_quarterly_zip(year: int, quarter: int, dest_dir: Path, *, cache=None) -> Path:
    """Downloads and caches quarterly bulk financial statement ZIP datasets from SEC DERA.

    Parameters:
    -----------
    year : int
        The reporting year (e.g., 2024).
    quarter : int
        The calendar quarter (1 through 4).
    dest_dir : Path
        The target local directory for caching datasets.

    Returns:
    --------
    Path
        The verified local file path to the downloaded ZIP archive.
    """
    if quarter not in (1, 2, 3, 4) or year < 2009:
        raise ValueError("Specify a supported year and quarter (1-4)")
    cache = cache if cache is not None else S3Cache.from_env()
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)

    file_name = f"{year}q{quarter}.zip"
    save_path = destination / file_name

    if save_path.exists() and save_path.stat().st_size > 0:
        logger.info(
            f"Cached file already exists: {file_name}. Skipping download."
        )
        if not zipfile.is_zipfile(save_path):
            raise ValueError("Local cache contains an invalid ZIP")
        if cache:
            cache.upload(f"xbrl/{file_name}", save_path)
        return save_path

    if cache and cache.restore(f"xbrl/{file_name}", save_path):
        if not zipfile.is_zipfile(save_path):
            raise ValueError("S3 cache contains an invalid ZIP")
        return save_path

    url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{file_name}"
    logger.info(f"Initiating download for bulk financial dataset: {file_name}")

    try:
        limiter.wait()
        response = requests.get(url, headers=sec_headers(), stream=True, timeout=60)
        response.raise_for_status()

        temp_path = save_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if not zipfile.is_zipfile(temp_path):
            raise ValueError("SEC returned an invalid ZIP")
        temp_path.rename(save_path)
        if cache:
            cache.upload(f"xbrl/{file_name}", save_path)
        logger.info(f"Successfully downloaded and cached: {file_name}")

    except Exception as e:
        logger.error(
            f"Failed to download bulk ZIP for {year}Q{quarter}: {e}"
        )
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"SEC DERA download failure: {e}")

    return save_path


def parse_zip_to_dataframe(zip_path: Path, *, ciks=None) -> pd.DataFrame:
    """Parses submission metadata and numerical XBRL facts from a quarterly DERA ZIP archive,

    joining them securely on accession numbers (adsh) while enforcing strict schema types.

    Parameters:
    -----------
    zip_path : Path
        The local path to the quarterly ZIP file.

    Returns:
    --------
    pd.DataFrame
        A tidy DataFrame containing columns: ['cik', 'adsh', 'tag', 'value', 'filing_date',
        'fiscal_period_end', 'fy', 'fp'].
    """
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"Quarterly ZIP archive not found: {path}")

    logger.info(f"Parsing bulk archive contents from {path.name}")

    try:
        with zipfile.ZipFile(path, "r") as z:
            # 1. Read submission metadata (sub.txt)
            with z.open("sub.txt") as sub_file:
                sub_df = pd.read_csv(
                    sub_file,
                    sep="\t",
                    dtype=str,
                    usecols=lambda c: c
                    in ["adsh", "cik", "name", "filed", "period", "fy", "fp", "form"],
                )

            # Standardize and rename columns to internal pipeline standards
            sub_df = sub_df.rename(
                columns={
                    "filed": "filing_date",
                    "period": "fiscal_period_end",
                }
            )
            sub_df["cik"] = sub_df["cik"].str.zfill(10)
            if ciks is not None:
                sub_df = sub_df.loc[sub_df.cik.isin({str(c).zfill(10) for c in ciks})].copy()
            sub_df["filing_date"] = pd.to_datetime(
                sub_df["filing_date"], errors="coerce"
            )
            sub_df["fiscal_period_end"] = pd.to_datetime(
                sub_df["fiscal_period_end"], errors="coerce"
            )

            # 2. Read raw numeric facts (num.txt)
            with z.open("num.txt") as num_file:
                parts = pd.read_csv(
                    num_file,
                    sep="\t",
                    dtype=str,
                    usecols=lambda c: c in ["adsh", "tag", "version", "segments", "value", "ddate", "qtrs", "uom", "coreg"],
                    chunksize=200_000,
                )
                selected = []
                accessions = set(sub_df.adsh)
                for part in parts:
                    selected.append(part.loc[part.adsh.isin(accessions) & part.tag.isin(TARGET_XBRL_TAGS)])
                num_df = pd.concat(selected, ignore_index=True)

        # Filter down memory footprint early by keeping only required analytics tags
        num_df = num_df[num_df["tag"].isin(TARGET_XBRL_TAGS)].copy()
        num_df["value"] = pd.to_numeric(num_df["value"], errors="coerce")
        num_df = num_df.dropna(subset=["value"])

        # 3. Join numerical facts with submission metadata on accession number (adsh)
        merged_df = pd.merge(num_df, sub_df, on="adsh", how="inner")

        if "form" in merged_df:
            merged_df = merged_df[merged_df["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])]
        if "ddate" in merged_df:
            fact_dates = pd.to_datetime(merged_df["ddate"], errors="coerce")
            merged_df = merged_df[fact_dates == merged_df["fiscal_period_end"]]
        if "uom" in merged_df:
            merged_df = merged_df[merged_df["uom"] == "USD"]
        if "coreg" in merged_df:
            merged_df = merged_df[merged_df["coreg"].isna() | (merged_df["coreg"] == "")]
        if "segments" in merged_df:
            merged_df = merged_df[merged_df.segments.isna() | merged_df.segments.str.strip().eq("")]
        if "version" in merged_df:
            merged_df = merged_df[merged_df.version.str.startswith("us-gaap/", na=False)]
        if "qtrs" in merged_df:
            durations = pd.to_numeric(merged_df["qtrs"], errors="coerce")
            expected = pd.to_numeric(merged_df["fp"].str.replace("Q", "", regex=False), errors="coerce").fillna(4)
            merged_df = merged_df[(durations == 0) | (durations == expected)]
        merged_df["fiscal_quarter"] = merged_df["fiscal_period_end"].dt.to_period("Q").astype(str)

        logger.info(
            f"Successfully parsed and joined {len(merged_df):,} facts from {path.name}"
        )
        return merged_df

    except Exception as e:
        logger.error(f"Failed to parse ZIP archive {path.name}: {e}")
        raise RuntimeError(f"XBRL ZIP processing failed: {e}")

def build_feature_table(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates multiple quarterly DataFrames and pivots structural XBRL metrics

    into a clean multi-tag financial table, retaining explicit separation between
    filing dates and fiscal period ends to eliminate look-ahead bias.

    Parameters:
    -----------
    dataframes : List[pd.DataFrame]
        A list of tidy quarterly DataFrames generated via parse_zip_to_dataframe.

    Returns:
    --------
    pd.DataFrame
        A wide-format feature table indexed by CIK and filing metadata, with columns
        representing distinct XBRL financial metrics.
    """
    if not dataframes:
        raise ValueError(
            "Cannot build feature table from an empty DataFrame list."
        )

    logger.info(
        f"Concatenating and pivoting {len(dataframes)} quarterly datasets."
    )
    combined_df = pd.concat(dataframes, ignore_index=True)

    if combined_df.empty:
        logger.warning(
            "Combined quarterly dataset is empty after concatenation."
        )
        return pd.DataFrame()

    # Pivot table to achieve one row per financial statement release entity instance,
    # mapping specific XBRL tags to individual columns.
    pivoted_df = combined_df.pivot_table(
        index=[
            "cik",
            "name",
            "adsh",
            "filing_date",
            "fiscal_period_end",
            "fy",
            "fp",
        ],
        columns="tag",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Flatten multi-index column names if any result from pivoting
    pivoted_df.columns.name = None
    pivoted_df["fiscal_quarter"] = pd.to_datetime(pivoted_df["fiscal_period_end"]).dt.to_period("Q").astype(str)

    logger.info(
        f"Successfully constructed feature table with {len(pivoted_df):,} rows and {len(pivoted_df.columns)} features."
    )
    return pivoted_df
