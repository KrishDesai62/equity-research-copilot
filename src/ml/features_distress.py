"""
Track A features: Altman Z-score components computed from XBRL data,
plus an optional RAG-derived sentiment score from Risk Factors text.

MVP scope: this file. Track A is the only track in the MVP.
"""
import pandas as pd


def compute_altman_components(xbrl_row: dict) -> dict:
    """
    Computes standard Altman Z-score component ratios from a parsed XBRL financial row dictionary.
    Returns NaN for ratios missing required tags or encountering division-by-zero errors.
    """
    # Helper function to safely extract and convert values across common XBRL tag variants
    def get_val(keys):
        for k in keys:
            if k in xbrl_row and xbrl_row[k] is not None:
                try:
                    return float(xbrl_row[k])
                except (ValueError, TypeError):
                    continue
        return float('nan')

    # Map raw XBRL concepts (with common alternative tag names)
    total_assets = get_val(["Assets", "us-gaap:Assets"])
    current_assets = get_val(["AssetsCurrent", "us-gaap:AssetsCurrent"])
    current_liabilities = get_val(["LiabilitiesCurrent", "us-gaap:LiabilitiesCurrent"])
    total_liabilities = get_val(["Liabilities", "us-gaap:Liabilities"])
    retained_earnings = get_val(["RetainedEarningsAccumulatedDeficit", "us-gaap:RetainedEarningsAccumulatedDeficit"])
    ebit = get_val(["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes", "us-gaap:OperatingIncomeLoss"])
    sales = get_val(["Revenues", "SalesRevenueNet", "us-gaap:SalesRevenueNet"])
    market_value_equity = get_val(["MarketValueOfEquity", "MarketValueEquity", "market_cap"])

    # Compute the 5 Altman components safely with zero/NaN guards
    components = {}
    
    components["wc_to_ta"] = (current_assets - current_liabilities) / total_assets if total_assets else float('nan')
    components["re_to_ta"] = retained_earnings / total_assets if total_assets else float('nan')
    components["ebit_to_ta"] = ebit / total_assets if total_assets else float('nan')
    components["mve_to_tl"] = market_value_equity / total_liabilities if total_liabilities else float('nan')
    components["sales_to_ta"] = sales / total_assets if total_assets else float('nan')

    return components
    


import logging
from typing import Dict, Callable, Optional

logger = logging.getLogger(__name__)

def add_rag_sentiment_feature(row: dict, retrieve_fn: Optional[Callable] = None) -> dict:
    """
    Production-grade RAG sentiment feature extractor.
    Queries the vector database for Item 1A (Risk Factors) AS OF the row's filing date,
    computes Loughran-McDonald negative financial tone density, and injects it into the row.
    """
    # 1. Initialize default/fallback feature value to prevent pipeline crashes
    row["risk_sentiment_score"] = float("nan")
    
    if retrieve_fn is None:
        return row

    # 2. Extract required metadata safely
    ticker = row.get("ticker")
    filing_date = row.get("filing_date")
    
    if not ticker or not filing_date:
        logger.warning(f"Missing ticker or filing_date for row. Skipping sentiment extraction.")
        return row

    # 3. Formulate targeted query for Risk Factors semantics
    query_text = (
        "material adverse effects financial condition liquidity risks "
        "operational uncertainties regulatory pressures market risks"
    )

    try:
        # Execute vector store retrieval filtered by section
        retrieved_docs = retrieve_fn(
            query=query_text,
            ticker=ticker,
            section="Item 1A",
            k=5
        )
    except Exception as e:
        logger.error(f"Vector store retrieval failed for {ticker} on {filing_date}: {e}")
        return row

    if not retrieved_docs:
        return row

    # 4. Strict Point-in-Time Filtering (Anti-Leakage Guard)
    # Ensure chunks originate ONLY from filings published on or before this row's filing date
    valid_texts = []
    for doc in retrieved_docs:
        meta = doc.get("metadata", {})
        doc_filing_date = meta.get("filing_date")
        
        if doc_filing_date and doc_filing_date <= filing_date:
            text_content = doc.get("text", "")
            if text_content:
                valid_texts.append(text_content)

    if not valid_texts:
        # If no valid historical text exists prior to/on filing date, set neutral baseline score
        row["risk_sentiment_score"] = 0.0
        return row

    combined_text = " ".join(valid_texts)

    # 5. Compute Quantitative Loughran-McDonald Negative Tone Score
    row["risk_sentiment_score"] = compute_loughran_mcdonald_score(combined_text)
    
    return row


def compute_loughran_mcdonald_score(text: str) -> float:
    """
    Computes financial negative sentiment density based on the Loughran-McDonald master dictionary.
    Production standard: Ratio of Negative Word Count to Total Word Count.
    """
    # Core financial distress & risk vocabulary subset from the LM Negative Master Dictionary
    lm_negative_lexicon = {
        "loss", "losses", "adverse", "fail", "failed", "failure", "failures",
        "decline", "declined", "declining", "decrease", "decreased", "decreasing",
        "default", "defaults", "bankruptcy", "bankruptcies", "uncertainty",
        "uncertainties", "litigation", "litigations", "restatement", "restatements",
        "impairment", "impairments", "writeoff", "writeoffs", "deficiency",
        "deficiencies", "difficulties", "restructuring", "vulnerability",
        "deteriorate", "deteriorated", "deterioration", "termination", "penalties"
    }

    # Normalize and tokenize text
    words = text.lower().split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0

    # Count occurrences of negative financial terms
    negative_count = sum(1 for word in words if word in lm_negative_lexicon)
    
    # Return normalized density score
    return float(negative_count / total_words)

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_feature_table(xbrl_df: pd.DataFrame, labels_df: pd.DataFrame, retrieve_fn=None) -> pd.DataFrame:
    """
    Joins XBRL-derived Altman components, optional RAG sentiment scores,
    and ground-truth distress labels into a unified modeling table.
    One row per (cik, fiscal_quarter), optimized for train_distress.py.
    """
    if xbrl_df.empty or labels_df.empty:
        logger.warning("Input xbrl_df or labels_df is empty. Returning empty feature table.")
        return pd.DataFrame()

    processed_rows = []

    # 1. Iterate through each financial filing row in the dataset
    for _, xbrl_row in xbrl_df.iterrows():
        row_dict = xbrl_row.to_dict()
        cik = row_dict.get("cik")
        fiscal_quarter = row_dict.get("fiscal_quarter")
        filing_date = row_dict.get("filing_date")

        if not cik or not fiscal_quarter:
            continue

        # 2. Compute quantitative Altman Z-score components
        altman_features = compute_altman_components(row_dict)

        # 3. Extract optional RAG-derived sentiment score safely
        sentiment_features = add_rag_sentiment_feature(row_dict, retrieve_fn=retrieve_fn)

        # 4. Merge all feature sets into a unified feature dictionary
        combined_row = {
            "cik": cik,
            "fiscal_quarter": fiscal_quarter,
            "filing_date": filing_date,
            **altman_features,
            "risk_sentiment_score": sentiment_features.get("risk_sentiment_score", float("nan"))
        }

        processed_rows.append(combined_row)

    features_df = pd.DataFrame(processed_rows)

    if features_df.empty:
        return pd.DataFrame()

    # 5. Strict Point-in-Time Join with Ground-Truth Labels
    # We merge on [cik, fiscal_quarter], ensuring labels align perfectly with features
    merged_df = pd.merge(
        features_df,
        labels_df,
        on=["cik", "fiscal_quarter"],
        how="inner",
        suffixes=('', '_label')
    )

    # 6. Final Leakage Defense: Verify label alignment dates
    # If a label's event date occurs BEFORE the feature filing date, it's an invalid look-ahead conflict
    if "event_date" in merged_df.columns and "filing_date" in merged_df.columns:
        # Convert to datetime for safe comparison
        merged_df["filing_date"] = pd.to_datetime(merged_df["filing_date"])
        merged_df["event_date"] = pd.to_datetime(merged_df["event_date"])
        
        # Filter out rows where bankruptcy event predates the public filing of financial data
        valid_leakage_mask = (merged_df["event_date"].isna()) | (merged_df["event_date"] > merged_df["filing_date"])
        
        dropped_count = len(merged_df) - valid_leakage_mask.sum()
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows due to potential look-ahead leakage (event_date <= filing_date).")
            merged_df = merged_df[valid_leakage_mask]

    return merged_df