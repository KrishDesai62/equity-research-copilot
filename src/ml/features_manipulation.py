import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_feature_table(xbrl_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the Track B feature table for earnings manipulation detection.
    Pairs consecutive filing periods per CIK to compute Year-over-Year Beneish components,
    then merges point-in-time against distinct AAER enforcement and restatement labels.
    """
    if xbrl_df.empty or labels_df.empty:
        raise ValueError("Input XBRL dataframe or labels dataframe is empty.")

    xbrl_df = xbrl_df.sort_values(["cik", "filing_date"]).reset_index(drop=True)
    
    features_rows = []

    for cik, group in xbrl_df.groupby("cik"):
        group = group.sort_values("filing_date").reset_index(drop=True)
        
        for i in range(1, len(group)):
            row_t = group.iloc[i].to_dict()
            if "fiscal_period_end" not in group:
                raise ValueError("Beneish features require fiscal_period_end for annual comparisons")
            target_period = pd.Timestamp(row_t["fiscal_period_end"]) - pd.DateOffset(years=1)
            prior = group.iloc[:i]
            prior = prior[pd.to_datetime(prior["fiscal_period_end"]) == target_period]
            if prior.empty:
                continue
            row_prev = prior.iloc[-1].to_dict()
            
            beneish_feats = compute_beneish_components(row_t, row_prev)
            
            record = {
                "cik": cik,
                "ticker": row_t.get("ticker"),
                "filing_date": row_t.get("filing_date"),
                **beneish_feats
            }
            features_rows.append(record)

    features_df = pd.DataFrame(features_rows)

    if features_df.empty:
        logger.warning("No valid two-period pairs found for Track B feature extraction.")
        return pd.DataFrame()

   
    features_df["filing_date"] = pd.to_datetime(features_df["filing_date"])
    labels_df = labels_df.copy()
    labels_df["filing_date"] = pd.to_datetime(labels_df["filing_date"])
    master_df = pd.merge_asof(
        features_df.sort_values("filing_date"),
        labels_df.sort_values("filing_date"),
        on="filing_date",
        by="cik",
        direction="backward"
    )

    logger.info(f"Successfully constructed Track B feature table with {len(master_df)} records.")
    return master_df

def compute_beneish_components(current, previous):
    """Eight-ratio Beneish model. Missing/undefined ratios remain NaN."""
    def value(row, *tags):
        for tag in tags:
            number = row.get(tag)
            if number is not None and pd.notna(number):
                return float(number)
        return np.nan

    def divide(a, b):
        return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan

    def ratios(row):
        sales = value(row, "Revenues", "SalesRevenueNet")
        assets = value(row, "Assets")
        cogs = value(row, "CostOfRevenue", "CostOfGoodsAndServicesSold")
        dep = value(row, "DepreciationDepletionAndAmortization", "Depreciation")
        ppe = value(row, "PropertyPlantAndEquipmentNet")
        return {
            "receivables": divide(value(row, "AccountsReceivableNetCurrent"), sales),
            "margin": divide(sales - cogs, sales),
            "quality": 1 - divide(value(row, "AssetsCurrent") + ppe, assets),
            "sales": sales,
            "depreciation": divide(dep, dep + ppe),
            "sga": divide(value(row, "SellingGeneralAndAdministrativeExpense"), sales),
            "leverage": divide(value(row, "LiabilitiesCurrent") + value(row, "LongTermDebtNoncurrent", "LongTermDebt"), assets),
            "accruals": divide(value(row, "NetIncomeLoss") - value(row, "NetCashProvidedByUsedInOperatingActivities"), assets)}
    t, p = ratios(current), ratios(previous)
    result = {
        "dsri": divide(t["receivables"], p["receivables"]),
        "gmi": divide(p["margin"], t["margin"]),
        "aqi": divide(t["quality"], p["quality"]),
        "sgi": divide(t["sales"], p["sales"]),
        "depi": divide(p["depreciation"], t["depreciation"]),
        "sgai": divide(t["sga"], p["sga"]),
        "lvgi": divide(t["leverage"], p["leverage"]),
        "tata": t["accruals"]}
    weights = {"dsri": .920, "gmi": .528, "aqi": .404, "sgi": .892,
               "depi": .115, "sgai": -.172, "tata": 4.679, "lvgi": -.327}
    result["beneish_m_score"] = -4.84 + sum(weights[key] * result[key] for key in weights)
    return result
