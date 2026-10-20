"""As-filed quarterly/annual financial features with explicit flow durations.

Quarterly cash-flow statements are cumulative: derive standalone quarters by
subtracting the immediately preceding YTD disclosure, then sum four contiguous
quarters. Annual flows already represent a trailing fiscal year. All inputs to a
row must have been published on or before that row's filing date.
"""
from collections import defaultdict
import math

import numpy as np
import pandas as pd

from src.ml.features_distress import compute_altman_components
from src.validation.sec_pilot import TAGS

FLOW_TAGS = {
    "Revenues": TAGS["Revenues"], "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "NetIncomeLoss": ["NetIncomeLoss"], "InterestExpense": ["InterestExpense", "InterestAndDebtExpense"],
    "PretaxIncome": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                     "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]}
INSTANT_TAGS = {key: tags for key, tags in TAGS.items() if key not in {"Revenues", "OperatingIncomeLoss"}}
INSTANT_TAGS["Cash"] = ["CashAndCashEquivalentsAtCarryingValue", "Cash"]
BASE_FEATURES = ["wc_to_ta", "re_to_ta", "ebit_to_ta", "sales_to_ta"]
EXTRA_FEATURES = ["current_ratio", "liabilities_to_assets", "cash_to_assets", "cashflow_to_assets",
                  "operating_margin", "cashflow_margin", "earnings_proxy_to_assets", "interest_coverage",
                  "revenue_growth_yoy", "liquidity_change_yoy", "leverage_change_yoy", "cashflow_margin_change_yoy"]


def divide(numerator, denominator):
    if not all(pd.notna(x) and math.isfinite(float(x)) for x in (numerator, denominator)) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def extract_history(payload, ticker, *, public_filings=None):
    """Reconcile facts before computing history when submissions are supplied.

    Company Facts occasionally misclassifies amendments or gives a conflicting
    public date. Such inputs must never influence later TTM or trend features.
    The optional argument preserves support for standalone extraction diagnostics.
    """
    allowed = None
    if public_filings is not None:
        cik = str(payload["cik"]).zfill(10)
        allowed = {(r["accession_number"], r["filing_date"], r["filing_type"])
                   for r in public_filings if str(r["cik"]).zfill(10) == cik
                   and r["filing_type"] in {"10-K", "10-Q"}}
    gaap = payload.get("facts", {}).get("us-gaap", {})
    indexed = {}
    for tag in {tag for tags in (*INSTANT_TAGS.values(), *FLOW_TAGS.values()) for tag in tags}:
        groups = defaultdict(list)
        for fact in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if fact.get("form") in {"10-K", "10-Q"}:
                groups[fact["accn"]].append(fact)
        indexed[tag] = groups
    rows, provenance, issues = [], [], []
    for accession, assets in indexed.get("Assets", {}).items():
        end = max(f["end"] for f in assets)
        meta = [f for f in assets if f["end"] == end]
        dates = {f["filed"] for f in meta}
        if len(dates) != 1:
            issues.append({"accession": accession, "reason": "Conflicting public dates"})
            continue
        filed = dates.pop()
        if end > filed:
            issues.append({"accession": accession, "filing_date": filed, "fiscal_period_end": end,
                           "reason": "Fiscal end follows public filing date; entire accession quarantined"})
            continue
        form = meta[0]["form"]
        if allowed is not None and (accession, filed, form) not in allowed:
            issues.append({"accession": accession, "filing_date": filed,
                           "reason": "Unreconciled original submission; excluded before history calculation"})
            continue
        row = {"cik": str(payload["cik"]).zfill(10), "ticker": ticker,
               "accession_number": accession, "filing_date": filed,
               "fiscal_period_end": end, "filing_type": form}
        for canonical, tags in {**INSTANT_TAGS, **FLOW_TAGS}.items():
            value, chosen = np.nan, None
            for tag in tags:
                candidates = [f for f in indexed[tag].get(accession, []) if f["end"] == end and f["filed"] == filed]
                if canonical in FLOW_TAGS:
                    minimum, maximum = (300, 380) if form == "10-K" else (60, 300)
                    candidates = [f for f in candidates if f.get("start") and
                                  minimum <= (pd.Timestamp(end) - pd.Timestamp(f["start"])).days <= maximum]
                    if candidates:
                        start = min(f["start"] for f in candidates)
                        candidates = [f for f in candidates if f["start"] == start]
                else:
                    candidates = [f for f in candidates if not f.get("start")]
                unique = {f["val"] for f in candidates}
                if len(unique) > 1:
                    issues.append({"accession": accession, "field": canonical, "reason": "Conflicting values withheld"})
                    break
                if len(unique) == 1:
                    chosen = candidates[0]
                    value = chosen["val"]
                    provenance.append({"cik": row["cik"], "field": canonical, "source_tag": tag, **chosen})
                    break
            row[canonical] = value
            if canonical in FLOW_TAGS:
                row[canonical + "_start"] = chosen["start"] if chosen else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No original 10-K/10-Q USD Assets facts for {ticker}")
    return add_history_features(frame), provenance, issues


def add_history_features(frame):
    result = []
    for _, group in frame.groupby("cik", sort=False):
        history = []
        for row in group.sort_values(["filing_date", "fiscal_period_end"]).to_dict("records"):
            end = pd.Timestamp(row["fiscal_period_end"])
            # Keep only distinct earlier periods actually available at this filing date.
            earlier = [r for r in history if pd.Timestamp(r["fiscal_period_end"]) < end]
            previous = max(earlier, key=lambda r: r["fiscal_period_end"], default=None)
            for field in FLOW_TAGS:
                start = row.get(field + "_start")
                days = (end - pd.Timestamp(start)).days if pd.notna(start) else 0
                standalone = np.nan
                quarter_start = None
                if 75 <= days <= 120:
                    standalone = row[field]
                    quarter_start = start
                elif previous and pd.notna(start) and previous.get(field + "_start") == start:
                    gap = (end - pd.Timestamp(previous["fiscal_period_end"])).days
                    if 75 <= gap <= 120:
                        standalone = row[field] - previous[field]
                        quarter_start = (pd.Timestamp(previous["fiscal_period_end"]) + pd.Timedelta(days=1)).date().isoformat()
                row[field + "_quarter"] = standalone
                row[field + "_quarter_start"] = quarter_start
                if row["filing_type"] == "10-K" and 350 <= days <= 380:
                    row[field + "_ttm"] = row[field]
                else:
                    distinct = {r["fiscal_period_end"]: r for r in earlier}
                    trailing = [distinct[k] for k in sorted(distinct)[-3:]] + [row]
                    gaps = [(pd.Timestamp(b["fiscal_period_end"]) - pd.Timestamp(a["fiscal_period_end"])).days
                            for a, b in zip(trailing, trailing[1:])]
                    values = [r.get(field + "_quarter", np.nan) for r in trailing]
                    contiguous = all(pd.notna(b.get(field + "_quarter_start")) and
                                     pd.Timestamp(b[field + "_quarter_start"]) == pd.Timestamp(a["fiscal_period_end"]) + pd.Timedelta(days=1)
                                     for a, b in zip(trailing, trailing[1:]))
                    first_start = trailing[0].get(field + "_quarter_start")
                    full_year = pd.notna(first_start) and 350 <= (end - pd.Timestamp(first_start)).days <= 380
                    row[field + "_ttm"] = sum(values) if len(values) == 4 and contiguous and full_year and all(75 <= gap <= 120 for gap in gaps) else np.nan
            annualized = {**row, "Revenues": row["Revenues_ttm"], "OperatingIncomeLoss": row["OperatingIncomeLoss_ttm"]}
            row.update(compute_altman_components(annualized))
            assets, liabilities = row["Assets"], row["Liabilities"]
            revenue, operating = row["Revenues_ttm"], row["OperatingIncomeLoss_ttm"]
            cashflow, interest = row["OperatingCashFlow_ttm"], row["InterestExpense_ttm"]
            # Explicit proxy: operating profit preferred; EBT plus interest is a fallback.
            proxy = operating if pd.notna(operating) else row["PretaxIncome_ttm"] + interest
            row["earnings_proxy_method"] = "operating_income" if pd.notna(operating) else "pretax_plus_interest"
            row.update({"current_ratio": divide(row["AssetsCurrent"], row["LiabilitiesCurrent"]),
                        "liabilities_to_assets": divide(liabilities, assets), "cash_to_assets": divide(row["Cash"], assets),
                        "cashflow_to_assets": divide(cashflow, assets), "operating_margin": divide(operating, revenue),
                        "cashflow_margin": divide(cashflow, revenue), "earnings_proxy_to_assets": divide(proxy, assets),
                        "interest_coverage": divide(proxy, interest) if pd.notna(interest) and interest > 0 else np.nan})
            year_ago = [r for r in earlier if 350 <= (end - pd.Timestamp(r["fiscal_period_end"])).days <= 380]
            prior = max(year_ago, key=lambda r: r["fiscal_period_end"], default=None)
            row["revenue_growth_yoy"] = divide(revenue, prior["Revenues_ttm"]) - 1 if prior else np.nan
            for feature, source in (("liquidity_change_yoy", "current_ratio"),
                                    ("leverage_change_yoy", "liabilities_to_assets"),
                                    ("cashflow_margin_change_yoy", "cashflow_margin")):
                row[feature] = row[source] - prior[source] if prior else np.nan
            row["latest_input_filing_date"] = max([row["filing_date"]] + [r["filing_date"] for r in earlier])
            history.append(row)
            result.append(row)
    return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan)
