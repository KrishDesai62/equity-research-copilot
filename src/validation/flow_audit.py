"""Cross-check trailing flows using the independent YTD + annual - prior-YTD identity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.financial_history import FLOW_TAGS


def audit_trailing_flows(frame):
    checks = []
    for _, group in frame.groupby("cik"):
        for row in group.loc[group.filing_type.eq("10-Q")].to_dict("records"):
            end, filed = pd.Timestamp(row["fiscal_period_end"]), pd.Timestamp(row["filing_date"])
            prior = group.loc[(pd.to_datetime(group.filing_date) <= filed) &
                              (pd.to_datetime(group.fiscal_period_end) < end)]
            for field in FLOW_TAGS:
                start = row.get(field + "_start")
                if pd.isna(start) or pd.isna(row.get(field + "_ttm")):
                    continue
                annual = prior.loc[prior.filing_type.eq("10-K") & (pd.to_datetime(prior.fiscal_period_end) < pd.Timestamp(start))]
                if annual.empty:
                    continue
                annual = annual.sort_values("fiscal_period_end").iloc[-1]
                annual_start = annual[field + "_start"]
                if pd.isna(annual_start) or not 350 <= (pd.Timestamp(annual.fiscal_period_end) - pd.Timestamp(annual_start)).days <= 380:
                    continue  # Short successor/transition periods are not a full annual basis.
                prior_ytd = prior.loc[prior.filing_type.eq("10-Q") &
                    ((end - pd.to_datetime(prior.fiscal_period_end)).dt.days.between(350, 380)) &
                    prior[field + "_start"].eq(annual[field + "_start"])]
                if prior_ytd.empty or pd.isna(annual[field]):
                    continue
                prior_ytd = prior_ytd.sort_values("filing_date").iloc[-1]
                if pd.isna(prior_ytd[field]):
                    continue
                current_span = (end - pd.Timestamp(start)).days
                prior_span = (pd.Timestamp(prior_ytd.fiscal_period_end) - pd.Timestamp(prior_ytd[field + "_start"])).days
                if abs(current_span - prior_span) > 8:
                    continue  # Fresh-start stubs are not comparable year-to-date intervals.
                expected = row[field] + annual[field] - prior_ytd[field]
                checks.append({"cik": row["cik"], "accession_number": row["accession_number"], "field": field,
                    "trailing_four_quarters": row[field + "_ttm"], "annual_plus_ytd_minus_prior_ytd": expected,
                    "matches": bool(np.isclose(row[field + "_ttm"], expected, rtol=1e-10, atol=1))})
    return {"method": "Four-quarter sum versus current YTD plus prior annual minus prior YTD, using as-of original disclosures",
            "check_count": len(checks), "mismatches": sum(not row["matches"] for row in checks), "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/validation/expanded/all_rows.csv")
    parser.add_argument("--output", default="data/processed/validation/expanded/flow_audit.json")
    args = parser.parse_args()
    result = audit_trailing_flows(pd.read_csv(args.input, dtype={"cik": str}))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(result["check_count"], "arithmetic checks;", result["mismatches"], "mismatches")
    if not result["check_count"] or result["mismatches"]:
        raise SystemExit(1)
