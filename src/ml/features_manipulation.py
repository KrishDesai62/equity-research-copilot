"""
Track B features: Beneish M-score components (accruals ratio,
days-sales-in-receivables index, asset quality index, leverage growth) —
same underlying XBRL data as Track A.

NOT in MVP scope. Build after Track A is validated (Phase 2), reusing
xbrl_bulk.py output — this is the "almost no added data-engineering
cost" part of the original spec, and it's true, but only once Track A's
pipeline is proven correct.
"""
import pandas as pd


def compute_beneish_components(xbrl_row_t: dict, xbrl_row_t_minus_1: dict) -> dict:
    """
    Beneish M-score components need TWO periods (year-over-year changes),
    unlike Altman which is single-period. TODO: implement DSRI, AQI,
    leverage growth, etc. Same missing-tag handling policy as
    features_distress.py.
    """
    raise NotImplementedError


def build_feature_table(xbrl_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: same pattern as features_distress.build_feature_table, but
    join against the two separate label columns from
    data_layer/restatement_labels.py (aaer_enforcement_flag,
    restatement_flag) — keep them separate through this table too.
    """
    raise NotImplementedError
