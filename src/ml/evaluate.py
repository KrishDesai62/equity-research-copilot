import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

logger = logging.getLogger(__name__)


def company_walk_forward_splits(df, *, company_col="cik", date_col="filing_date",
                                label_available_col="label_available_date", min_train_periods=4,
                                n_company_folds=3, test_companies=None, period_freq="Q"):
    """Cross company holdouts with time windows and purge outcomes not yet known.

    Company cohorts depend only on identifiers, never on outcome labels. All
    historical rows of a held-out company stay outside that fold's training set.
    Returned indices refer to original row positions, including unsorted input.
    """
    from hashlib import sha256
    required = {company_col, date_col, label_available_col}
    if not required <= set(df):
        raise ValueError(f"Missing split columns: {sorted(required - set(df))}")
    if df[list(required)].isna().any().any():
        raise ValueError("Company identifiers and split dates cannot be missing")
    dates = pd.to_datetime(df[date_col], errors="raise")
    available = pd.to_datetime(df[label_available_col], errors="raise")
    if (available < dates).any():
        raise ValueError("Label availability cannot precede the public feature date")
    companies = df[company_col].astype(str)
    unique = sorted(companies.unique(), key=lambda value: sha256(value.encode()).hexdigest())
    if test_companies is not None:
        heldout = {str(company) for company in test_companies}
        if not heldout or not heldout <= set(unique) or heldout == set(unique):
            raise ValueError("Test companies must be a nonempty proper subset of the dataset")
        cohorts = [heldout]
    else:
        if n_company_folds < 2 or len(unique) < 2:
            raise ValueError("Company-separated evaluation requires at least two companies and cohorts")
        cohorts = [set(unique[i::min(n_company_folds, len(unique))])
                   for i in range(min(n_company_folds, len(unique)))]
    for cohort_id, cohort in enumerate(cohorts):
        for prior_idx, test_idx in walk_forward_splits(df, date_col, min_train_periods, period_freq):
            test_idx = test_idx[companies.iloc[test_idx].isin(cohort).to_numpy()]
            if not len(test_idx):
                continue
            cutoff = dates.iloc[test_idx].min()
            prior = np.zeros(len(df), dtype=bool)
            prior[prior_idx] = True
            train_mask = prior & ~companies.isin(cohort) & (dates < cutoff) & (available < cutoff)
            train_idx = np.flatnonzero(train_mask.to_numpy())
            yield cohort_id, train_idx, test_idx

def walk_forward_splits(df: pd.DataFrame, date_col: str = "filing_date", min_train_periods: int = 4, period_freq="Q") -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generates chronological walk-forward train/test index splits.
    Ensures zero data leakage by expanding training windows forward one period at a time,
    strictly preventing any look-ahead bias or random cross-validation shuffling.
    """
    if df.empty or date_col not in df.columns:
        raise ValueError(f"Dataframe is empty or missing required date column '{date_col}'.")

    if min_train_periods < 1:
        raise ValueError("min_train_periods must be positive")
    dates = pd.to_datetime(df[date_col], errors="raise")
    if dates.isna().any():
        raise ValueError("Filing dates cannot be missing")
    if period_freq not in {"Q", "Y"}:
        raise ValueError("Evaluation periods must be Q (quarter) or Y (year)")
    periods = dates.dt.to_period(period_freq)
    unique_periods = sorted(periods.unique())
    splits = []
    for i in range(min_train_periods, len(unique_periods)):
        train_idx = np.flatnonzero((periods < unique_periods[i]).to_numpy())
        test_idx = np.flatnonzero((periods == unique_periods[i]).to_numpy())
        splits.append((train_idx, test_idx))

    logger.info(f"Generated {len(splits)} chronological walk-forward splits.")
    return splits


def evaluate_classifier(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Evaluates classifier performance using rigorous metrics tailored for rare classes:
    PR-AUC (Primary), ROC-AUC, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    if y_true.ndim != 1 or y_pred_proba.ndim != 1 or len(y_true) == 0 or len(y_true) != len(y_pred_proba):
        raise ValueError("Provide equally sized, nonempty one-dimensional targets and probabilities")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("Targets must be binary (0 or 1)")
    if not np.isfinite(y_pred_proba).all() or ((y_pred_proba < 0) | (y_pred_proba > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    if len(np.unique(y_true)) < 2:
        logger.warning("Evaluation target contains only a single class label. Metrics will be NaN or uninformative.")
        return {
            "pr_auc": float('nan'),
            "roc_auc": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1": float('nan'),
            "confusion_matrix": None
        }

    # Binarize predictions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute core imbalanced metrics
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        raise e

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix
    }


def compare_to_naive_baseline(y_true, y_pred_proba = None) -> dict:
    """
    Computes the performance of a naive baseline ("always predict majority class" 
    or prior base rate) to establish an explicit lower-bound floor for model validation.
    Under heavy class imbalance, naive PR-AUC equals the positive class prevalence.
    """
    y_true = np.array(y_true)
    if len(y_true) == 0:
        return {"naive_pr_auc": 0.0, "beats_naive": False}

    # The baseline PR-AUC for a random/constant predictor in an imbalanced set 
    # equals the proportion of positive samples (base rate).
    positive_prevalence = float(y_true.mean())
    
    result = {
        "naive_pr_auc": positive_prevalence,
        "positive_class_ratio": positive_prevalence
    }

    if y_pred_proba is not None:
        model_pr_auc = average_precision_score(y_true, y_pred_proba)
        result["model_pr_auc"] = float(model_pr_auc)
        result["beats_naive"] = bool(model_pr_auc > positive_prevalence)

    return result
