"""
Shared evaluation utilities for both tracks. Imbalanced classes on both
(bankruptcy and manipulation are rare) — accuracy is close to useless
here (a naive "predict never" model scores >95% and means nothing).
Always report AUC-PR, precision, recall, and a confusion matrix at a
chosen threshold, not accuracy alone.
"""
import pandas as pd


def walk_forward_splits(df: pd.DataFrame, date_col: str, min_train_years: int) -> list[tuple]:
    """
    Yield (train_df, test_df) pairs where train is everything up to time
    T and test is the next period after T — never randomly shuffle rows
    across time. Also make sure no single company's rows appear in both
    train and test within a split (split by company AND walk forward in
    time, per the original spec).

    TODO: implement. A simple approach: sort unique fiscal periods,
    slide a window forward one period at a time after the minimum
    training window is satisfied.
    """
    raise NotImplementedError


def evaluate_classifier(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    TODO: return {auc_pr, precision, recall, f1, confusion_matrix} using
    sklearn.metrics (average_precision_score for AUC-PR, not roc_auc_score
    alone — report both if you want, but AUC-PR is the one that matters
    under class imbalance).
    """
    raise NotImplementedError


def compare_to_naive_baseline(y_true) -> dict:
    """
    TODO: compute what a "always predict majority class" model would
    score, so every reported result has an explicit baseline to beat.
    This number belongs in every write-up next to your model's score.
    """
    raise NotImplementedError
