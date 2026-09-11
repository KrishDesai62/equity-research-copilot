import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
import xgboost as xgb
import numpy as np
import urllib.request
import io
import zipfile


logger = logging.getLogger(__name__)

def train_logistic_baseline(train_df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """
    Trains a Logistic Regression baseline model with balanced class weights 
    to account for rare bankruptcy/distress events. Includes an imputer
    to safely handle any remaining NaN values in the feature set.
    """
    if train_df.empty or not feature_cols or not label_col:
        raise ValueError("Training dataframe is empty or feature/label columns are missing.")

    # 1. Extract feature matrix (X) and target vector (y)
    X = train_df[feature_cols]
    y = train_df[label_col]

    # 2. Build a robust pipeline with median imputation and logistic regression
    # Median imputation handles missing XBRL/sentiment values before they hit the solver
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42
        ))
    ])

    # 3. Fit the model on the training data
    try:
        pipeline.fit(X, y)
        logger.info(f"Successfully trained logistic regression baseline on {len(train_df)} samples.")
    except Exception as e:
        logger.error(f"Error during logistic regression training: {e}")
        raise e

    return pipeline

logger = logging.getLogger(__name__)

def train_xgboost(train_df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """
    Trains a production-grade XGBoost classifier for financial distress prediction.
    Automatically computes scale_pos_weight from the training class imbalance ratio
    and embeds a median imputer to handle missing XBRL/sentiment features cleanly.
    """
    if train_df.empty or not feature_cols or not label_col:
        raise ValueError("Training dataframe is empty or feature/label columns are missing.")

    # 1. Extract feature matrix (X) and target vector (y)
    X = train_df[feature_cols]
    y = train_df[label_col]

    # 2. Compute dynamic class imbalance ratio for scale_pos_weight
    # scale_pos_weight = count(negative classes) / count(positive classes)
    # This prevents the rare bankruptcy/distress class from being ignored by gradient boosting.
    num_negative = (y == 0).sum()
    num_positive = (y == 1).sum()
    
    if num_positive > 0:
        scale_pos_weight_val = float(num_negative) / float(num_positive)
    else:
        logger.warning("No positive distress labels found in training set. Defaulting scale_pos_weight to 1.0.")
        scale_pos_weight_val = 1.0

    # 3. Build a production-grade pipeline handling imputation and XGBoost modeling
    # Note: XGBoost natively handles NaNs, but passing an explicit SimpleImputer 
    # guarantees consistent feature matrix shapes during inference and testing.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight_val,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4. Fit the model on the training data
    try:
        pipeline.fit(X, y)
        logger.info(
            f"Successfully trained XGBoost classifier on {len(train_df)} samples "
            f"(computed scale_pos_weight: {scale_pos_weight_val:.2f})."
        )
    except Exception as e:
        logger.error(f"Error during XGBoost training: {e}")
        raise e

    return pipeline


logger = logging.getLogger(__name__)

def run_walk_forward_evaluation(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> dict:
    """
    Executes a production-grade walk-forward evaluation across temporal folds.
    Applies train_logistic_baseline or train_xgboost sequentially over rolling windows,
    evaluates performance using PR-AUC, and benchmarks against a naive prior baseline
    to fulfill MVP 'done' criterion #3.
    """
    if df.empty or "filing_date" not in df.columns:
        raise ValueError("Evaluation dataframe is empty or missing 'filing_date' index column.")

    # Ensure chronological order for strict time-series integrity
    df = df.sort_values("filing_date").reset_index(drop=True)

    # 1. Generate rolling walk-forward splits 
    # (Assuming walk_forward_splits yields train/test index tuples or boolean masks)
    splits = walk_forward_splits(df["filing_date"])
    
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_subset = df.iloc[train_idx]
        test_subset = df.iloc[test_idx]

        if train_subset.empty or test_subset.empty:
            continue

        # 2. Train model on historical rolling window
        # For MVP baseline, we use the logistic regression function
        model = train_logistic_baseline(train_subset, feature_cols, label_col)

        # 3. Generate out-of-sample probability predictions
        X_test = test_subset[feature_cols]
        y_test = test_subset[label_col]
        
        if len(y_test.unique()) < 2:
            logger.warning(f"Fold {fold_idx} test set contains only one class label. Skipping PR-AUC calculation.")
            continue

        preds_proba = model.predict_proba(X_test)[:, 1]

        # 4. Evaluate classifier using strict metrics (PR-AUC)
        fold_metrics = evaluate_classifier(y_test, preds_proba)

        # 5. Compare performance directly against a naive baseline
        naive_comparison = compare_to_naive_baseline(y_test, preds_proba)

        fold_results.append({
            "fold": fold_idx,
            "pr_auc": fold_metrics.get("pr_auc"),
            "beats_naive": naive_comparison.get("beats_naive"),
            "test_samples": len(test_subset)
        })

    if not fold_results:
        logger.error("Walk-forward evaluation completed with zero valid evaluation folds.")
        return {}

    # Aggregate fold performance summary
    metrics_df = pd.DataFrame(fold_results)
    summary = {
        "mean_pr_auc": float(metrics_df["pr_auc"].mean()),
        "success_rate_vs_naive": float(metrics_df["beats_naive"].mean()),
        "total_folds_evaluated": len(metrics_df),
        "fold_details": fold_results
    }

    logger.info(f"Walk-forward evaluation complete. Mean PR-AUC: {summary['mean_pr_auc']:.4f}")
    return summary


def sanity_check_against_published_benchmark():
    url = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
    
    logger.info("Downloading UCI Polish Bankruptcy benchmark dataset...")
    try:
        response = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            file_list = z.namelist()
            target_file = [f for f in file_list if f.endswith('.arff') or f.endswith('.csv')][0]
            with z.open(target_file) as f:
                df = pd.read_csv(f) if target_file.endswith('.csv') else parse_arff_to_dataframe(f)
                return df
    except Exception as e:
        logger.error(f"Failed to load benchmark dataset: {e}")
        raise e
