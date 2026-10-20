from src.ml.neural_network import train_neural_network, validate_training_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
import numpy as np

logger = logging.getLogger(__name__)

def train_track_b_logistic_baseline(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_aaer_enforcement_flag"):
    """
    Trains a production-grade Logistic Regression baseline for Track B (fraud/manipulation).
    Uses the AAER target separately from the restatement target. Historical
    restatement features must be available as of each filing date.
    """
    if train_df.empty or not feature_cols or not target_col:
        raise ValueError("Training dataframe is empty or feature/target columns are missing.")

    # 1. Extract feature matrix (X) — includes auxiliary features like restatement_flag
    if "target_restatement_flag" in feature_cols:
        raise ValueError("Future restatement outcome cannot be a predictive feature")
    X, y = validate_training_data(train_df, feature_cols, target_col)

    # 2. Build robust imputation and logistic regression pipeline
    # Fraud and accounting anomalies are exceptionally rare; balanced class weights are mandatory.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced', 
            max_iter=1500, 
            random_state=42
        ))
    ])

    try:
        pipeline.fit(X, y)
        logger.info(f"Successfully trained Track B Logistic baseline on {len(train_df)} samples targeting {target_col}.")
    except Exception as e:
        logger.error(f"Error during Track B logistic training: {e}")
        raise e

    return pipeline


def train_track_b_xgboost(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_aaer_enforcement_flag"):
    """
    Trains a production-grade XGBoost classifier for Track B fraud detection.
    Computes dynamic scale_pos_weight for extreme class imbalance and embeds a median imputer.
    """
    if train_df.empty or not feature_cols or not target_col:
        raise ValueError("Training dataframe is empty or feature/target columns are missing.")

    try:
        import xgboost as xgb
    except (ImportError, OSError) as exc:
        raise RuntimeError("XGBoost requires its native runtime; on macOS install libomp") from exc

    if "target_restatement_flag" in feature_cols:
        raise ValueError("Future restatement outcome cannot be a predictive feature")
    X, y = validate_training_data(train_df, feature_cols, target_col)

    # Compute dynamic class imbalance ratio for extreme minority fraud events
    num_negative = (y == 0).sum()
    num_positive = (y == 1).sum()
    
    if num_positive > 0:
        scale_pos_weight_val = float(num_negative) / float(num_positive)
    else:
        logger.warning(f"No positive {target_col} labels found in training set. Defaulting scale_pos_weight to 1.0.")
        scale_pos_weight_val = 1.0

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight_val,
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])

    try:
        pipeline.fit(X, y)
        logger.info(
            f"Successfully trained Track B XGBoost classifier on {len(train_df)} samples "
            f"(Target: {target_col}, computed scale_pos_weight: {scale_pos_weight_val:.2f})."
        )
    except Exception as e:
        logger.error(f"Error during Track B XGBoost training: {e}")
        raise e

    return pipeline

def train_track_b_neural_network(train_df, feature_cols, target_col="target_aaer_enforcement_flag", **kwargs):
    if "target_restatement_flag" in feature_cols:
        raise ValueError("Future restatement outcome cannot be a predictive feature")
    return train_neural_network(train_df, feature_cols, target_col, **kwargs)
