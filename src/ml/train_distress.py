from src.ml.evaluate import company_walk_forward_splits, evaluate_classifier, compare_to_naive_baseline
from src.ml.neural_network import train_neural_network, validate_training_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
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
    X, y = validate_training_data(train_df, feature_cols, label_col)

    # 2. Build a robust pipeline with median imputation and logistic regression
    # Median imputation handles missing XBRL/sentiment values before they hit the solver
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
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
    try:
        import xgboost as xgb
    except (ImportError, OSError) as exc:
        raise RuntimeError("XGBoost requires its native runtime; on macOS install libomp") from exc

    X, y = validate_training_data(train_df, feature_cols, label_col)

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
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
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

def run_walk_forward_evaluation(df, feature_cols, label_col, *, trainer=None, company_col="cik",
                                min_train_periods=4, label_available_col=None,
                                n_company_folds=3, test_companies=None, period_freq="Q"):
    """Evaluate recurring company panels with disjoint company/time holdouts."""
    if df.empty or label_available_col is None:
        raise ValueError("Nonempty data and label_available_col are required")
    validate_training_data(df, feature_cols, label_col)
    trainer = trainer or train_logistic_baseline
    splits = company_walk_forward_splits(df, company_col=company_col,
        label_available_col=label_available_col, min_train_periods=min_train_periods,
        n_company_folds=n_company_folds, test_companies=test_companies, period_freq=period_freq)
    details, skipped, predictions = [], [], []
    for fold, (cohort, train_idx, test_idx) in enumerate(splits):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        metadata = {"fold": fold, "company_cohort": cohort, "train_samples": len(train),
                    "test_samples": len(test), "train_companies": train[company_col].nunique(),
                    "test_companies": test[company_col].nunique(),
                    "test_positives": int(test[label_col].sum()),
                    "test_start": str(pd.to_datetime(test["filing_date"]).min().date())}
        if train[label_col].nunique() < 2:
            skipped.append({**metadata, "reason": "Training has fewer than two classes"})
            continue
        model = trainer(train, feature_cols, label_col)
        test_features = test[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        probs = model.predict_proba(test_features)[:, 1]
        predictions.extend({"row_position": int(pos), "target": int(target), "score": float(score)}
                           for pos, target, score in zip(test_idx, test[label_col], probs))
        if test[label_col].nunique() < 2:
            skipped.append({**metadata, "reason": "Single-class test fold; included in pooled predictions"})
            continue
        metrics = evaluate_classifier(test[label_col], probs)
        naive = compare_to_naive_baseline(test[label_col], probs)
        details.append({**metadata, **metrics, "naive_pr_auc": naive["naive_pr_auc"],
                        "beats_naive": naive["beats_naive"]})
    if not predictions or len({p["target"] for p in predictions}) < 2:
        raise ValueError("No usable out-of-sample evaluation with both classes; inspect dates, cohorts and label availability")
    if len({p["row_position"] for p in predictions}) != len(predictions):
        raise ValueError("Evaluation generated duplicate out-of-sample rows")
    y, scores = [p["target"] for p in predictions], [p["score"] for p in predictions]
    pooled = {**evaluate_classifier(y, scores), **compare_to_naive_baseline(y, scores)}
    return {"mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in details])) if details else None,
            "validation_status": "evaluated" if details else "inconclusive_single_class_folds",
            "success_rate_vs_naive": float(np.mean([fold["beats_naive"] for fold in details])) if details else None,
            "total_folds_evaluated": len(details), "fold_details": details,
            "skipped_folds": skipped, "pooled_metrics": pooled, "predictions": predictions,
            "note": "Repeated quarterly positives may describe the same bankruptcy event; this is not an independent event count."}


def sanity_check_against_published_benchmark():
    url = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
    
    logger.info("Downloading UCI Polish Bankruptcy benchmark dataset...")
    try:
        response = urllib.request.urlopen(url, timeout=60)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            file_list = z.namelist()
            target_file = next(f for f in file_list if f.endswith('5year.arff'))
            with z.open(target_file) as f:
                df = pd.read_csv(f) if target_file.endswith('.csv') else parse_arff_to_dataframe(f)
                return df
    except Exception as e:
        logger.error(f"Failed to load benchmark dataset: {e}")
        raise e


def parse_arff_to_dataframe(file):
    from scipy.io import arff
    import io
    data, _ = arff.loadarff(io.StringIO(file.read().decode("utf-8")))
    frame = pd.DataFrame(data)
    for col in frame.select_dtypes(include="object"):
        frame[col] = frame[col].map(lambda value: value.decode() if isinstance(value, bytes) else value)
    return frame.apply(pd.to_numeric, errors="raise")


def compare_models(df, feature_cols, label_col, **evaluation_options):
    return {name: run_walk_forward_evaluation(df, feature_cols, label_col, trainer=trainer,
                                             **evaluation_options)
            for name, trainer in {"logistic": train_logistic_baseline,
                                  "xgboost": train_xgboost,
                                  "neural_network": train_neural_network}.items()}
