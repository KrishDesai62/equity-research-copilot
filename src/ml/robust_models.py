"""Conservative tabular research models with training-only outlier treatment."""
import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.ml.neural_network import validate_training_data


class QuantileClipper(TransformerMixin, BaseEstimator):
    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Invalid quantile clipping limits")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("Clipping needs a finite nonempty training matrix")
        self.lower_ = np.quantile(values, self.lower, axis=0)
        self.upper_ = np.quantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature schema changed after fitting")
        return np.clip(values, self.lower_, self.upper_)


def train_candidate(frame, features, target, name, sample_weight=None):
    X, y = validate_training_data(frame, features, target)
    # Legacy experiments weight companies equally. Probability-sampled cohorts
    # supply their frozen inverse inclusion weights for every snapshot instead.
    weights = (1 / frame.groupby("cik").cik.transform("size").to_numpy()
               if sample_weight is None else np.asarray(sample_weight, dtype=float).copy())
    if weights.shape != (len(frame),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training weights must be finite, positive and match the rows")
    weights *= len(weights) / weights.sum()
    balance = "balanced"
    if sample_weight is not None:
        balance = {label: float(weights.sum() / (2 * weights[y.to_numpy() == label].sum())) for label in (0, 1)}
    if name.startswith("logistic"):
        classifier = LogisticRegression(C=.1, class_weight=balance if name.endswith("balanced") else None,
                                        max_iter=2000, random_state=42)
    elif name.startswith("xgb"):
        from xgboost import XGBClassifier
        ratio = (float((y == 0).sum() / (y == 1).sum()) if sample_weight is None else
                 float(weights[y.to_numpy() == 0].sum() / weights[y.to_numpy() == 1].sum()))
        classifier = XGBClassifier(n_estimators=200, learning_rate=.03, max_depth=2,
            min_child_weight=5, reg_lambda=10, reg_alpha=1, subsample=.8,
            colsample_bytree=.8, scale_pos_weight=np.sqrt(ratio) if name.endswith("sqrt_balance") else 1.,
            n_jobs=2, random_state=42, eval_metric="logloss")
    elif name == "neural_network":
        classifier = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1., max_iter=500,
            batch_size=min(64, len(frame)), random_state=42, early_stopping=False)
    else:
        raise ValueError(f"Unknown development model: {name}")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
                         ("clipper", QuantileClipper()), ("scaler", RobustScaler()), ("classifier", classifier)])
    pipeline.fit(X, y, classifier__sample_weight=weights)
    return pipeline


class InterceptCalibrator:
    """Shift log odds on separate natural calibration data; preserve ranking.

    A fixed pseudocount prevents zero observed rare events from asserting zero
    risk. This is an uncertain research estimate, not validated calibration.
    """
    def __init__(self, model, pseudocount=1.):
        self.model = model
        self.pseudocount = pseudocount
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1 or not len(y) or not set(y) <= {0, 1} or not np.isfinite(self.pseudocount) or self.pseudocount <= 0:
            raise ValueError("Calibration needs nonempty binary outcomes and a positive pseudocount")
        scores = self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or len(scores) != len(y):
            raise ValueError("Invalid base calibration scores")
        logits = logit(np.clip(scores, 1e-7, 1 - 1e-7))
        self.smoothed_prevalence_ = float((y.sum() + self.pseudocount) / (len(y) + 2 * self.pseudocount))
        low, high = -50., 50.
        for _ in range(100):
            midpoint = (low + high) / 2
            if expit(logits + midpoint).mean() < self.smoothed_prevalence_:
                low = midpoint
            else:
                high = midpoint
        self.offset_ = (low + high) / 2
        return self

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]
        if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("Invalid base prediction scores")
        calibrated = expit(logit(np.clip(raw, 1e-7, 1 - 1e-7)) + self.offset_)
        return np.column_stack([1 - calibrated, calibrated])
