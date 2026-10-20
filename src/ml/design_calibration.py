"""Design-weighted calibration and metrics for fixed company-sampled cohorts."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)


def validate_weights(y, weights):
    y, weights = np.asarray(y), np.asarray(weights, dtype=float)
    if (y.ndim != 1 or not len(y) or weights.shape != y.shape or
            not set(y) <= {0, 1} or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Expected binary outcomes and matching finite positive design weights")
    return y.astype(int), weights


def metrics(y, probabilities, weights, threshold=None):
    y, w = validate_weights(y, weights)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid risk probabilities")
    both = len(np.unique(y)) == 2
    result = {"rows": len(y), "positive_rows": int(y.sum()), "weighted_prevalence": float(np.average(y, weights=w)),
              "mean_probability": float(np.average(p, weights=w)),
              "brier": float(brier_score_loss(y, p, sample_weight=w)),
              "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
              "average_precision": float(average_precision_score(y, p, sample_weight=w)) if both else None,
              "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if both else None}
    if threshold is not None:
        alert = p >= threshold
        tp, fp, fn, tn = [float(w[mask].sum()) for mask in (alert & (y == 1), alert & (y == 0), ~alert & (y == 1), ~alert & (y == 0))]
        result.update(precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.,
                      accuracy=(tp + tn) / w.sum(), weighted_confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return result


def choose_threshold(y, probabilities, weights, min_precision=.5, min_recall=.5):
    y, w = validate_weights(y, weights)
    if len(np.unique(y)) != 2:
        return None
    p, r, thresholds = precision_recall_curve(y, probabilities, sample_weight=w)
    feasible = np.flatnonzero((p[:-1] >= min_precision) & (r[:-1] >= min_recall))
    if not len(feasible):
        return None
    # Maximum recall subject to the fixed precision floor; precision breaks ties.
    best = max(feasible, key=lambda i: (r[i], p[i], thresholds[i]))
    return float(thresholds[best])


def reliability(y, probabilities, weights):
    """Fixed bins; descriptive diagnostics rather than a calibration certificate."""
    y, w = validate_weights(y, weights)
    p = np.asarray(probabilities, dtype=float)
    metrics(y, p, w)  # Validate predictions using the same scoring contract.
    bins = [0., .01, .02, .05, .1, .2, .5, 1.]
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (p >= lower) & ((p <= upper) if upper == 1 else (p < upper))
        if mask.any():
            rows.append({"lower": lower, "upper": upper, "rows": int(mask.sum()),
                "positive_rows": int(y[mask].sum()), "weighted_mean_probability": float(np.average(p[mask], weights=w[mask])),
                "weighted_observed_rate": float(np.average(y[mask], weights=w[mask]))})
    return rows


class WeightedSigmoid:
    """Separate held-out calibration; no resampling or class balancing."""
    def __init__(self, model, regularization=.1):
        self.model = model
        self.regularization = regularization
        self.classes_ = np.array([0, 1])

    def raw(self, X):
        return self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]

    @staticmethod
    def inputs(probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid calibration probabilities")
        return logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)

    def fit_scores(self, probabilities, y, weights):
        y, w = validate_weights(y, weights)
        if len(np.unique(y)) != 2:
            raise ValueError("Calibration requires both outcomes; no synthetic negatives")
        self.calibrator_ = LogisticRegression(C=self.regularization, max_iter=2000, random_state=42)
        self.calibrator_.fit(self.inputs(probabilities), y, sample_weight=w * len(w) / w.sum())
        return self

    def predict_scores(self, probabilities):
        return self.calibrator_.predict_proba(self.inputs(probabilities))[:, 1]

    def predict_proba(self, X):
        p = self.predict_scores(self.raw(X))
        return np.column_stack([1 - p, p])
