"""Fit and tune research candidates on three disjoint, chronological datasets."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve

from src.ml.evaluate import evaluate_classifier
from src.ml.neural_network import validate_training_data


def check_partitions(parts, features, target, available_col="label_available_date_1yr"):
    """Purged outcomes and company separation apply to calibration and tuning too."""
    names = ("training", "calibration", "threshold_tuning")
    if set(parts) != set(names):
        raise ValueError("Provide training, calibration and threshold_tuning partitions")
    company_sets = []
    for name in names:
        frame = parts[name]
        if frame.empty:
            raise ValueError(f"{name} partition is empty")
        validate_training_data(frame, features, target)
        if frame[["cik", "filing_date", available_col]].isna().any().any():
            raise ValueError(f"{name} partition lacks identifiers or dates")
        dates = pd.to_datetime(frame.filing_date, errors="raise")
        available = pd.to_datetime(frame[available_col], errors="raise")
        if (available < dates).any():
            raise ValueError("Label availability precedes public features")
        companies = set(frame.cik.astype(str).str.zfill(10))
        if any(companies & earlier for earlier in company_sets):
            raise ValueError("Companies overlap across fitting, calibration or threshold tuning")
        company_sets.append(companies)
    stages = [parts[name] for name in names]
    for earlier, later in zip(stages, stages[1:]):
        if pd.to_datetime(earlier[available_col]).max() >= pd.to_datetime(later.filing_date).min():
            raise ValueError("Earlier-stage outcomes are not known before the next stage")


def select_threshold(y, probabilities, *, min_precision=.5, min_recall=.5):
    """Maximize F2 among feasible tuning thresholds; never fall back to all-negative."""
    evaluate_classifier(y, probabilities)  # Validate finite, binary inputs.
    if set(np.asarray(y)) != {0, 1}:
        raise ValueError("Threshold tuning needs both classes")
    if not 0 <= min_precision <= 1 or not 0 <= min_recall <= 1:
        raise ValueError("Precision and recall floors must lie between zero and one")
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    valid = (precision >= min_precision) & (recall >= min_recall) & (thresholds > 0) & (thresholds < 1)
    if not valid.any():
        return {"status": "blocked_no_feasible_threshold", "threshold": None}
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(precision), where=(4 * precision + recall) > 0)
    index = max(np.flatnonzero(valid), key=lambda i: (f2[i], precision[i], thresholds[i]))
    threshold = float(thresholds[index])
    return {"status": "selected_on_development_only", "threshold": threshold,
        "selection_metrics": evaluate_classifier(y, probabilities, threshold),
        "note": "These are tuning metrics, not independent evaluation or production accuracy."}


def fit_candidate(parts, features, target, trainer, *, min_precision=.5, min_recall=.5):
    check_partitions(parts, features, target)
    train, calibration, tuning = (parts[name] for name in ("training", "calibration", "threshold_tuning"))
    model = trainer(train, features, target)
    clean = lambda frame: frame[features].astype(float).replace([np.inf, -np.inf], np.nan)
    # FrozenEstimator keeps preprocessing and the classifier fixed on training rows.
    # Predict every calibration row once with the frozen base model. The empty
    # fit side is deliberate: FrozenEstimator.fit is a no-op, not a random CV fit.
    cv = [(np.array([], dtype=int), np.arange(len(calibration)))]
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", cv=cv, ensemble=False)
    calibrated.fit(clean(calibration), calibration[target].astype(int))
    raw = model.predict_proba(clean(tuning))[:, list(model.classes_).index(1)]
    probabilities = calibrated.predict_proba(clean(tuning))[:, list(calibrated.classes_).index(1)]
    selection = select_threshold(tuning[target], probabilities,
        min_precision=min_precision, min_recall=min_recall)
    diagnostics = {"threshold_selection": selection,
        "raw_default_threshold_metrics": evaluate_classifier(tuning[target], raw),
        "calibrated_default_threshold_metrics": evaluate_classifier(tuning[target], probabilities),
        "raw_brier": float(brier_score_loss(tuning[target], raw)),
        "calibrated_brier": float(brier_score_loss(tuning[target], probabilities)),
        "calibrated_log_loss": float(log_loss(tuning[target], probabilities, labels=[0, 1])),
        "calibration_fitted": True, "validated_calibration": False, "production_ready": False,
        "note": "Calibration is fitted, not independently validated; Brier loss alone does not establish calibration."}
    return calibrated, diagnostics
