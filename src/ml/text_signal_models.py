"""Research classifiers that preserve sparse public filing-event features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.ml.artifacts import load_model
from src.ml.robust_models import QuantileClipper
from src.validation.sec_disclosure_data import FEATURES as FINANCIAL, TARGET
from src.validation.sec_filing_signals import FEATURES as FILING
from src.validation.filing_text_signals import FEATURES as TEXT

FEATURES = FINANCIAL + FILING + TEXT


class MetadataPreservingClipper(QuantileClipper):
    """Clip financial outliers while retaining rare event counts and indicators."""
    def __init__(self, financial_columns=21, lower=.01, upper=.99):
        super().__init__(lower, upper)
        self.financial_columns = financial_columns

    def fit(self, X, y=None):
        super().fit(X, y)
        if not 0 < self.financial_columns <= self.n_features_in_:
            raise ValueError('Invalid financial feature boundary')
        self.lower_[self.financial_columns:] = -np.inf
        self.upper_[self.financial_columns:] = np.inf
        return self


class FeatureSubsetReference:
    def __init__(self, model, columns):
        self.model, self.columns = model, list(columns)
        self.classes_ = model.classes_

    def predict_proba(self, X):
        selected = X[self.columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X)[:, :len(self.columns)], columns=self.columns)
        return self.model.predict_proba(selected)


def fit_candidate(frame, name):
    if name == 'previous_frozen_reference':
        root = Path('data/processed/validation/sec_disclosure_v1/experiment')
        selection = json.loads((root / 'selection.json').read_text())
        artifact = load_model(root / 'candidate.joblib', expected_sha256=selection['artifact_sha256'])
        if artifact['feature_cols'] != FINANCIAL or artifact['target'] != TARGET:
            raise ValueError('Reference schema/target mismatch')
        return FeatureSubsetReference(artifact['model'].model, FINANCIAL)
    y = frame[TARGET].to_numpy(int)
    weight = frame.design_weight.to_numpy(float).copy()
    if set(y) != {0, 1} or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError('Invalid training outcomes or weights')
    weight /= weight.mean()
    balance = {c: weight.sum() / (2 * weight[y == c].sum()) for c in (0, 1)}
    if name.startswith('logistic'):
        classifier = LogisticRegression(C=1. if name == 'logistic_c1_balanced' else .1,
            class_weight=balance if name.endswith('balanced') else None, max_iter=10000, random_state=42)
    elif name.startswith('xgb'):
        from xgboost import XGBClassifier
        ratio = weight[y == 0].sum() / weight[y == 1].sum()
        classifier = XGBClassifier(n_estimators=200, learning_rate=.03, max_depth=2,
            min_child_weight=5, reg_lambda=10, reg_alpha=1, subsample=.8, colsample_bytree=.8,
            scale_pos_weight=np.sqrt(ratio) if name.endswith('sqrt_balance') else 1., n_jobs=2,
            random_state=42, eval_metric='logloss')
    elif name == 'neural_network':
        classifier = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1., max_iter=2000,
            batch_size=min(64, len(frame)), random_state=42, early_stopping=False)
    elif name == 'hist_gradient_balanced':
        classifier = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=7,
            learning_rate=.03, min_samples_leaf=15, l2_regularization=10, early_stopping=False, random_state=42)
        weight *= np.array([balance[c] for c in y])
    elif name == 'random_forest_balanced':
        classifier = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=4,
            max_features='sqrt', n_jobs=2, random_state=42)
        weight *= np.array([balance[c] for c in y])
    else:
        raise ValueError('Unknown filing-signal model')
    model = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
        ('clipper', MetadataPreservingClipper(financial_columns=len(FINANCIAL))),
        ('scaler', RobustScaler()), ('classifier', classifier)])
    model.fit(frame[FEATURES], y, classifier__sample_weight=weight)
    return model
