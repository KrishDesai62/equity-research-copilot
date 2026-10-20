"""Local tabular neural network shared by distress and manipulation tracks."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def validate_training_data(df, feature_cols, label_col):
    if df.empty or not feature_cols or label_col in feature_cols:
        raise ValueError("Provide nonempty training data and features excluding the target")
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Feature columns must be unique")
    missing = X.columns[X.isna().all()].tolist()
    if missing:
        raise ValueError(f"Features have no observed training values: {missing}")
    y = df[label_col]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Training requires both binary classes (0 and 1), with no missing labels")
    return X, y


def train_neural_network(train_df, feature_cols, label_col, *, hidden_layer_sizes=(32, 16),
                         max_iter=300, random_state=42):
    X, y = validate_training_data(train_df, feature_cols, label_col)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter, random_state=random_state, early_stopping=False,
            alpha=0.01))])
    # No random validation split: all preprocessing and fitting use training rows only.
    model.fit(X, y, classifier__sample_weight=compute_sample_weight("balanced", y))
    return model
