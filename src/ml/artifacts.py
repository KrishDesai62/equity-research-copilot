"""Persist trusted local model artifacts with their ordered input schema."""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
from datetime import date
import os
from tempfile import NamedTemporaryFile
import joblib


def save_model(model, feature_cols, path, *, trained_through, target):
    if not feature_cols or not trained_through or not target:
        raise ValueError("Feature schema, training cutoff and target are required")
    date.fromisoformat(str(trained_through))
    if len(feature_cols) != len(set(feature_cols)) or target in feature_cols:
        raise ValueError("Invalid artifact feature schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        joblib.dump({"schema_version": 1, "model": model, "feature_cols": list(feature_cols),
                     "trained_through": str(trained_through), "target": target}, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def load_model(path, *, expected_sha256=None):
    # joblib/pickle files must originate from trusted local training.
    path = Path(path)
    payload = path.read_bytes()
    if expected_sha256 is not None and sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Model artifact integrity check failed")
    # Deserialize the exact bytes checked, even if another process replaces path.
    artifact = joblib.load(BytesIO(payload))
    if not isinstance(artifact, dict) or not {"model", "feature_cols", "trained_through", "target"} <= artifact.keys():
        raise ValueError("Model artifact is missing schema or training metadata")
    features = artifact["feature_cols"]
    if not features or len(features) != len(set(features)) or artifact["target"] in features:
        raise ValueError("Invalid artifact feature schema")
    date.fromisoformat(artifact["trained_through"])
    return artifact
