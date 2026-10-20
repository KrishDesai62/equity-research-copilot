"""Point-in-time inference from trusted local model artifacts; never mock scores."""
from pathlib import Path
import json
import logging
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from src.ml.artifacts import load_model
from src.ml.release import validate_release
from src.ml.risk_contracts import target_definition

logger = logging.getLogger(__name__)


class RiskAgentTool:
    def __init__(self, model_registry_path="src/ml/models/", features_path="data/processed/features/risk.csv", *,
                 require_release=True, max_feature_age_days=180, max_missing_fraction=.25):
        self.model_registry_path = Path(model_registry_path)
        self.features_path = Path(features_path)
        if max_feature_age_days < 0 or not 0 <= max_missing_fraction <= 1:
            raise ValueError("Invalid feature freshness or missingness limit")
        self.require_release = require_release
        self.max_feature_age_days = max_feature_age_days
        self.max_missing_fraction = max_missing_fraction

    def get_risk_profile(self, ticker, as_of_date, *, features_df=None):
        frame = features_df.copy() if features_df is not None else pd.read_csv(self.features_path)
        cutoff = pd.Timestamp(as_of_date)
        dates = pd.to_datetime(frame["filing_date"], errors="raise")
        if pd.isna(cutoff) or dates.isna().any():
            raise ValueError("As-of and filing dates must be present")
        eligible = frame[(frame["ticker"].str.upper() == ticker.upper()) & (dates <= cutoff)]
        if eligible.empty:
            raise ValueError(f"No public features for {ticker} as of {as_of_date}")
        row = eligible.assign(filing_date=pd.to_datetime(eligible["filing_date"])).sort_values("filing_date").tail(1)
        latest_date = row.iloc[0].filing_date
        if (pd.to_datetime(eligible.filing_date) == latest_date).sum() != 1:
            raise ValueError("Ambiguous latest feature rows; reconcile duplicate filings before inference")
        if "latest_input_filing_date" in row:
            input_date = pd.Timestamp(row.iloc[0].latest_input_filing_date)
            if pd.isna(input_date):
                raise ValueError("Feature input publication date is missing")
            if input_date > latest_date:
                raise ValueError("Feature inputs were published after the feature filing date")
        output = {"ticker": ticker.upper(), "as_of_date": str(cutoff.date()),
                  "feature_filing_date": str(row.iloc[0]["filing_date"].date()),
                  "distress_probability": None, "manipulation_probability": None,
                  "manipulation_flag": None, "beneish_m_score": None,
                  "audit_status": "UNAVAILABLE", "unavailable_tracks": [], "unavailable_reasons": {},
                  "model_versions": {}, "calibration_status": {},
                  "prediction_targets": {}, "target_definitions": {},
                  "inference_mode": "production" if self.require_release else "research"}
        for track in ("distress", "manipulation"):
            path = self.model_registry_path / f"{track}.joblib"
            if not path.exists():
                output["unavailable_tracks"].append(track)
                output["unavailable_reasons"][track] = "model_not_available"
                continue
            release = None
            if self.require_release:
                release_path = path.with_suffix(".release.json")
                if not release_path.exists():
                    output["unavailable_tracks"].append(track)
                    output["unavailable_reasons"][track] = "validated_release_required"
                    continue
                release = validate_release(release_path, path)
                if pd.Timestamp(release["released_on"]) > cutoff:
                    raise ValueError("Production release postdates requested as-of date")
                if any(key not in row or pd.isna(row.iloc[0][key]) or not str(row.iloc[0][key]).strip()
                       for key in ("cik", "accession_number", "source_url", "latest_input_filing_date")):
                    output["unavailable_tracks"].append(track)
                    output["unavailable_reasons"][track] = "feature_provenance_required"
                    continue
                source = urlsplit(str(row.iloc[0].source_url))
                if source.scheme not in {"http", "https"} or not source.hostname:
                    output["unavailable_tracks"].append(track)
                    output["unavailable_reasons"][track] = "feature_provenance_required"
                    continue
            artifact = load_model(path, expected_sha256=release["artifact_sha256"] if release else None)
            if self.require_release:
                definition = target_definition(track, artifact["target"])
            else:
                definition = "Unreleased research target; outcome meaning has not been approved"
            output["prediction_targets"][track] = artifact["target"]
            output["target_definitions"][track] = definition
            if pd.Timestamp(artifact["trained_through"]) > cutoff:
                raise ValueError(f"{track} model was trained after the requested as-of date")
            if any(feature not in row.columns for feature in artifact["feature_cols"]):
                output["unavailable_tracks"].append(track)
                output["unavailable_reasons"][track] = "feature_schema_mismatch"
                continue
            X = row[artifact["feature_cols"]].astype(float).replace([np.inf, -np.inf], np.nan)
            reason = None
            if (cutoff - pd.Timestamp(row.iloc[0].filing_date)).days > self.max_feature_age_days:
                reason = "stale_financial_features"
            elif X.isna().mean(axis=1).iloc[0] > self.max_missing_fraction:
                reason = "insufficient_observed_features"
            if reason:
                output["unavailable_tracks"].append(track)
                output["unavailable_reasons"][track] = reason
                continue
            model = artifact["model"]
            classes = list(model.classes_)
            if set(classes) != {0, 1} or len(classes) != 2:
                raise ValueError("Risk model must predict binary classes 0 and 1")
            probability = float(model.predict_proba(X)[0, classes.index(1)])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Model returned an invalid probability")
            output[f"{track}_probability"] = probability
            output["model_versions"][track] = release["artifact_sha256"] if release else "research_unreleased"
            output["calibration_status"][track] = "validated" if release else "unvalidated_research_score"
            if track == "manipulation":
                output["manipulation_flag"] = probability >= (release["threshold"] if release else .5)
        if "beneish_m_score" in row and pd.notna(row.iloc[0]["beneish_m_score"]):
            output["beneish_m_score"] = float(row.iloc[0]["beneish_m_score"])
        if not output["unavailable_tracks"]:
            output["audit_status"] = "COMPLETED_DETERMINISTIC"
        elif len(output["unavailable_tracks"]) == 1:
            output["audit_status"] = "PARTIAL"
        logger.info("risk_inference %s", json.dumps({key: output[key] for key in
            ("ticker", "as_of_date", "feature_filing_date", "audit_status", "unavailable_reasons", "model_versions")}))
        return output
