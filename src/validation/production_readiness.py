"""Report production blockers from real evidence; never confuse tests with accuracy."""
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.release import assess
from src.validation.dataset_audit import audit_features


def run(as_of):
    root = Path("data/processed/research")
    final_path = Path("data/processed/validation/reserved_20261017/final_result.json")
    independent = Path("data/processed/validation/independent_candidate_20261017/final_result.json")
    if independent.exists():
        candidate = Path("data/processed/validation/calibration_expanded/candidate.joblib")
        result = json.loads(independent.read_text())
        if not candidate.exists() or sha256(candidate.read_bytes()).hexdigest() != result["candidate_artifact_sha256"]:
            raise ValueError("Independent evaluation does not match the current frozen candidate")
        final_path = independent
    evidence = json.loads(final_path.read_text()) if final_path.exists() else {}
    policy = json.loads(Path("config/release_policy.json").read_text())
    release = assess(evidence, policy)
    frame = pd.read_csv(root / "financial_features.csv", dtype={"cik": str})
    issues = audit_features(frame, as_of)
    summary = json.loads((root / "summary.json").read_text())
    if not summary.get("complete"):
        issues.append("Research source refresh incomplete")
    age = (date.fromisoformat(as_of) - date.fromisoformat(summary["observation_end"])).days
    if age < 0 or age > 7:
        issues.append("Research snapshot is future-dated or more than seven days old")
    report = {"as_of": as_of, "production_ready": release["eligible"] and not issues,
        "model_release": release, "data_issues": issues, "research_snapshot_age_days": age,
        "aws_deployed": False, "final_evaluation": evidence,
        "evaluation_evidence_path": str(final_path),
        "unfinished_product_components": ["Authenticated service and operator access controls", "Live AWS validation and alert delivery",
            "Representative recent outcomes and independent probability calibration", "Numerical Q&A and retrieval relevance evaluation",
            "Synthesis/portfolio/backtest agents remain outside the validated product scope"],
        "note": "Passing unit tests establishes software checks, not production model fitness."}
    report["production_ready"] = False  # Product components above are not yet delivered.
    candidate_path = Path("data/processed/validation/calibration_expanded/report.json")
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text())
        report["development_candidate"] = {key: candidate.get(key) for key in
            ("status", "partitions", "calibration_fitted", "validated_calibration", "threshold_selection", "candidate")}
        report["development_candidate"]["note"] = "Development tuning results; does not supersede the exposed final result or approve production."
    calibration_path = Path("data/processed/validation/population_calibration_20261017/report.json")
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        candidate_file = Path("data/processed/validation/calibration_expanded/candidate.joblib")
        if sha256(candidate_file.read_bytes()).hexdigest() != calibration["candidate_artifact_sha256"]:
            raise ValueError("Population calibration audit does not match the frozen candidate")
        report["population_calibration_audit"] = {key: calibration[key] for key in
            ("metrics", "validated_calibration", "representative_population", "limitations", "decision")}
    if Path("data/processed/validation/improvement_v2/final_result.json").exists():
        from src.validation.improve_v2 import summary
        report["v2_research_assessment"] = summary()["assessment"]
    if Path("data/processed/validation/cohort_v3/final_evaluation.json").exists():
        from src.validation.cohort_v3 import summary as v3_summary
        report["v3_design_weighted_research_assessment"] = v3_summary()
    write_json(Path("data/processed/validation/production_readiness.json"), report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Explicit YYYY-MM-DD in the operator's timezone")
    result = run(**vars(parser.parse_args()))
    if not result["production_ready"]:
        raise SystemExit(2)
