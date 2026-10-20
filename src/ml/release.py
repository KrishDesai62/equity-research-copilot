"""Fail-closed release assessment. Diagnostic experiments never auto-promote."""
import argparse
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path

from src.data_layer.research_dataset import write_json


def assess(evidence, policy):
    failures = []
    metrics = evidence.get("test_metrics", {})
    for metric in ("precision", "recall"):
        value = metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            failures.append(f"Missing or invalid held-out {metric}")
        elif value < policy[f"min_{metric}"]:
            failures.append(f"Held-out {metric} {value:.3f} below release floor {policy[f'min_{metric}']:.3f}")
    for key, floor in (("test_companies", "min_test_companies"), ("independent_positive_events", "min_independent_positive_events")):
        value = evidence.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < policy[floor]:
            failures.append(f"Insufficient independently evaluated {key}")
    for name in ("independent_test", "representative_population", "validated_calibration"):
        if policy.get("require_" + name) and evidence.get(name) is not True:
            failures.append(f"Required evidence absent: {name}")
    if evidence.get("preprocessing_fit_on") != "training rows only":
        failures.append("Training-only preprocessing evidence is missing")
    return {"eligible": not failures, "failures": failures, "policy_version": policy["policy_version"],
        "evidence_sha256": sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        "policy_sha256": sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()}


def validate_release(path, artifact_path):
    """Validate a locally trusted release record before unpickling any artifact."""
    release = json.loads(Path(path).read_text())
    try:
        date.fromisoformat(release["released_on"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Production release requires a valid released_on date") from exc
    if release.get("status") != "approved" or release.get("calibrated") is not True:
        raise ValueError("Production model has no approved calibrated release")
    if release.get("artifact_sha256") != sha256(Path(artifact_path).read_bytes()).hexdigest():
        raise ValueError("Production model artifact integrity check failed")
    assessment = release.get("assessment", {})
    policy = json.loads((Path(__file__).resolve().parents[2] / "config/release_policy.json").read_text())
    recomputed = assess(release.get("evidence", {}), policy)
    if assessment != recomputed or not recomputed["eligible"]:
        raise ValueError("Production release did not pass validation")
    threshold = release.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Production release requires a valid decision threshold")
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--policy", default="config/release_policy.json")
    parser.add_argument("--output", default="data/processed/validation/release_assessment.json")
    args = parser.parse_args()
    if args.evidence is None:
        independent = Path("data/processed/validation/independent_candidate_20261017/final_result.json")
        args.evidence = str(independent if independent.exists() else Path("data/processed/validation/reserved_20261017/final_result.json"))
    result = assess(json.loads(Path(args.evidence).read_text()), json.loads(Path(args.policy).read_text()))
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    if not result["eligible"]:
        raise SystemExit(2)
