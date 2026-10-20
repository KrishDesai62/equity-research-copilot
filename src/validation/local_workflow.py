"""Run cached research and optional SEC refresh/development fitting without AWS."""
import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo


@contextmanager
def local_only():
    previous = os.environ.get("EQUITY_LOCAL_ONLY")
    os.environ["EQUITY_LOCAL_ONLY"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EQUITY_LOCAL_ONLY", None)
        else:
            os.environ["EQUITY_LOCAL_ONLY"] = previous


def saved_evidence(path, candidate_hash):
    """Read saved evaluations, never generate predictions on exposed cohorts."""
    path = Path(path)
    if not path.exists():
        return {"status": "not_available", "path": str(path)}
    report = json.loads(path.read_text())
    recorded = report.get("candidate_artifact_sha256")
    if not candidate_hash or recorded != candidate_hash:
        raise ValueError(f"Saved evidence does not match the research candidate: {path}")
    return {"status": "saved_evidence_only", "path": str(path),
            "test_metrics": report.get("test_metrics"), "metrics": report.get("metrics"),
            "test_companies": report.get("test_companies"),
            "validated_calibration": report.get("validated_calibration", False),
            "limitations": report.get("limitations")}


def run(*, as_of=None, refresh=False, train_development=False,
        config="config/research_recent.json", ticker=None, question=None, train_v2=False):
    today = datetime.now(ZoneInfo("America/Chicago")).date()
    cutoff = date.fromisoformat(as_of) if as_of else today
    if cutoff > today:
        raise ValueError("As-of date cannot be in the future")
    if bool(ticker) != bool(question):
        raise ValueError("Provide both --ticker and --question for cited local Q&A")
    settings = json.loads(Path(config).read_text())
    if cutoff < date.fromisoformat(settings.get("start", "2010-01-01")):
        raise ValueError("As-of date precedes the research configuration start")
    output = Path("data/processed/validation/local_runs") / f"{cutoff}_{uuid4().hex}"
    output.mkdir(parents=True)
    report = {"as_of": cutoff.isoformat(), "mode": "local_only", "aws_required": False,
        "aws_deployed": False, "network_requested": refresh, "holdout_rescored": False,
        "production_ready": False, "run_directory": str(output), "steps": {}}
    with local_only(), (output / "details.log").open("w") as log, redirect_stdout(log):
        from src.data_layer.research_dataset import write_json
        try:
            if refresh:
                # Generate a run-specific config without changing the user's file.
                from src.validation.update_recent import run as update
                settings["observation_end"] = cutoff.isoformat()
                effective = output / "research_config.json"
                write_json(effective, settings)
                result = update(config_path=str(effective))
                report["steps"]["refresh"] = {"status": "complete", "publication": result["publication"]}
            root = Path("data/processed/research")
            required = [root / name for name in ("summary.json", "financial_features.csv", "filings.sqlite")]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                report.update(status="needs_local_data", missing_files=missing,
                    next_step="Run with --refresh to download public SEC sources; configure EDGAR_USER_AGENT in .env.")
            else:
                from src.validation.production_readiness import run as readiness
                result = readiness(cutoff.isoformat())
                report["steps"]["validation"] = {key: result[key] for key in
                    ("data_issues", "model_release", "research_snapshot_age_days", "unfinished_product_components")}
                if train_development:
                    from src.validation.calibration_development import run as fit
                    fitted = fit(include_expansion=True)
                    report["steps"]["development"] = {"status": fitted["status"],
                        "note": "Frozen candidates are verified and reused, never silently retuned.",
                        "validated_calibration": fitted["validated_calibration"]}
                if train_v2:
                    if not Path("data/processed/validation/improvement_v2/data_manifest.json").exists():
                        raise ValueError("Prepare V2 sources first: python -m src.validation.improve_v2 --prepare")
                    from src.validation.improve_v2 import fit
                    fitted_v2 = fit()
                    report["steps"]["v2_development"] = {"status": fitted_v2["status"],
                        "selected_candidate": fitted_v2["candidate"]["selected_candidate"],
                        "production_ready": False, "note": "Frozen research candidate verified; no final tests rerun."}
                v2_path = Path("data/processed/validation/improvement_v2/final_result.json")
                if v2_path.exists():
                    from src.validation.improve_v2 import summary
                    latest = summary()
                    report["steps"]["v2_research_assessment"] = latest["assessment"]
                if Path("data/processed/validation/cohort_v3/final_evaluation.json").exists():
                    from src.validation.cohort_v3 import summary as v3_summary
                    report["steps"]["v3_design_weighted_research_assessment"] = v3_summary()
                candidate = Path("data/processed/validation/calibration_expanded/candidate.joblib")
                candidate_hash = sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
                report["steps"]["independent_test"] = saved_evidence(
                    "data/processed/validation/independent_candidate_20261017/final_result.json", candidate_hash)
                report["steps"]["population_calibration"] = saved_evidence(
                    "data/processed/validation/population_calibration_20261017/report.json", candidate_hash)
                if question:
                    from src.agents.qa_agent import answer_question
                    from src.rag.local_index import LocalIndex
                    report["answer"] = answer_question(question, ticker, as_of=cutoff.isoformat(),
                        retriever=LocalIndex(root / "filings.sqlite").retrieve)
                report.update(status="completed_research_only",
                    next_step="Collect additional reviewed population outcomes before recalibration; no production model was promoted.")
            write_json(output / "report.json", report)
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(output / "report.json", report)
            raise
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="YYYY-MM-DD; defaults to today's date in America/Chicago")
    parser.add_argument("--refresh", action="store_true", help="Download missing public SEC sources and publish a complete local dataset")
    parser.add_argument("--train-development", action="store_true", help="Fit/verify the existing research candidate; never rerun final tests")
    parser.add_argument("--train-v2", action="store_true", help="Fit/verify V2 using already prepared development data; never rerun final tests")
    parser.add_argument("--config", default="config/research_recent.json")
    parser.add_argument("--ticker")
    parser.add_argument("--question")
    args = vars(parser.parse_args())
    try:
        result = run(**args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Local workflow failed: {exc}\n")
    # Research completion is separate from production release readiness.
    if result["status"] == "needs_local_data":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
