"""Audit corrected US model inputs and report whether accuracy is measurable.

Never rescore an exposed holdout or silently replace an old result. An unavailable
accuracy is represented as null, not zero or a cached, invalidated percentage.
"""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.bankruptcy_labels import build_distress_labels
from src.data_layer.research_dataset import write_json
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES


def reviewed_labels(features, coverage, companies, observation_end):
    """Apply explicit, dated, cited coverage approvals; unknown stays unknown."""
    coverage = coverage.copy()
    coverage["cik"] = coverage.cik.astype(str).str.zfill(10)
    if coverage.cik.duplicated().any():
        raise ValueError("Duplicate CIKs in outcome coverage review")
    approved = coverage.event_coverage_reviewed.astype(str).str.lower().isin(["true", "1"])
    required = ["coverage_start", "observed_through", "reviewer", "coverage_source_url"]
    for column in required:
        if column not in coverage:
            raise ValueError(f"Missing coverage field: {column}")
        if coverage.loc[approved, column].isna().any() or coverage.loc[approved, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Approved coverage requires {column}")
    for column in ("coverage_start", "observed_through"):
        coverage[column] = pd.to_datetime(coverage[column], errors="raise")
    if (coverage.loc[approved, "coverage_start"] > coverage.loc[approved, "observed_through"]).any():
        raise ValueError("Coverage start follows observation end")
    if not coverage.loc[approved, "coverage_source_url"].astype(str).str.match(r"https?://\S+$").all():
        raise ValueError("Reviewed coverage needs source URLs")
    brd = coverage.coverage_source_url.astype(str).str.contains(r"lopucki\.law\.", regex=True)
    if (approved & brd & coverage.observed_through.gt(pd.Timestamp("2022-12-31"))).any():
        raise ValueError("LoPucki snapshot cannot establish outcome coverage after 2022-12-31")
    coverage["event_coverage_reviewed"] = approved
    frame = features.drop(columns=[c for c in ["coverage_start", "observed_through", "event_coverage_reviewed"] if c in features]).merge(
        coverage[["cik", "coverage_start", "observed_through", "event_coverage_reviewed"]], on="cik", how="left", validate="many_to_one")
    # No review may extend beyond the dataset's declared observation endpoint.
    frame["observed_through"] = pd.to_datetime(frame.observed_through).clip(upper=pd.Timestamp(observation_end))
    # Known positives can still be labeled through the snapshot even if coverage
    # is pending. Missing review never authorizes negative outcomes.
    frame["observed_through"] = frame.observed_through.fillna(pd.Timestamp(observation_end))
    events = pd.DataFrame([{"cik": str(c["cik"]).zfill(10), "filing_date": c["event_date"], "source_url": c["event_source"]}
        for c in companies if c.get("event_date") and c.get("event_source")], columns=["cik", "filing_date", "source_url"])
    result = build_distress_labels(frame, events, horizon_years=(1,), drop_invalid_rows=False)
    if "bankruptcy_label_conflict" in result:
        conflict = result.bankruptcy_label_conflict.astype(str).str.lower().isin(["true", "1"])
        result.loc[conflict, "target_distressed_1yr"] = pd.NA
        result.loc[conflict, "label_available_date_1yr"] = pd.NaT
        result.loc[conflict, "valid_for_training"] = False
    return result


def audit_old_inputs(original, corrected, split):
    """Check old train/test input identities and values without model inference."""
    keys = ["cik", "accession_number"]
    fields = BASE_FEATURES + EXTRA_FEATURES
    joined = original.merge(corrected[keys + fields], on=keys, how="left",
                            suffixes=("_old", "_corrected"), indicator=True, validate="one_to_one")
    missing = joined["_merge"].eq("left_only")
    changed = np.zeros(len(joined), dtype=bool)
    changes_by_feature = {}
    for field in fields:
        mask = ~np.isclose(joined[field + "_old"], joined[field + "_corrected"], equal_nan=True) & ~missing
        changed |= mask
        changes_by_feature[field] = int(mask.sum())
    dates, available = pd.to_datetime(joined.filing_date), pd.to_datetime(joined.label_available_date)
    cutoff = pd.Timestamp(split["policy"]["development_cutoff"])
    training = joined.cik.isin(split["development_companies"]) & dates.lt(cutoff) & available.lt(cutoff)
    test = joined.cik.isin(split["holdout_companies"]) & dates.ge(pd.Timestamp(split["policy"]["test_start"]))
    affected = missing | changed
    rows = joined.loc[affected, keys + ["filing_date"]].copy()
    rows["reason"] = np.where(missing[affected], "unreconciled_or_outside_verified_window", "derived_features_changed")
    rows["used_in_old_training"] = training[affected].to_numpy()
    rows["used_in_old_holdout"] = test[affected].to_numpy()
    return {"old_rows": len(original), "unmatched_rows": int(missing.sum()),
        "changed_common_rows": int(changed.sum()), "affected_training_rows": int((affected & training).sum()),
        "affected_holdout_rows": int((affected & test).sum()), "changes_by_feature": changes_by_feature,
        "holdout_rescored": False, "old_metrics_status": "superseded_unverified",
        "reason": "Original experiment used provisional negative outcomes; source reconciliation is now required."}, rows


def run(research="data/processed/research", old="data/processed/validation/expanded",
        config=None, output="data/processed/validation/model_readiness", evaluate=False):
    research, old, output = Path(research), Path(old), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config is None:
        snapshot = json.loads((research / "summary.json").read_text())
        matches = [p for p in (research / "research_config.json", Path("config/research_recent.json"), Path("config/research_universe.json"))
                   if p.exists() and sha256(p.read_bytes()).hexdigest() == snapshot["config_sha256"]]
        if not matches:
            raise ValueError("Specify the configuration matching the research snapshot")
        config = matches[0]
    settings = json.loads(Path(config).read_text())
    frame = pd.read_csv(research / "financial_features.csv", dtype={"cik": str})
    brd_path = Path("data/processed/lopucki/chapter11_events.csv")
    if brd_path.exists():
        from src.data_layer.lopucki import reconcile_events
        settings["companies"], _ = reconcile_events(settings["companies"], brd_path)
        resolved = {c["cik"] for c in settings["companies"] if c.get("event_date_policy")}
        if "bankruptcy_label_conflict" in frame:
            frame.loc[frame.cik.isin(resolved), "bankruptcy_label_conflict"] = False
    coverage = pd.read_csv(research / "outcome_coverage_review.csv", dtype={"cik": str})
    labeled = reviewed_labels(frame, coverage, settings["companies"], settings["observation_end"])
    from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry
    filings = pd.DataFrame([json.loads(line) for line in (research / "filings.jsonl").read_text().splitlines()])
    labeled, row_evidence = apply_historical_coverage(labeled, load_reviewed_registry(), filings,
        observation_end=settings["observation_end"])
    row_evidence.to_csv(output / "historical_coverage_evidence.csv", index=False)
    labeled.to_csv(output / "reviewed_features.csv", index=False)
    eligible = labeled.loc[labeled.valid_for_training].copy()
    positives = int(eligible.target_distressed_1yr.eq(1).sum())
    negatives = int(eligible.target_distressed_1yr.eq(0).sum())
    report = {"accuracy": None, "precision": None, "recall": None, "pr_auc": None,
        "status": "blocked_missing_reviewed_negative_labels" if not negatives else "ready_for_development_evaluation",
        "eligible_positive_rows": positives, "eligible_negative_rows": negatives,
        "unknown_rows": int(labeled.target_distressed_1yr.isna().sum()),
        "positive_companies": int(eligible.loc[eligible.target_distressed_1yr.eq(1), "cik"].nunique()),
        "models_retrained": False, "production_ready": False,
        "holdout_rescored": False, "review_file": str(research / "outcome_coverage_review.csv")}
    report["negative_label_scope"] = "BRD-eligible large public-company cohort, complete-identity 2010–2022 registry interval, annual size/status evidence; not all US companies or later years"
    if not positives:
        report["status"] = "blocked_missing_reviewed_positive_labels"
    if (old / "features.csv").exists():
        original = pd.read_csv(old / "features.csv", dtype={"cik": str})
        split = json.loads((old / "committed_holdout.json").read_text())
        audit, affected = audit_old_inputs(original, frame, split)
        affected.to_csv(output / "affected_old_inputs.csv", index=False)
        report["old_input_audit"] = audit
    if evaluate and positives and negatives:
        from src.ml.train_distress import compare_models
        # This is development cross-validation only. Previously held-out firms
        # cannot be relabeled as a fresh test set by changing the output path.
        reserved = split["holdout_companies"] if (old / "features.csv").exists() else []
        final_manifest = Path("config/reserved_cohort_20261017.json")
        if final_manifest.exists():
            reserved = set(reserved) | {c["cik"] for c in json.loads(final_manifest.read_text())["companies"]}
        development = eligible.loc[~eligible.cik.isin(reserved)].copy()
        development = development.loc[development.historical_coverage_eligible &
            pd.to_datetime(development.filing_date).lt("2020-01-01") &
            pd.to_datetime(development.label_available_date_1yr).lt("2020-01-01")].copy()
        try:
            report["development_comparison"] = compare_models(development, BASE_FEATURES + EXTRA_FEATURES,
                "target_distressed_1yr", label_available_col="label_available_date_1yr",
                period_freq="Y", min_train_periods=4)
            report["models_retrained"] = True
            report["status"] = "development_evaluated_not_independent_final_accuracy"
        except ValueError as exc:
            report["status"] = "blocked_insufficient_company_time_folds"
            report["evaluation_error"] = str(exc)
    write_json(output / "report.json", report)
    destination = Path("reports/model_readiness.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_path = Path("data/processed/validation/reserved_20261017/final_result.json")
    independent_path = Path("data/processed/validation/independent_candidate_20261017/final_result.json")
    if independent_path.exists():
        final_path = independent_path
    final_status = "Independent final accuracy has not yet been measured."
    if final_path.exists():
        final = json.loads(final_path.read_text())
        final_status = (f"The final cohort has been scored once: precision {final['test_metrics']['precision']:.3f}, "
            f"recall {final['test_metrics']['recall']:.3f}. It is exposed and must not be reused for tuning. "
            f"Evidence: {final_path}. See reports/production_readiness.md; production release remains blocked.")
    destination.write_text("# Current US model validation status\n\n"
        f"**Historical labels are available within reviewed coverage. {final_status} Previous accuracy figures remain superseded diagnostics.**\n\n"
        "SEC submissions now filter raw inputs before historical features are computed, preventing amendments and conflicting dates from influencing later ratios.\n\n"
        f"Eligible one-year labels: {positives} positive rows across {report['positive_companies']} companies, {negatives} reviewed negative rows. {report['unknown_rows']} rows remain unlabeled.\n\n"
        "Negative outcomes are source-reviewed only within the BRD's large-public-company scope and 2010–2022 interval. Row-level annual filing and asset evidence is exported separately; other unknown outcomes stay unknown.\n\n"
        "The audit in `data/processed/validation/model_readiness/report.json` lists affected old model inputs without rescoring the exposed holdout. Existing experimental artifacts and scores are preserved as historical records, not current validated performance.\n\n"
        "Complete cited bankruptcy-event coverage in `data/processed/research/outcome_coverage_review.csv`: coverage_start, observed_through, reviewer, coverage_source_url, and event_coverage_reviewed. A citation to a financial filing alone does not establish complete bankruptcy follow-up. Missing required evidence fails validation.\n\n"
        "After adequate reviewed examples of both classes exist, run company/time-separated development comparisons for logistic regression, XGBoost and the neural network. Reserve genuinely unseen firms before a final evaluation; the previous holdout cannot supply new independent evidence. Accounting enforcement and financial Q&A are separate evaluation tasks.\n")
    print(json.dumps({k: v for k, v in report.items() if k != "development_comparison"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", default="data/processed/research")
    parser.add_argument("--old", default="data/processed/validation/expanded")
    parser.add_argument("--config")
    parser.add_argument("--output", default="data/processed/validation/model_readiness")
    parser.add_argument("--evaluate", action="store_true", help="Compare all three models on reviewed development labels only")
    args = vars(parser.parse_args())
    report = run(**args)
    if args["evaluate"] and report["status"].startswith("blocked_"):
        raise SystemExit(2)
