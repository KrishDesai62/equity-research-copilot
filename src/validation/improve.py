"""Reproducible feature/tuning experiment with a precommitted company holdout.

All US negative labels remain provisional pending independent outcome-coverage
review. This experiment measures pipeline behavior; it cannot authorize deployment.
"""
import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data_layer.filing_changes import normalize_filing
from src.ml.evaluate import evaluate_classifier, compare_to_naive_baseline
from src.ml.financial_history import extract_history, BASE_FEATURES, EXTRA_FEATURES
from src.ml.neural_network import train_neural_network, validate_training_data
from src.ml.train_distress import run_walk_forward_evaluation, train_logistic_baseline, train_xgboost
from src.validation.download import companyfacts, fetch
from src.validation.sec_pilot import label_rows
from src.data_layer.research_dataset import load_submissions


def commit_split(config, destination):
    """A split cannot change when adding features or looking at outcomes."""
    policy = config["holdout_policy"]
    if not 0 < policy["fraction"] < 1 or len(config["companies"]) < 2:
        raise ValueError("Holdout needs at least two companies and a fraction strictly between zero and one")
    companies = sorted({str(c["cik"]).zfill(10) for c in config["companies"]},
        key=lambda cik: sha256(f"{policy['seed']}:{cik}".encode()).hexdigest())
    count = max(1, int(np.ceil(len(companies) * policy["fraction"])))
    split = {"policy": policy, "holdout_companies": companies[:count], "development_companies": companies[count:]}
    path = Path(destination)
    if path.exists() and json.loads(path.read_text()) != split:
        raise ValueError("Committed holdout differs. Use a new experiment directory; never silently reshuffle a test set")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return split


def prepare(config, output):
    frames, provenance, sources, conflicts, event_checks = [], [], [], [], []
    for company in config["companies"]:
        payload, source = companyfacts(company["cik"])
        sources.append(source)
        if company.get("event_source"):
            path = Path("data/raw/validation/events") / (company["ticker"] + ".html")
            sources.append(fetch(company["event_source"], path))
            text = " ".join(normalize_filing(path.read_text()).split())
            expected = pd.Timestamp(company["event_date"]).strftime("%B %d, %Y").replace(" 0", " ")
            event_checks.append({"ticker": company["ticker"], "cik": company["cik"],
                "event_date": company["event_date"], "source_url": company["event_source"],
                "date_text_present": expected in text, "chapter_11_present": "chapter 11" in text.lower()})
        public_filings, _ = load_submissions(company, "2010-01-01", config["observation_end"],
            Path("data/raw/research"), sources)
        frame, cells, issues = extract_history(payload, company["ticker"], public_filings=public_filings)
        if frame["latest_input_filing_date"].gt(frame["filing_date"]).any():
            raise ValueError("Future input leaked into historical features")
        frames.append(label_rows(frame, company, config["observation_end"]))
        provenance.extend(cells)
        conflicts.extend(issues)
        print(company["ticker"], len(frame), "filing rows", flush=True)
    all_rows = pd.concat(frames, ignore_index=True)
    # First original disclosure per fiscal period; later duplicates are not independent examples.
    all_rows = all_rows.sort_values("filing_date").drop_duplicates(["cik", "fiscal_period_end"], keep="first")
    usable = all_rows.loc[all_rows.exclusion_reason.eq("")].copy().reset_index(drop=True)
    all_rows.to_csv(output / "all_rows.csv", index=False)
    usable.to_csv(output / "features.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "fact_provenance.csv", index=False)
    pd.DataFrame(event_checks).to_csv(output / "event_source_checks.csv", index=False)
    audit = {"raw_rows": len(all_rows), "usable_rows": len(usable), "companies": usable.cik.nunique(),
        "positive_rows": int(usable.target_distressed_1yr.sum()),
        "independent_positive_companies": usable.loc[usable.target_distressed_1yr.eq(1), "cik"].nunique(),
        "exclusions": all_rows.exclusion_reason.value_counts().to_dict(), "source_manifest": sources,
        "missing_fraction": usable[BASE_FEATURES + EXTRA_FEATURES].isna().mean().to_dict(),
        "conflicting_facts": conflicts, "event_source_checks": event_checks,
        "production_ready": False, "negative_label_status": "coverage_review_required"}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    return usable, audit


def tuned_xgb(df, features, target, **params):
    from xgboost import XGBClassifier
    X, y = validate_training_data(df, features, target)
    model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", XGBClassifier(scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
            eval_metric="logloss", random_state=42, n_jobs=2, **params))])
    return model.fit(X, y)


def choose_threshold(targets, scores, minimum_precision=.5):
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    eligible = np.flatnonzero(precision[:-1] >= minimum_precision)
    if not len(eligible):
        return {"threshold": .5, "status": "precision target not achieved; default threshold"}
    index = max(eligible, key=lambda i: (recall[i], precision[i]))
    return {"threshold": float(thresholds[index]), "status": "selected on development predictions only",
            "development_precision": float(precision[index]), "development_recall": float(recall[index])}


def company_bootstrap_delta(frame, baseline, selected, repetitions=300):
    """Paired cluster bootstrap keeps all quarterly observations of a firm together."""
    rng = np.random.default_rng(42)
    companies = frame.cik.unique()
    targets = frame.target_distressed_1yr.to_numpy(dtype=int)
    differences = []
    for _ in range(repetitions):
        indices = np.concatenate([np.flatnonzero(frame.cik.to_numpy() == company)
                                  for company in rng.choice(companies, len(companies), replace=True)])
        if len(np.unique(targets[indices])) < 2:
            continue
        differences.append(average_precision_score(targets[indices], selected[indices]) -
                           average_precision_score(targets[indices], baseline[indices]))
    return {"valid_resamples": len(differences), "pr_auc_difference_95pct_interval":
            np.quantile(differences, [.025, .975]).tolist() if differences else None}


def experiment(frame, config, split, output):
    if (output / "experiment.json").exists():
        raise ValueError("This holdout has already been evaluated. Read the saved result; do not retune against it")
    cutoff = pd.Timestamp(split["policy"]["development_cutoff"])
    test_start = pd.Timestamp(split["policy"]["test_start"])
    if test_start < cutoff:
        raise ValueError("Final test period cannot start before the development cutoff")
    dates, available = pd.to_datetime(frame.filing_date), pd.to_datetime(frame.label_available_date)
    development = frame.loc[frame.cik.isin(split["development_companies"]) & (dates < cutoff) & (available < cutoff)].reset_index(drop=True)
    final = frame.loc[frame.cik.isin(split["holdout_companies"]) & (dates >= test_start)].reset_index(drop=True)
    if development.target_distressed_1yr.nunique() != 2:
        raise ValueError("Committed development split lacks both classes; do not tune on final holdout")
    enhanced = BASE_FEATURES + EXTRA_FEATURES
    candidates = {
        "baseline_logistic": (BASE_FEATURES, train_logistic_baseline),
        "baseline_xgboost": (BASE_FEATURES, train_xgboost),
        "enhanced_logistic": (enhanced, train_logistic_baseline),
        "enhanced_xgb_depth2": (enhanced, partial(tuned_xgb, max_depth=2, n_estimators=150, learning_rate=.05, reg_lambda=3)),
        "enhanced_xgb_depth3": (enhanced, partial(tuned_xgb, max_depth=3, n_estimators=200, learning_rate=.03, min_child_weight=3, reg_lambda=3)),
        "enhanced_xgb_depth4": (enhanced, partial(tuned_xgb, max_depth=4, n_estimators=150, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=3)),
        "enhanced_neural_network": (enhanced, partial(train_neural_network, hidden_layer_sizes=(16, 8), max_iter=600))}
    results = {}
    with threadpool_limits(limits=2):
        for name, (features, trainer) in candidates.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_walk_forward_evaluation(development, features, "target_distressed_1yr",
                    trainer=trainer, label_available_col="label_available_date", period_freq="Y", min_train_periods=4)
            result["training_warnings"] = sorted({str(w.message) for w in caught})
            result["features"] = features
            results[name] = result
            print(name, "development mean PR-AUC:", result["mean_pr_auc"], flush=True)
        eligible = {name: result for name, result in results.items() if result["mean_pr_auc"] is not None}
        if not eligible:
            raise ValueError("No usable development folds; final holdout will not be evaluated")
        winner = max(eligible, key=lambda name: eligible[name]["mean_pr_auc"])
        chosen = results[winner]
        threshold = choose_threshold([p["target"] for p in chosen["predictions"]], [p["score"] for p in chosen["predictions"]])
        selection = {"selected_candidate": winner, "selection_metric": "development mean fold PR-AUC",
                     "features": candidates[winner][0], "threshold": threshold, "split": split,
                     "development_rows": len(development), "development_positive_companies": development.loc[development.target_distressed_1yr.eq(1), "cik"].nunique()}
        # Freeze feature/model/threshold choices BEFORE computing final holdout scores.
        (output / "selection_before_test.json").write_text(json.dumps(selection, indent=2, allow_nan=False))
        report = {"production_ready": False, "label_status": "Provisional negative labels; case-control sample; diagnostic comparison only",
                  "configuration_sha256": sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "selection": selection, "development_results": results,
                  "holdout_rows": len(final), "holdout_companies": final.cik.nunique(),
                  "holdout_positive_companies": final.loc[final.target_distressed_1yr.eq(1), "cik"].nunique()}
        if final.target_distressed_1yr.nunique() != 2:
            report["final_status"] = "inconclusive: committed holdout lacks both classes; split was not changed"
        else:
            probabilities = {}
            for name in dict.fromkeys(["baseline_xgboost", winner]):
                features, trainer = candidates[name]
                model = trainer(development, features, "target_distressed_1yr")
                scores = model.predict_proba(final[features])[:, 1]
                probabilities[name] = scores
                report[name + "_holdout"] = {
                    **evaluate_classifier(final.target_distressed_1yr, scores, threshold["threshold"] if name == winner else .5),
                    **compare_to_naive_baseline(final.target_distressed_1yr, scores),
                    "brier_score": float(brier_score_loss(final.target_distressed_1yr, scores)),
                    "threshold": threshold["threshold"] if name == winner else .5}
                joblib.dump({"model": model, "feature_cols": features, "purpose": "Expanded diagnostic; provisional outcomes, not production"}, output / (name + ".joblib"))
            report["final_status"] = "evaluated once after development selection; outcome coverage remains provisional"
            report["paired_company_bootstrap"] = company_bootstrap_delta(final, probabilities["baseline_xgboost"], probabilities[winner])
            final_predictions = final[["cik", "ticker", "filing_date", "target_distressed_1yr"]].copy()
            for name, scores in probabilities.items():
                final_predictions[name + "_score"] = scores
            final_predictions.to_csv(output / "holdout_predictions.csv", index=False)
    (output / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def audit_completed_experiment(config, split, output):
    """Audit corrected data without another look at held-out model performance."""
    from src.validation.audit_filings import audit_comparison
    from src.validation.flow_audit import audit_trailing_flows
    quality = output.parent / (output.name + "_quality_check")
    quality.mkdir(parents=True, exist_ok=True)
    corrected, _ = prepare(config, quality)
    original = pd.read_csv(output / "features.csv", dtype={"cik": str})
    fields = BASE_FEATURES + EXTRA_FEATURES
    joined = original[["cik", "accession_number", "filing_date", "label_available_date"] + fields].merge(
        corrected[["cik", "accession_number"] + fields], on=["cik", "accession_number"],
        suffixes=("_original", "_corrected"), validate="one_to_one")
    changes = {field: ~np.isclose(joined[field + "_original"], joined[field + "_corrected"], equal_nan=True)
               for field in fields}
    changed = np.logical_or.reduce(list(changes.values()))
    dates = pd.to_datetime(joined.filing_date)
    available = pd.to_datetime(joined.label_available_date)
    cutoff = pd.Timestamp(split["policy"]["development_cutoff"])
    used = ((joined.cik.isin(split["development_companies"]) & (dates < cutoff) & (available < cutoff)) |
            (joined.cik.isin(split["holdout_companies"]) & (dates >= pd.Timestamp(split["policy"]["test_start"]))))
    summary = {"old_rows": len(original), "corrected_rows": len(corrected), "common_rows": len(joined),
        "changed_rows": int(changed.sum()), "changed_modeling_rows": int((changed & used).sum()),
        "changed_values_by_feature": {field: int(mask.sum()) for field, mask in changes.items()},
        "original_experiment_inputs_unchanged": not (changed & used).any() and len(joined) == len(original) == len(corrected),
        "holdout_rescored": False}
    (quality / "feature_change_audit.json").write_text(json.dumps(summary, indent=2))
    flow_audit = audit_trailing_flows(pd.read_csv(quality / "all_rows.csv", dtype={"cik": str}))
    (quality / "flow_audit.json").write_text(json.dumps(flow_audit, indent=2, allow_nan=False))
    summary["flow_checks"] = flow_audit["check_count"]
    summary["flow_mismatches"] = flow_audit["mismatches"]
    summary["original_filing_checks"] = 0
    for name in ("apple", "bbby"):
        path = output.parent / (name + "_annual_changes.json")
        if path.exists():
            source_audit = audit_comparison(path, quality / "fact_provenance.csv", destination=quality / (name + "_source_audit.json"))
            summary["original_filing_checks"] += len(source_audit["checks"])
    review_path = quality / "outcome_coverage_review.csv"
    if not review_path.exists():
        pd.DataFrame([{"cik": c["cik"], "ticker": c["ticker"], "requested_coverage_end": config["observation_end"],
                       "event_source": c.get("event_source"), "coverage_start": None, "reviewed_through": None,
                       "coverage_source": None, "review_status": "pending", "event_coverage_reviewed": False}
                      for c in config["companies"]]).to_csv(review_path, index=False)
    (quality / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    if not summary["original_experiment_inputs_unchanged"] or summary["flow_mismatches"]:
        raise ValueError("Data audit failed or corrections affect modeling inputs. Keep old scores separate; do not silently rescore the exposed holdout")
    return summary


def write_report(audit, report, destination="reports/improvement_experiment.md", *, quality=None):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = report["selection"]["selected_candidate"]
    lines = ["# Expanded US model improvement experiment", "",
        "**Outcome: data coverage improved; predictive improvement was not demonstrated. Production readiness remains false.**", "",
        "## Step 1: expand sources", "",
        f"Expanded from 10 to {audit['companies']} companies. Downloaded original SEC Company Facts histories and primary SEC disclosures for 21 cited bankruptcy dates. This is a purposive case-control sample, not a representative historical investment universe.", "",
        f"Extracted {audit['raw_rows']} distinct original annual/quarterly filing periods; {audit['usable_rows']} remain after post-event and outcome-follow-up exclusions. There are {audit['positive_rows']} positive rows across {audit['independent_positive_companies']} companies. Those rows do not represent {audit['positive_rows']} independent events.", "",
        "Automated source checks found the specified event date and Chapter 11 text in each cited disclosure. They do not establish complete negative-event coverage. Negative labels remain provisional, and the diagnostic models are excluded from the production registry.", "",
        "## Step 2: financial history", "",
        "Added cash/assets, liabilities/assets, liquidity, operating cash flow, margins, an explicitly named earnings proxy, interest coverage, and year-over-year changes. All sources use original accessions and public filing dates. Year-to-date flows are differenced into standalone quarters, then four consecutive quarters are summed. Annual flows are used directly. Missing quarters leave TTM values missing.", "",
        "| Measure on the expanded sample | Missing fraction |", "| --- | ---: |",
        f"| Operating-income / assets | {audit['missing_fraction']['ebit_to_ta']:.1%} |",
        f"| Explicit earnings proxy / assets | {audit['missing_fraction']['earnings_proxy_to_assets']:.1%} |",
        f"| Consolidated operating cash flow / assets | {audit['missing_fraction']['cashflow_to_assets']:.1%} |", "",
        "The earnings proxy prefers operating profit; when unavailable it uses pretax income plus disclosed interest expense. The method is recorded. It is not silently relabeled as operating income. Continuing-operations cash flow is not silently substituted for consolidated cash flow, so some cash-flow values remain missing.", "",
        "## Step 3: reserve the test set", "",
        "Company IDs were assigned by a seeded hash, without using outcome labels or financial values. All history of reserved companies was excluded from development. Training and development labels had to be available before 2020-01-01. The final period starts on 2020-01-01.", "",
        f"The development sample contains {report['selection']['development_rows']} rows and {report['selection']['development_positive_companies']} positive companies. Final test: {report['holdout_rows']} rows from {report['holdout_companies']} reserved companies, including {report['holdout_positive_companies']} positive companies.", "",
        "## Step 4: compare features and tune on development only", "",
        "Used chronological yearly windows crossed with company holdout cohorts. Imputation, scaling, and class weights were fit on each training fold. Compared the four-ratio baselines with enhanced logistic regression, three prespecified XGBoost configurations, and a small neural network.", "",
        "| Candidate | Development mean fold PR-AUC |", "| --- | ---: |"]
    for name, result in report["development_results"].items():
        metric = result["mean_pr_auc"]
        lines.append(f"| {name} | {metric:.4f} |" if metric is not None else f"| {name} | Inconclusive |")
    lines.extend(["", f"Selected **{winner}** on development results. Extra features and a larger model did not help in this experiment. Model, feature, and threshold choices were saved before test scoring.", "",
        "## Step 5: final holdout", "", "| Model | Test PR-AUC | Precision | Recall | Brier score |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in dict.fromkeys(["baseline_xgboost", winner]):
        result = report.get(name + "_holdout")
        if result:
            lines.append(f"| {name} | {result['pr_auc']:.4f} | {result['precision']:.3f} | {result['recall']:.3f} | {result['brier_score']:.4f} |")
    result = report.get("baseline_xgboost_holdout")
    if result:
        interval = report["paired_company_bootstrap"]["pr_auc_difference_95pct_interval"]
        lines.extend(["", f"The test constant-score PR-AUC baseline is {result['naive_pr_auc']:.4f}. The paired company-bootstrap interval for selected-model PR-AUC minus baseline XGBoost is [{interval[0]:.4f}, {interval[1]:.4f}]. It includes zero; this is not reliable evidence of improvement.", "",
            "The threshold sought 50% precision on development predictions. That target was not achieved, so the documented fallback was 0.5. Test thresholds were not optimized. Brier scores describe uncalibrated predictions on a selected sample; they do not establish deployable bankruptcy probabilities."])
    if quality:
        lines.extend(["", "## Step 6: regression and source audits", "",
            f"Cross-checked {quality['flow_checks']} trailing-flow values against the independent annual-plus-YTD-minus-prior-YTD identity: {quality['flow_mismatches']} mismatches. Short reorganization/transition periods are excluded from full-year comparisons.", "",
            f"Reconciled {quality['original_filing_checks']} raw financial values against the original Apple and Bed Bath & Beyond inline-XBRL documents; all matched.", "",
            f"A transition-period correction changed {quality['changed_rows']} historical rows but **{quality['changed_modeling_rows']} modeling rows**. Every development and final-test input was checked unchanged, so the reported experiment remains the same. The held-out predictions were not recomputed.", "",
            "The coverage-review worksheet is saved under `data/processed/validation/expanded_quality_check/outcome_coverage_review.csv`; pending reviews were not automatically approved."])
    lines.extend(["", "## What remains", "",
        "- Independently review complete event coverage before using negative labels for production training.",
        "- Add independent bankruptcy events across more years; development has only five positive companies.",
        "- Audit additional source tags and use explicit continuing-operations features where consolidated cash flow is unavailable.",
        "- Reserve new, genuinely unseen companies before the next adaptive experiment. This test set is now exposed and cannot be reused as untouched evidence.",
        "- Evaluate text-derived features only after their as-of extraction and incremental benefit can be checked.", "",
        "## Files and reproduction", "",
        "Detailed data, provenance, source checks, the committed split, development results, selected configuration, and final predictions are in `data/processed/validation/expanded/`.", "",
        "```bash", ".venv/bin/python -m src.validation.improve", ".venv/bin/python -m src.validation.improve --audit-data", ".venv/bin/python -m pytest -q", "```", "",
        "A completed experiment is reused rather than retuned against the exposed holdout. No diagnostic model was promoted into the risk agent. Source citations are in `config/validation_expanded.json`; raw documents are cached locally."])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/validation_expanded.json")
    parser.add_argument("--output", default="data/processed/validation/expanded")
    parser.add_argument("--audit-data", action="store_true", help="Reaudit extraction and arithmetic without rescoring the completed holdout")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split = commit_split(config, output / "committed_holdout.json")
    if (output / "experiment.json").exists():
        audit = json.loads((output / "data_audit.json").read_text())
        report = json.loads((output / "experiment.json").read_text())
        if report.get("configuration_sha256") != sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
            raise ValueError("Configuration differs from the completed experiment; do not silently reuse or rescore its holdout")
        quality_path = output.parent / (output.name + "_quality_check") / "quality_summary.json"
        quality = audit_completed_experiment(config, split, output) if args.audit_data else (
            json.loads(quality_path.read_text()) if quality_path.exists() else None)
        if Path("data/processed/validation/model_readiness/report.json").exists():
            print("Historical experiment only: current source/label audit supersedes these scores. See reports/model_readiness.md")
            return
        write_report(audit, report, quality=quality)
        print("Reused completed experiment; holdout was not rescored. See reports/improvement_experiment.md")
        return
    frame, audit = prepare(config, output)
    report = experiment(frame, config, split, output)
    write_report(audit, report)
    print(json.dumps({"usable_rows": audit["usable_rows"], "positive_companies": audit["independent_positive_companies"],
                      "selected": report["selection"]["selected_candidate"], "final_status": report["final_status"]}, indent=2))


if __name__ == "__main__":
    main()
