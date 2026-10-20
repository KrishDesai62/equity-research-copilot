"""Run reproducible local validation and write an explicit readiness report."""
import json
from pathlib import Path
import warnings

import joblib
from threadpoolctl import threadpool_limits

from src.data_layer.filing_changes import detect_latest_changes, render_markdown
from src.ml.train_distress import compare_models, train_logistic_baseline
from src.validation.audit_filings import audit_comparison
from src.validation.benchmark import run_benchmark
from src.validation.sec_pilot import FEATURES, prepare_pilot


def write_readable_report(summary, benchmark, comparison, destination="reports/validation.md"):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = summary["sec_pilot"]
    lines = ["# Local validation results", "", "**Readiness: not production validated.**", "",
        "The extraction and model-training paths executed successfully. The US pilot has provisional negative labels and too few independent events to establish prediction quality.",
        "", "## Real-data classifier diagnostic", "",
        "Source: [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), Tomczak (2016), DOI 10.24432/C5F600, CC BY 4.0.",
        "", "Used `5year.arff`, whose forecasting case predicts bankruptcy after one year. No horizons were mixed. Exact feature duplicates were removed before splitting. Fixed stratified 60/20/20 training/validation/test partitions were used; preprocessing was fit on training rows only.",
        "", "| Model | Validation PR-AUC |", "| --- | ---: |"]
    for name, value in benchmark["models"].items():
        lines.append(f"| {name} | {value['validation']['pr_auc']:.4f} |")
    test = benchmark["selected_model_test"]
    lines.extend(["", f"Selected on validation: **{benchmark['selected_model']}**. Held-out test PR-AUC: **{test['pr_auc']:.4f}**, compared with a constant-score baseline of **{test['naive_pr_auc']:.4f}**.",
        "", f"Test observations: {benchmark['split_counts']['test']['rows']}; positive outcomes: {benchmark['split_counts']['test']['positives']}.",
        "", "The benchmark has no company IDs or filing timestamps. Company separation and chronological validity cannot be established. These results are diagnostic and do not demonstrate US-market generalization or calibrated bankruptcy probabilities.",
        "", "## SEC extraction pilot", "",
        f"Downloaded Company Facts for {audit['company_count']} purposively selected companies, including historical bankrupt entities. Extracted **{audit['raw_filing_rows']} original annual filings**; **{audit['uncensored_pre_event_rows']} rows** remain after post-bankruptcy and incomplete-horizon exclusions.",
        "", "| Feature | Missing fraction |", "| --- | ---: |"])
    for feature, missing in audit["missing_feature_fraction"].items():
        lines.append(f"| {feature} | {missing:.1%} |")
    lines.extend(["", "Facts are selected by original accession, public filing date, fiscal end, USD unit, and annual duration. Cell-level source tags and values are retained in `fact_provenance.csv`.",
        "", "Used four accounting ratios. Historical market capitalization was not sourced, so the market-equity ratio is excluded. This is not a complete Altman Z-score. Operating income is a documented EBIT proxy.",
        "", "## Original-filing reconciliation and disclosure checks", "",
        "Independently parsed the original non-dimensional inline-XBRL financial values in four SEC filings:", "",
        "| Company | Original filing checks | Result |", "| --- | ---: | --- |"])
    for name, value in summary["original_filing_checks"].items():
        lines.append(f"| {name} | {value['checks']} | {'All matched' if value['all_match'] else 'Failed'} |")
    lines.extend(["", "Apple 2023/2024 and Bed Bath & Beyond 2021/2022 disclosure comparisons both extracted Risk Factors and MD&A. Each produced two changed sections and zero unavailable sections. Reports retain original excerpts and SEC URLs; this verifies text extraction and differences, not semantic financial materiality.",
        "", "## US labels and evaluation", "",
        f"The annual pilot contains **{audit['positive_rows']} positive rows across {audit['positive_companies']} companies**. Five bankruptcy dates have primary SEC citations in the pilot configuration. A cited positive event does not prove complete historical event coverage for negative examples.",
        "", "Negative pilot labels are explicitly provisional. Production training rejects those rows. The existing uncited/secondary-source bankruptcy CSV is not silently treated as a complete registry.",
        "", "Company cohorts are held out across all their history; yearly test windows follow at least five initial training years. Training labels are purged until their availability dates. This fixes the earlier evaluator that erased the training set when companies recurred each quarter.",
        "", "The exploratory US run is saved for pipeline inspection only:", "",
        "| Model | Mixed-class test folds | Out-of-sample rows | Positive out-of-sample rows |", "| --- | ---: | ---: | ---: |"])
    for name, value in comparison["models"].items():
        lines.append(f"| {name} | {value['total_folds_evaluated']} | {len(value['predictions'])} | {sum(p['target'] for p in value['predictions'])} |")
    lines.extend(["", "These counts are too small for a trustworthy model ranking. Any exploratory PR-AUC in the JSON must not be represented as production validation. Some neural-network folds reached their iteration limit; warnings are retained in the report.",
        "", "No diagnostic model was promoted into the risk agent's production model registry. Benchmark and smoke-test artifacts remain under `data/processed/validation/`.",
        "", "## Remaining gates", ""])
    lines.extend("- " + gate for gate in summary["remaining_gates"])
    lines.extend(["", "## Reproduce", "", "```bash", ".venv/bin/python -m pytest -q", ".venv/bin/python -m src.validation.run", "```", "",
        "The SEC contact is configured in the gitignored `.env`; it is not included in this report. The second command reuses cached data. A fresh run requires network access. Detailed outputs and source checksums are in `data/processed/validation/`."])
    if Path("data/processed/validation/model_readiness/report.json").exists():
        lines[2:2] = ["**Current US model status:** these are historical diagnostics. See [model_readiness.md](model_readiness.md) for source reconciliation and reviewed-label availability.", ""]
    path.write_text("\n".join(lines) + "\n")


def main():
    root = Path("data/processed/validation")
    root.mkdir(parents=True, exist_ok=True)
    features, audit = prepare_pilot()
    benchmark = run_benchmark()
    original_audits = {}
    for name, cik, cutoff in (("apple", "0000320193", "2024-12-31"),
                               ("bbby", "0000886158", "2022-12-31")):
        report_path = root / f"{name}_annual_changes.json"
        if not report_path.exists():
            report = detect_latest_changes(cik, as_of=cutoff)
            report_path.write_text(json.dumps(report, indent=2))
            report_path.with_suffix(".md").write_text(render_markdown(report))
        original_audits[name] = audit_comparison(report_path)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # This is explicitly a pipeline diagnostic using provisional outcomes.
        # Never promote its fitted model to the production risk-agent registry.
        comparison = compare_models(features, FEATURES, "target_distressed_1yr",
            label_available_col="label_available_date", period_freq="Y", min_train_periods=5)
        smoke = train_logistic_baseline(features, FEATURES, "target_distressed_1yr")
        joblib.dump({"model": smoke, "feature_cols": FEATURES,
                     "purpose": "Extraction pilot smoke test; provisional labels, NOT production inference"},
                    root / "sec_pilot/logistic_smoke.joblib")
    comparison_report = {"production_ready": False,
        "warning": "Exploratory pipeline diagnostic only: provisional negative labels, a purposive ten-company sample and four positive annual observations.",
        "protocol": "Company-disjoint cohorts, yearly test windows, five initial training years, purged label availability",
        "training_warnings": sorted({str(w.message) for w in caught}), "models": comparison}
    (root / "sec_pilot/exploratory_comparison.json").write_text(json.dumps(comparison_report, indent=2, allow_nan=False))
    summary = {"production_ready": False, "benchmark": benchmark["selected_model_test"],
               "benchmark_selected_model": benchmark["selected_model"], "sec_pilot": audit,
               "original_filing_checks": {name: {"checks": len(value["checks"]), "all_match": value["all_match"]}
                                          for name, value in original_audits.items()},
               "remaining_gates": ["Review complete bankruptcy-event coverage before using any pilot negative labels in production.",
                                   "Expand a documented historical universe with more independent bankruptcy events and untouched final holdout companies.",
                                   "Source point-in-time market capitalization before using the full five-component Altman feature set.",
                                   "Quarterly/TTM features and independent final US holdout validation remain outstanding."]}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_readable_report(summary, benchmark, comparison_report)
    print("Validation finished; production_ready=False. See data/processed/validation/summary.json")
    return summary


if __name__ == "__main__":
    main()
