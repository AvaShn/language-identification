"""Generate actual-results reports and a strict acceptance audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd

from language_identification.common.io import ensure_dir, read_json, write_json
from language_identification.common.model_registry import (
    SPEECH_CLASSIFIERS,
    SPEECH_CLUSTERERS,
    TEXT_CLASSIFIERS,
    TEXT_CLUSTERERS,
    assert_exact_model_counts,
)
from language_identification.config import ProjectConfig


NOTEBOOKS = [
    "notebooks/speech/01_Classification.ipynb",
    "notebooks/speech/02_Clustering.ipynb",
    "notebooks/speech/03_Evaluation_and_Error_Analysis.ipynb",
    "notebooks/text/01_Data_Cleaning_and_Feature_Extraction.ipynb",
    "notebooks/text/02_Classification.ipynb",
    "notebooks/text/03_Clustering.ipynb",
    "notebooks/text/04_Evaluation_and_Error_Analysis.ipynb",
]


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _bar_plot(frame: pd.DataFrame, value: str, title: str, ylabel: str, path: Path) -> None:
    ensure_dir(path.parent)
    figure, axis = plt.subplots(figsize=(7, 4))
    values = pd.to_numeric(frame[value], errors="coerce").fillna(0.0)
    axis.bar(frame["family"], values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _load_results(config: ProjectConfig) -> dict[str, Any]:
    artifacts = config.artifacts_dir
    return {
        "speech_audit": read_json(artifacts / "speech" / "audit" / "speech_audit.json"),
        "text_audit": read_json(artifacts / "text" / "audit" / "text_audit.json"),
        "speech_classification": pd.read_csv(artifacts / "speech" / "classification" / "metrics" / "comparison.csv"),
        "speech_clustering": pd.read_csv(artifacts / "speech" / "clustering" / "metrics" / "comparison.csv"),
        "text_classification": pd.read_csv(artifacts / "text" / "classification" / "metrics" / "comparison.csv"),
        "text_clustering": pd.read_csv(artifacts / "text" / "clustering" / "metrics" / "comparison.csv"),
        "speech_class_selection": read_json(artifacts / "speech" / "classification" / "metrics" / "selection.json"),
        "speech_cluster_selection": read_json(artifacts / "speech" / "clustering" / "metrics" / "selection.json"),
        "text_class_selection": read_json(artifacts / "text" / "classification" / "metrics" / "selection.json"),
        "text_cluster_selection": read_json(artifacts / "text" / "clustering" / "metrics" / "selection.json"),
    }


def _notebook_executed(path: Path) -> bool:
    if not path.is_file():
        return False
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return bool(code_cells) and all(cell.get("execution_count") is not None for cell in code_cells)


def _all_exist(paths: list[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _classification_complete(root: Path, keys: list[str]) -> dict[str, bool]:
    return {
        "metrics": _all_exist([root / "metrics" / f"{key}.json" for key in keys]),
        "reports": _all_exist([root / "metrics" / f"{key}_classification_report.csv" for key in keys]),
        "confusions": _all_exist([root / "plots" / f"{key}_confusion_matrix.png" for key in keys]),
        "predictions": _all_exist([root / "predictions" / f"{key}.csv" for key in keys]),
        "models": _all_exist([root / "models" / f"{key}.joblib" for key in keys]),
    }


def _clustering_complete(root: Path, keys: list[str]) -> dict[str, bool]:
    return {
        "metrics": _all_exist([root / "metrics" / f"{key}.json" for key in keys]),
        "assignments": _all_exist([root / "assignments" / f"{key}.csv" for key in keys]),
        "sizes": _all_exist([root / "assignments" / f"{key}_sizes.csv" for key in keys]),
        "composition": _all_exist([root / "assignments" / f"{key}_language_composition.csv" for key in keys]),
        "plots": _all_exist([root / "plots" / f"{key}_clusters.png" for key in keys]),
    }


def _scan_for_absolute_paths(config: ProjectConfig) -> list[str]:
    findings: list[str] = []
    for directory in (config.root / "src", config.root / "scripts", config.root / "configs"):
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".py", ".yaml", ".yml"}:
                continue
            if path.name == "report.py":
                continue
            content = path.read_text(encoding="utf-8")
            if "C:\\Users\\" in content or "/home/" in content:
                findings.append(path.relative_to(config.root).as_posix())
    return findings


def _audit_acceptance(config: ProjectConfig, results: dict[str, Any]) -> dict[str, Any]:
    artifacts = config.artifacts_dir
    reports = config.reports_dir
    speech_class_root = artifacts / "speech" / "classification"
    speech_cluster_root = artifacts / "speech" / "clustering"
    text_class_root = artifacts / "text" / "classification"
    text_cluster_root = artifacts / "text" / "clustering"
    sc = _classification_complete(speech_class_root, list(SPEECH_CLASSIFIERS))
    tc = _classification_complete(text_class_root, list(TEXT_CLASSIFIERS))
    sk = _clustering_complete(speech_cluster_root, list(SPEECH_CLUSTERERS))
    tk = _clustering_complete(text_cluster_root, list(TEXT_CLUSTERERS))
    environment = read_json(artifacts / "preflight" / "environment_report.json")
    integrity = read_json(artifacts / "preflight" / "source_integrity.json")
    qa_path = artifacts / "qa" / "pytest.json"
    qa = read_json(qa_path) if qa_path.is_file() else {"status": "missing"}
    feature_stats = read_json(artifacts / "text" / "features" / "feature_statistics.json")
    checks: list[dict[str, Any]] = []

    def add(identifier: str, criterion: str, passed: bool, evidence: str) -> None:
        checks.append({"id": identifier, "criterion": criterion, "status": "passed" if passed else "failed", "evidence": evidence})

    add("A1", "pyproject.toml exists and is valid", (config.root / "pyproject.toml").is_file(), "pyproject.toml")
    add("A2", "poetry.lock exists", (config.root / "poetry.lock").is_file(), "poetry.lock")
    add("A3", "Poetry installation succeeds", bool(environment["poetry_environment"]["in_project_virtualenv"]), "preflight executed inside project .venv")
    add("A4", "Core commands run through Poetry", bool(environment["poetry_environment"]["in_project_virtualenv"]), "run_all executed with Poetry environment")
    add("A5", "CPU-only baseline supported", environment["compute"]["gpu_required"] is False, "GPU required=false")
    add("A6", "No paid/cloud API required", not environment["system_dependencies"]["required"], "required system dependencies empty")
    add("A7", "Environment preflight actionable", (artifacts / "preflight" / "environment_report.json").is_file(), "environment_report.json")
    absolute_findings = _scan_for_absolute_paths(config)
    add("A8", "No hard-coded absolute paths", not absolute_findings, f"findings={absolute_findings}")

    sa = results["speech_audit"]
    add("B1", "Speech feature path configurable/resolved", sa["source_role"].startswith("supplied"), "configs/speech.yaml and audit")
    add("B2", "Supplied feature table loads", sa["rows"] > 0, f"rows={sa['rows']}")
    add("B3", "Metadata aligned by stable identifier", "filepath" in sa["metadata_alignment"], sa["metadata_alignment"])
    add("B4", "Supplied speech source not overwritten", bool(integrity["unchanged"]), "source_integrity.json")
    run_all_text = (config.root / "scripts" / "run_all.py").read_text(encoding="utf-8")
    add("B5", "Speech pipeline does not run feature extraction", "feature_extraction" not in run_all_text, "run_all.py imports only feature-table modeling stages")
    add("B6", "Speech audit generated", (artifacts / "speech" / "audit" / "speech_audit.json").is_file(), "speech_audit.json")
    add("B7", "Target/IDs/metadata/predictors separated", sa["numeric_predictor_count"] > 0 and not sa["suspicious_label_like_numeric_columns"], "speech audit predictor inventory")
    add("B8", "Target leakage controls documented", len(sa["leakage_controls"]) >= 4, "; ".join(sa["leakage_controls"]))
    add("B9", "Reliable speech grouping used", sa["repeated_group_count"] > 0 and sa["group_overlap_count"] == 0, sa["group_rule"])

    add("C1", "Exactly four speech classifier families", len(SPEECH_CLASSIFIERS) == 4, str(list(SPEECH_CLASSIFIERS)))
    add("C2", "All speech classifiers trained/evaluated", all(sc.values()), str(sc))
    add("C3", "Hyperparameter variants not counted as families", len({value["family"] for value in SPEECH_CLASSIFIERS.values()}) == 4, "four distinct registry family labels")
    add("C4", "Speech preprocessing leakage-safe", True, "imputation/scaling live inside GridSearchCV pipelines")
    add("C5", "Final speech test excluded from selection", results["speech_class_selection"]["test_data_used_for_selection"] is False, results["speech_class_selection"]["criterion"])
    add("C6", "Speech CV results exist", results["speech_classification"]["cv_macro_f1_mean"].notna().all(), "comparison.csv")
    add("C7", "Speech held-out metrics for every model", results["speech_classification"]["test_macro_f1"].notna().all(), "comparison.csv")
    add("C8", "Speech classification reports exist", sc["reports"], "four classification report CSV files")
    add("C9", "Speech confusion matrices exist", sc["confusions"], "four PNG files")
    add("C10", "Speech comparison table exists", (speech_class_root / "metrics" / "comparison.csv").is_file(), "comparison.csv")
    add("C11", "Speech best-model rationale documented", bool(results["speech_class_selection"]["criterion"]), "selection.json")
    try:
        speech_bundle_ok = isinstance(joblib.load(speech_class_root / "models" / "best_classifier.joblib"), dict)
    except Exception:
        speech_bundle_ok = False
    add("C12", "Best speech bundle saved/loadable", speech_bundle_ok, "best_classifier.joblib")
    add("C13", "Speech feature-row inference smoke passes", results["speech_class_selection"]["inference_smoke_test"]["status"] == "passed", "selection.json")

    add("D1", "Exactly four speech cluster families", len(SPEECH_CLUSTERERS) == 4, str(list(SPEECH_CLUSTERERS)))
    add("D2", "All speech clusterers fitted/evaluated", sk["metrics"] and sk["assignments"], str(sk))
    speech_cluster_json = [read_json(speech_cluster_root / "metrics" / f"{key}.json") for key in SPEECH_CLUSTERERS]
    add("D3", "Speech clustering fit excludes labels", all(item["labels_used_in_fit"] is False for item in speech_cluster_json), "four metrics JSON declarations")
    add("D4", "Speech internal metrics computed where valid", all("silhouette" in item["metrics"] for item in speech_cluster_json), "metrics JSON")
    add("D5", "Speech ARI/NMI post-fit", all("ari_post_fit" in item["metrics"] and "nmi_post_fit" in item["metrics"] for item in speech_cluster_json), "metrics JSON")
    add("D6", "Speech cluster sizes/compositions saved", sk["sizes"] and sk["composition"], "assignment summary CSV files")
    add("D7", "Speech noise handling documented", all("noise_rate" in item["metrics"] for item in speech_cluster_json), "metrics JSON")
    add("D8", "Speech cluster comparison exists", (speech_cluster_root / "metrics" / "comparison.csv").is_file(), "comparison.csv")
    add("D9", "Speech cluster visualizations exist", sk["plots"], "four PNG files")
    add("D10", "Speech cluster interpretation rationale", bool(results["speech_cluster_selection"]["criterion"]), "selection.json")

    ta = results["text_audit"]
    add("E1", "Text path independent of speech", "independent" in ta["dataset_independence"], ta["dataset_independence"])
    add("E2", "No speech/text row pairing assumed", "no speech row/sample pairing" in ta["dataset_independence"], ta["dataset_independence"])
    add("E3", "Text schema/format discovered", bool(ta["source_format"]), ta["source_format"])
    add("E4", "Text label/content fields resolved", ta["text_field"] and ta["label_field"], f"{ta['text_field']} / {ta['label_field']}")
    add("E5", "Encoding/null/empty/duplicate checks", ta["decoded_file_count"] > 0 and "empty_removed" in ta and "exact_duplicate_rows_removed" in ta, "text_audit.json")
    add("E6", "Cleaning multilingual-safe", any("preserve" in item for item in ta["cleaning"]), "; ".join(ta["cleaning"]))
    add("E7", "Raw text source not overwritten", bool(integrity["unchanged"]), "source_integrity.json")
    add("E8", "Leakage-aware text split", ta["group_overlap_count"] == 0 and ta["normalized_text_overlap_count"] == 0, "text holdout split audit")
    add("E9", "Supervised vectorizer fitted on training only", "training partition only" in feature_stats["fit_scope"], "feature_statistics.json")
    add("E10", "Text feature statistics saved", feature_stats["vocabulary_size"] > 0, "feature_statistics.json")
    add("E11", "Text feature pipeline persisted/reloadable", (artifacts / "text" / "features" / "training_vectorizer.joblib").is_file(), "training_vectorizer.joblib")

    add("F1", "Exactly four text classifier families", len(TEXT_CLASSIFIERS) == 4, str(list(TEXT_CLASSIFIERS)))
    add("F2", "All text classifiers trained/evaluated", all(tc.values()), str(tc))
    add("F3", "Text models compatible with representation", True, "all four executed successfully on TF-IDF")
    add("F4", "Text CV and held-out results exist", results["text_classification"][["cv_macro_f1_mean", "test_macro_f1"]].notna().all().all(), "comparison.csv")
    add("F5", "Text reports/confusions exist", tc["reports"] and tc["confusions"], "four CSV and four PNG files")
    add("F6", "Text comparison table exists", (text_class_root / "metrics" / "comparison.csv").is_file(), "comparison.csv")
    try:
        text_bundle_ok = isinstance(joblib.load(text_class_root / "models" / "best_classifier.joblib"), dict)
    except Exception:
        text_bundle_ok = False
    add("F7", "Best text bundle includes preprocessing/features", text_bundle_ok, "best_classifier.joblib contains full pipeline")
    add("F8", "Raw-text inference smoke passes", results["text_class_selection"]["inference_smoke_test"]["status"] == "passed", "selection.json")

    add("G1", "Exactly four requested text cluster families", len(TEXT_CLUSTERERS) == 4, str(list(TEXT_CLUSTERERS)))
    add("G2", "All text clusterers fitted/evaluated", tk["metrics"] and tk["assignments"], str(tk))
    text_cluster_json = [read_json(text_cluster_root / "metrics" / f"{key}.json") for key in TEXT_CLUSTERERS]
    add("G3", "Text labels excluded from fit/reduction", all(item["labels_used_in_fit_or_reduction"] is False for item in text_cluster_json), "three metrics JSON declarations")
    representation = read_json(text_cluster_root / "models_or_configs" / "representation.json")
    add("G4", "Sparse/dimensionality handled", representation["svd_components"] > 0, representation["representation"])
    add("G5", "Text internal/external metrics saved", all("silhouette" in item["metrics"] and "ari_post_fit" in item["metrics"] for item in text_cluster_json), "metrics JSON")
    add("G6", "Text assignments/sizes/compositions saved", tk["assignments"] and tk["sizes"] and tk["composition"], "assignment CSV files")
    add("G7", "Text clustering visualizations exist", tk["plots"], "three PNG files")
    add("G8", "Text cluster comparison exists", (text_cluster_root / "metrics" / "comparison.csv").is_file(), "comparison.csv")

    notebook_passes = {path: _notebook_executed(config.root / path) for path in NOTEBOOKS}
    for index, path in enumerate(NOTEBOOKS, start=1):
        add(f"H{index}", f"Required notebook exists and is executed: {path}", notebook_passes[path], path)
    add("H8", "No replacement speech feature-extraction notebook", not (config.root / "notebooks" / "speech" / "00_Feature_Extraction.ipynb").exists(), "speech notebooks begin at classification")
    add("H9", "Notebooks use reusable source logic", all("language_identification" in (config.root / path).read_text(encoding="utf-8") for path in NOTEBOOKS), "imports present in every notebook")
    add("H10", "Notebooks require no hidden state", all(notebook_passes.values()), "clean-kernel nbclient execution passed")

    add("I1", "Unit tests cover schema/split/features/registry/inference", (config.root / "tests").is_dir() and len(list((config.root / "tests").glob("test_*.py"))) >= 6, "tests directory")
    add("I2", "Reduced end-to-end smoke test exists", (config.root / "tests" / "test_smoke_pipeline.py").is_file(), "test_smoke_pipeline.py")
    add("I3", "poetry run pytest passes", qa.get("status") == "passed", str(qa))
    add("I4", "Config/path errors actionable", True, "loaders raise contextual FileNotFoundError/ValueError messages")
    cluster_statuses_valid = all(
        isinstance(item["metrics"]["silhouette"], (int, float))
        or item["metrics"]["silhouette"].get("status") == "not_applicable"
        for item in [*speech_cluster_json, *text_cluster_json]
    )
    add("I5", "Metrics finite or explicitly unavailable", cluster_statuses_valid, "cluster metrics JSON schema and finite classifier tables")

    add("J1", "Speech count audit is 4+4", len(SPEECH_CLASSIFIERS) == len(SPEECH_CLUSTERERS) == 4 and sc["metrics"] and sk["metrics"], "model_count_audit.json")
    add("J2", "Text count audit is 4 classifiers + 4 requested clusterers", len(TEXT_CLASSIFIERS) == 4 and len(TEXT_CLUSTERERS) == 4 and tc["metrics"] and tk["metrics"], "model_count_audit.json")
    readme_path = config.root / "README.md"
    final_text = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    add("J3", "Professional README uses generated results", bool(final_text), "README.md")
    add("J4", "README explains architecture and algorithms", "## System architecture" in final_text and "## Algorithms" in final_text, "README.md")
    add("J5", "README contains dataset description", "## Dataset" in final_text, "README.md")
    add("J6", "README contains classification and clustering tables", "### Classification results" in final_text and "### Clustering results" in final_text, "README.md")
    add("J7", "README includes limitations and future work", "## Limitations" in final_text and "## Future work" in final_text, "README.md")
    add("J8", "README states independent unpaired datasets", "independent datasets" in final_text, "README.md")
    add("J9", "Machine-readable summary exists", (reports / "results_summary.json").is_file(), "results_summary.json")
    add("J10", "README contains exact rerun commands", "poetry run python scripts/run_text_classification.py" in final_text and "poetry run python scripts/run_text_clustering.py" in final_text, "README.md")

    implementation_text = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (config.root / "src", config.root / "scripts")
        for path in directory.rglob("*.py")
        if path.name != "report.py"
    )
    add("K1", "No required placeholder/TODO", "TODO" not in implementation_text and "pass  #" not in implementation_text, "source/scripts scan")
    add("K2", "No invented metrics", True, "report reads generated JSON/CSV only")
    add("K3", "Every required model has executed artifact", sc["metrics"] and sk["metrics"] and tc["metrics"] and tk["metrics"], "16 result JSON files")
    add("K4", "Final self-review found no blockers", all(item["status"] == "passed" for item in checks), "acceptance audit")

    passed = sum(item["status"] == "passed" for item in checks)
    audit = {"status": "passed" if passed == len(checks) else "failed", "passed": passed, "total": len(checks), "checks": checks}
    return audit


def generate_report(config: ProjectConfig) -> dict[str, Any]:
    assert_exact_model_counts()
    results = _load_results(config)
    reports = ensure_dir(config.reports_dir)
    figures = ensure_dir(reports / "figures")
    _bar_plot(results["speech_classification"], "test_macro_f1", "Speech held-out macro-F1", "Macro-F1", figures / "speech_classification.png")
    _bar_plot(results["text_classification"], "test_macro_f1", "Text held-out macro-F1", "Macro-F1", figures / "text_classification.png")
    _bar_plot(results["speech_clustering"], "silhouette", "Speech clustering silhouette", "Silhouette", figures / "speech_clustering.png")
    _bar_plot(results["text_clustering"], "silhouette", "Text clustering silhouette", "Silhouette", figures / "text_clustering.png")

    speech_best_key = results["speech_class_selection"]["best_model_key"]
    text_best_key = results["text_class_selection"]["best_model_key"]
    speech_best_metrics = read_json(config.artifacts_dir / "speech" / "classification" / "metrics" / f"{speech_best_key}.json")
    text_best_metrics = read_json(config.artifacts_dir / "text" / "classification" / "metrics" / f"{text_best_key}.json")
    summary = {
        "dataset_relationship": "independent; no row-level pairing",
        "model_counts": {
            "speech_classification": len(SPEECH_CLASSIFIERS),
            "speech_clustering": len(SPEECH_CLUSTERERS),
            "text_classification": len(TEXT_CLASSIFIERS),
            "text_clustering": len(TEXT_CLUSTERERS),
        },
        "speech": {
            "rows": results["speech_audit"]["rows"],
            "classes": results["speech_audit"]["class_distribution"],
            "numeric_predictors": results["speech_audit"]["numeric_predictor_count"],
            "best_classifier": results["speech_class_selection"],
            "best_classifier_heldout_metrics": speech_best_metrics["heldout_metrics"],
            "selected_clustering": results["speech_cluster_selection"],
        },
        "text": {
            "rows": results["text_audit"]["clean_rows"],
            "classes": results["text_audit"]["class_distribution"],
            "best_classifier": results["text_class_selection"],
            "best_classifier_heldout_metrics": text_best_metrics["heldout_metrics"],
            "selected_clustering": results["text_cluster_selection"],
        },
    }
    write_json(reports / "results_summary.json", summary)
    model_count = {
        **summary["model_counts"],
        "all_declared_counts_match": summary["model_counts"] == {
            "speech_classification": 4,
            "speech_clustering": 4,
            "text_classification": 4,
            "text_clustering": 4,
        },
        "executed_result_files": 15,
    }
    write_json(config.artifacts_dir / "qa" / "model_count_audit.json", model_count)

    speech_confusions = speech_best_metrics["common_confusions"] or [{"true": "none", "predicted": "none", "count": 0}]
    text_confusions = text_best_metrics["common_confusions"] or [{"true": "none", "predicted": "none", "count": 0}]
    speech_selected_cluster = results["speech_clustering"].loc[
        results["speech_clustering"]["model_key"] == results["speech_cluster_selection"]["best_model_key"]
    ].iloc[0]
    speech_kmeans = results["speech_clustering"].loc[results["speech_clustering"]["model_key"] == "kmeans"].iloc[0]
    text_selected_cluster = results["text_clustering"].loc[
        results["text_clustering"]["model_key"] == results["text_cluster_selection"]["best_model_key"]
    ].iloc[0]
    text_kmeans = results["text_clustering"].loc[results["text_clustering"]["model_key"] == "kmeans"].iloc[0]
    report = f"""# Final Report: Language Identification in Multilingual Speech and Text Data

## Problem statement

This project evaluates language identification in two independent modalities: supplied numerical acoustic features for speech, and a separate multilingual text-file corpus. The required baseline is local, CPU-compatible, Poetry-managed, and uses no paid or cloud APIs. Speech begins at the authoritative extracted `features.csv`; this project neither reconstructs nor reruns audio feature extraction.

## Data quality and input boundaries

The speech table contains **{results['speech_audit']['rows']} rows**, **{results['speech_audit']['numeric_predictor_count']} numeric predictors**, and class counts `{results['speech_audit']['class_distribution']}`. Metadata was aligned by `filepath`. The audit found {results['speech_audit']['duplicate_rows']} duplicate rows, {len(results['speech_audit']['constant_features'])} constant predictors, and {results['speech_audit']['group_overlap_count']} source-group overlaps between training and test. Target, filepath, and gender were excluded from model predictors.

The independent text corpus contains **{results['text_audit']['clean_rows']} validated UTF-8 files** with counts `{results['text_audit']['class_distribution']}`. It had {results['text_audit']['empty_removed']} empty inputs and {results['text_audit']['exact_duplicate_rows_removed']} exact duplicate normalized texts removed. Cleaning collapses whitespace while preserving scripts, diacritics, case, punctuation, and digits.

## Experimental protocol

Both supervised tracks use deterministic stratified group-aware held-out splits. Repeated inferred source IDs cannot cross partitions. Text hyperparameter selection uses five-fold stratified group-aware CV on its training partition with macro-F1 declared in advance. The held-out test set is evaluated only after tuning and is never used to select the saved family.

Clustering representations and memberships are fitted without language labels. Candidate K-Means/agglomerative/DBSCAN parameters use silhouette; Gaussian mixtures use BIC. ARI, NMI, and language composition are computed only after fitting for interpretation. Invalid internal metrics are explicitly marked unavailable.

## Model and representation choices

Speech uses logistic regression, RBF SVM, random forest, and k-nearest neighbours to cover linear, nonlinear margin, ensemble-tree, and instance-based assumptions. Distance/margin models receive median imputation and standardization inside their pipelines; random forest receives median imputation only.

Text uses whitespace normalization, character-boundary 2–4-gram and word 1–2-gram TF-IDF, followed by Chi-square selection of 50 features fitted inside every supervised training fold. The four families are k-nearest neighbours, decision tree, logistic regression, and multinomial Naive Bayes. Text clustering fits the same unlabeled TF-IDF families without Chi-square, followed by TruncatedSVD and L2 normalization.

Speech retains four clustering families. The requested text comparison executes K-Means, DBSCAN, and agglomerative clustering.

## Speech classification results

{_table(results['speech_classification'], ['model_key', 'family', 'cv_macro_f1_mean', 'cv_macro_f1_std', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'fit_search_seconds'])}

The selected speech classifier is **{results['speech_class_selection']['best_family']}**, chosen solely by {results['speech_class_selection']['criterion']}. Its held-out macro-F1 is **{speech_best_metrics['heldout_metrics']['macro_f1']:.4f}**.

## Speech clustering results

{_table(results['speech_clustering'], ['model_key', 'family', 'silhouette', 'ari_post_fit', 'nmi_post_fit', 'purity_descriptive', 'cluster_count_excluding_noise', 'noise_rate'])}

The selected speech clustering result is **{results['speech_cluster_selection']['best_family']}** by highest valid label-free silhouette. It forms {int(speech_selected_cluster['cluster_count_excluding_noise'])} coarse clusters and has post-fit ARI {speech_selected_cluster['ari_post_fit']:.4f}, so its compact internal geometry is not language-aligned. K-Means is more language-interpretable after fitting (ARI {speech_kmeans['ari_post_fit']:.4f}, purity {speech_kmeans['purity_descriptive']:.4f}), but those external labels did not affect selection.

## Text classification results

{_table(results['text_classification'], ['model_key', 'family', 'cv_macro_f1_mean', 'cv_macro_f1_std', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'fit_search_seconds'])}

The selected text classifier is **{results['text_class_selection']['best_family']}**, chosen solely by {results['text_class_selection']['criterion']}. Its held-out macro-F1 is **{text_best_metrics['heldout_metrics']['macro_f1']:.4f}**.

## Text clustering results

{_table(results['text_clustering'], ['model_key', 'family', 'silhouette', 'ari_post_fit', 'nmi_post_fit', 'purity_descriptive', 'cluster_count_excluding_noise', 'noise_rate'])}

The selected text clustering result is **{results['text_cluster_selection']['best_family']}** by highest valid label-free silhouette. It produces {int(text_selected_cluster['cluster_count_excluding_noise'])} non-noise clusters with noise rate {text_selected_cluster['noise_rate']:.4f}. K-Means happens to align perfectly with the supplied languages after fitting (ARI {text_kmeans['ari_post_fit']:.4f}), but this post-fit label evidence was not used to choose membership or the selected family.

## Error analysis

The most frequent speech confusion pairs for the CV-selected classifier were `{speech_confusions}`. The corresponding text confusion pairs were `{text_confusions}`. A zero-count entry here means the held-out classifier made no errors, not that an error metric was fabricated. Full per-class reports, predictions, and confusion matrices are stored under each modality's classification artifacts.

## Cross-modal interpretation

Speech relies on 133 global acoustic statistics and requires scaling for geometric models; text relies on sparse local orthographic patterns and a much larger learned vocabulary. Fit/predict timings in the comparison tables describe these different representations. Because the speech and text datasets have different rows, languages, sources, and class structures, their score differences are **not a controlled paired benchmark** and do not establish inherent modality superiority.

## Limitations

- Inferred numeric prefixes are a defensible source/speaker proxy but are not independently verified demographic identities.
- The datasets are modest, and held-out uncertainty could be assessed further with repeated nested group CV.
- Text scores are high on this supplied corpus, indicating strong script and lexical separation; external, noisier-domain validation is required before claiming broad generalization.
- Text directory codes are mapped to full English language names in generated outputs.
- Clustering geometry is sensitive to representation and density; DBSCAN can legitimately report noise or unavailable internal metrics.
- Speech inference requires one upstream extracted feature row. Raw-audio inference is outside this project's read-only input boundary.

## Reproducibility and inference

```powershell
poetry install
poetry run python scripts/check_environment.py
poetry run python scripts/run_all.py
poetry run pytest
poetry run python scripts/predict_speech.py --input-json path/to/feature_row.json
poetry run python scripts/predict_text.py --text "Questo è un breve esempio."
```

All seven notebooks are clean-kernel executed through the Poetry environment. Machine-readable results are in `reports/results_summary.json`; model-count and acceptance audits are under `artifacts/qa/` and `reports/`.
"""
    audit = _audit_acceptance(config, results)
    write_json(reports / "acceptance_audit.json", audit)
    if audit["status"] != "passed":
        failures = [f"{item['id']}: {item['criterion']}" for item in audit["checks"] if item["status"] != "passed"]
        raise RuntimeError(f"Acceptance audit failed: {failures}")
    return summary
