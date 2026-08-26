"""Exactly four leakage-safe multilingual text classifier-family experiments."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from language_identification.common.evaluation import classification_metrics, common_confusions
from language_identification.common.io import ensure_dir, write_csv, write_json
from language_identification.common.model_registry import TEXT_CLASSIFIERS
from language_identification.common.plotting import plot_confusion
from language_identification.common.splitting import group_cv
from language_identification.config import ProjectConfig
from language_identification.text.cleaning import MultilingualTextCleaner
from language_identification.text.data import audit_text_data
from language_identification.text.features import make_feature_selector, make_feature_union, run_text_preprocessing
from language_identification.text.inference import predict_text

LOGGER = logging.getLogger(__name__)


def _text_estimators(config: ProjectConfig) -> dict[str, tuple[Pipeline, dict[str, list[Any]], str]]:
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [
                ("cleaner", MultilingualTextCleaner()),
                ("features", make_feature_union(config)),
                ("select", make_feature_selector(config)),
                ("classifier", classifier),
            ]
        )

    return {
        "knn": (
            pipeline(KNeighborsClassifier(weights="uniform", metric="cosine", algorithm="brute")),
            {"classifier__n_neighbors": [1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250]},
            "Instance-based cosine-distance model using the exact candidate grid from the reference project.",
        ),
        "decision_tree": (
            pipeline(DecisionTreeClassifier(random_state=config.seed)),
            {"classifier__max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]},
            "Nonlinear decision-tree baseline with depth selected by group-aware validation Macro F1.",
        ),
        "logistic_regression": (
            pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.seed)),
            {"classifier__C": [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]},
            "Balanced regularized linear classifier for the selected character and word features.",
        ),
        "multinomial_nb": (
            pipeline(MultinomialNB()),
            {"classifier__alpha": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]},
            "Generative non-negative feature baseline using the reference alpha grid.",
        ),
    }


def run_text_classification(config: ProjectConfig) -> pd.DataFrame:
    if len(TEXT_CLASSIFIERS) != 4:
        raise AssertionError("Text classifier registry must contain exactly four families")
    run_text_preprocessing(config)
    dataset, split, _ = audit_text_data(config)
    root = ensure_dir(config.artifacts_dir / "text" / "classification")
    for name in ("models", "predictions", "metrics", "plots"):
        ensure_dir(root / name)
    estimators = _text_estimators(config)
    if set(estimators) != set(TEXT_CLASSIFIERS):
        raise AssertionError("Text classifier implementation and registry differ")
    write_json(
        root / "metrics" / "registry.json",
        {
            key: {
                **TEXT_CLASSIFIERS[key],
                "rationale": values[2],
                "parameter_grid": values[1],
                "representation": "cleaner + character/word TF-IDF FeatureUnion + Chi-square top-50 fitted inside each CV fold",
                "selection_metric": "group-aware CV macro-F1",
            }
            for key, values in estimators.items()
        },
    )

    train_x = dataset.raw_texts[split.train]
    train_y = dataset.labels[split.train]
    train_groups = dataset.groups[split.train]
    test_x = dataset.raw_texts[split.test]
    test_y = dataset.labels[split.test]
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    text_cv_folds = int(config.section("text").get("cv_folds", config.cv_folds))
    for registry_order, (key, (pipeline, parameter_grid, _)) in enumerate(estimators.items()):
        LOGGER.info("Tuning text classifier %s", key)
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=group_cv(text_cv_folds, config.seed),
            n_jobs=config.n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        start = perf_counter()
        search.fit(train_x, train_y, groups=train_groups)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predictions = search.best_estimator_.predict(test_x)
        predict_seconds = perf_counter() - start
        metrics, report, matrix, labels = classification_metrics(test_y, predictions)
        train_predictions = search.best_estimator_.predict(train_x)
        train_metrics, _, _, _ = classification_metrics(train_y, train_predictions)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        result = {
            "model_key": key,
            "family": TEXT_CLASSIFIERS[key]["family"],
            "selection_metric": "group-aware CV macro-F1 on training partition",
            "cv_macro_f1_mean": float(search.best_score_),
            "cv_macro_f1_std": cv_std,
            "best_parameters": search.best_params_,
            "fit_search_seconds": fit_seconds,
            "heldout_predict_seconds": predict_seconds,
            "heldout_rows": int(len(test_y)),
            "training_metrics": train_metrics,
            "heldout_metrics": metrics,
            "common_confusions": common_confusions(matrix, labels),
        }
        write_json(root / "metrics" / f"{key}.json", result)
        cv_results = pd.DataFrame(search.cv_results_)
        cv_columns = [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "mean_score_time",
            "params",
        ]
        write_csv(root / "metrics" / f"{key}_cv_results.csv", cv_results.loc[:, cv_columns].sort_values("rank_test_score"))
        write_csv(root / "metrics" / f"{key}_classification_report.csv", report.reset_index(names="label"))
        write_csv(
            root / "predictions" / f"{key}.csv",
            pd.DataFrame({"sample_id": dataset.sample_ids[split.test], "true_language": test_y, "predicted_language": predictions}),
        )
        plot_confusion(matrix, labels, root / "plots" / f"{key}_confusion_matrix.png", f"Text: {TEXT_CLASSIFIERS[key]['family']}")
        joblib.dump(search.best_estimator_, root / "models" / f"{key}.joblib")
        fitted[key] = search.best_estimator_
        rows.append(
            {
                "model_key": key,
                "family": TEXT_CLASSIFIERS[key]["family"],
                "cv_macro_f1_mean": float(search.best_score_),
                "cv_macro_f1_std": cv_std,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_precision": train_metrics["macro_precision"],
                "train_macro_recall": train_metrics["macro_recall"],
                "train_macro_f1": train_metrics["macro_f1"],
                **{f"test_{name}": value for name, value in metrics.items()},
                "fit_search_seconds": fit_seconds,
                "heldout_predict_seconds": predict_seconds,
                "registry_order_tiebreak": registry_order,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_mean", "registry_order_tiebreak"], ascending=[False, True]
    ).reset_index(drop=True)
    write_csv(root / "metrics" / "comparison.csv", comparison)
    best_key = str(comparison.iloc[0]["model_key"])
    bundle = {
        "pipeline": fitted[best_key],
        "model_key": best_key,
        "selection_rule": "highest group-aware training CV macro-F1; held-out test excluded from selection",
        "input": "raw Unicode text",
    }
    bundle_path = root / "models" / "best_classifier.joblib"
    joblib.dump(bundle, bundle_path)
    smoke_prediction = predict_text(str(dataset.raw_texts[split.test[0]]), bundle_path)
    write_json(
        root / "metrics" / "selection.json",
        {
            "best_model_key": best_key,
            "best_family": TEXT_CLASSIFIERS[best_key]["family"],
            "criterion": "highest group-aware training CV macro-F1; exact ties use predeclared registry order",
            "test_data_used_for_selection": False,
            "cv_macro_f1": float(comparison.iloc[0]["cv_macro_f1_mean"]),
            "inference_smoke_test": {"status": "passed", "prediction": smoke_prediction, "input": "raw text"},
        },
    )
    return comparison
