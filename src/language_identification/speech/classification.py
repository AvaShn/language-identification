"""Exactly four leakage-safe speech classifier-family experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from language_identification.common.evaluation import classification_metrics, common_confusions
from language_identification.common.io import ensure_dir, write_csv, write_json
from language_identification.common.model_registry import SPEECH_CLASSIFIERS
from language_identification.common.plotting import plot_confusion
from language_identification.common.splitting import group_cv
from language_identification.config import ProjectConfig
from language_identification.speech.data import audit_speech_data
from language_identification.speech.inference import predict_speech_row

LOGGER = logging.getLogger(__name__)


def _speech_estimators(seed: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]], str]]:
    scaled = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    return {
        "logistic_regression": (
            Pipeline([*scaled, ("classifier", LogisticRegression(max_iter=3000, random_state=seed))]),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
            "Linear probabilistic baseline with standardized dense predictors.",
        ),
        "rbf_svm": (
            Pipeline([*scaled, ("classifier", SVC(kernel="rbf", cache_size=512, random_state=seed))]),
            {"classifier__C": [1.0, 10.0], "classifier__gamma": ["scale", 0.01], "classifier__class_weight": [None, "balanced"]},
            "Nonlinear maximum-margin model for a moderate dense feature table.",
        ),
        "random_forest": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", RandomForestClassifier(random_state=seed, n_jobs=1))]),
            {"classifier__n_estimators": [200], "classifier__max_depth": [None, 12], "classifier__min_samples_leaf": [1, 2], "classifier__class_weight": [None, "balanced"]},
            "Tree ensemble captures nonlinear feature interactions without scaling.",
        ),
        "knn": (
            Pipeline([*scaled, ("classifier", KNeighborsClassifier())]),
            {"classifier__n_neighbors": [3, 7, 11], "classifier__weights": ["uniform", "distance"], "classifier__p": [1, 2]},
            "Instance-based geometric model provides an algorithmically distinct contrast.",
        ),
    }


def run_speech_classification(config: ProjectConfig) -> pd.DataFrame:
    if len(SPEECH_CLASSIFIERS) != 4:
        raise AssertionError("Speech classifier registry must contain exactly four families")
    dataset, split, _ = audit_speech_data(config)
    root = ensure_dir(config.artifacts_dir / "speech" / "classification")
    for name in ("models", "predictions", "metrics", "plots"):
        ensure_dir(root / name)
    estimators = _speech_estimators(config.seed)
    if set(estimators) != set(SPEECH_CLASSIFIERS):
        raise AssertionError("Speech classifier implementation and registry differ")
    registry_artifact = {
        key: {
            **SPEECH_CLASSIFIERS[key],
            "rationale": values[2],
            "parameter_grid": values[1],
            "selection_metric": "group-aware CV macro-F1",
        }
        for key, values in estimators.items()
    }
    write_json(root / "metrics" / "registry.json", registry_artifact)

    train_x = dataset.features.iloc[split.train]

    train_y_text = dataset.labels[split.train]

    train_groups = dataset.groups[split.train]

    test_x = dataset.features.iloc[split.test]

    test_y_text = dataset.labels[split.test]


    label_encoder = LabelEncoder()

    train_y = label_encoder.fit_transform(train_y_text)

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for registry_order, (key, (pipeline, parameter_grid, _)) in enumerate(estimators.items()):
        LOGGER.info("Tuning speech classifier %s", key)
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=group_cv(config.cv_folds, config.seed),
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
        predictions_text = label_encoder.inverse_transform(predictions)
        predict_seconds = perf_counter() - start
        metrics, report, matrix, labels = classification_metrics(test_y_text, predictions_text)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        result = {
            "model_key": key,
            "family": SPEECH_CLASSIFIERS[key]["family"],
            "selection_metric": "group-aware CV macro-F1 on training partition",
            "cv_macro_f1_mean": float(search.best_score_),
            "cv_macro_f1_std": cv_std,
            "best_parameters": search.best_params_,
            "fit_search_seconds": fit_seconds,
            "heldout_predict_seconds": predict_seconds,
            "heldout_rows": int(len(test_y_text)),
            "heldout_metrics": metrics,
            "common_confusions": common_confusions(matrix, labels),
        }
        write_json(root / "metrics" / f"{key}.json", result)
        write_csv(root / "metrics" / f"{key}_classification_report.csv", report.reset_index(names="label"))
        write_csv(
            root / "predictions" / f"{key}.csv",
            pd.DataFrame({"sample_id": dataset.sample_ids[split.test], "true_language": test_y_text, "predicted_language": predictions}),
        )
        plot_confusion(matrix, labels, root / "plots" / f"{key}_confusion_matrix.png", f"Speech: {SPEECH_CLASSIFIERS[key]['family']}")
        joblib.dump(search.best_estimator_, root / "models" / f"{key}.joblib")
        fitted[key] = search.best_estimator_
        rows.append(
            {
                "model_key": key,
                "family": SPEECH_CLASSIFIERS[key]["family"],
                "cv_macro_f1_mean": float(search.best_score_),
                "cv_macro_f1_std": cv_std,
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
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "label_encoder": label_encoder,
        "model_key": best_key,
        "selection_rule": "highest group-aware training CV macro-F1; held-out test excluded from selection",
    }
    bundle_path = root / "models" / "best_classifier.joblib"
    joblib.dump(bundle, bundle_path)
    smoke_row = dataset.features.iloc[split.test[0]].to_dict()
    smoke_prediction = predict_speech_row(smoke_row, bundle_path)
    selection = {
        "best_model_key": best_key,
        "best_family": SPEECH_CLASSIFIERS[best_key]["family"],
        "criterion": "highest group-aware training CV macro-F1; exact ties use predeclared registry order",
        "test_data_used_for_selection": False,
        "cv_macro_f1": float(comparison.iloc[0]["cv_macro_f1_mean"]),
        "inference_smoke_test": {"status": "passed", "prediction": smoke_prediction, "required_feature_count": len(dataset.feature_columns)},
    }
    write_json(root / "metrics" / "selection.json", selection)
    return comparison
