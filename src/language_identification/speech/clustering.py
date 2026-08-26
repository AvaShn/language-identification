"""Exactly four label-free speech clustering-family experiments."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language_identification.common.clustering import fit_cluster_family
from language_identification.common.evaluation import cluster_metrics, metric_number
from language_identification.common.io import ensure_dir, write_csv, write_json
from language_identification.common.model_registry import SPEECH_CLUSTERERS
from language_identification.common.plotting import plot_clusters
from language_identification.config import ProjectConfig
from language_identification.speech.data import audit_speech_data

LOGGER = logging.getLogger(__name__)


def run_speech_clustering(config: ProjectConfig) -> pd.DataFrame:
    if len(SPEECH_CLUSTERERS) != 4:
        raise AssertionError("Speech clusterer registry must contain exactly four families")
    dataset, _, _ = audit_speech_data(config)
    root = ensure_dir(config.artifacts_dir / "speech" / "clustering")
    for name in ("assignments", "metrics", "models_or_configs", "plots"):
        ensure_dir(root / name)
    section = config.section("speech")
    component_count = min(int(section["pca_components"]), len(dataset.feature_columns), len(dataset.features) - 1)
    representation = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=component_count, random_state=config.seed)),
        ]
    )
    matrix = representation.fit_transform(dataset.features)
    joblib.dump(representation, root / "models_or_configs" / "unlabeled_representation.joblib")
    write_json(
        root / "models_or_configs" / "representation.json",
        {"input_features": len(dataset.feature_columns), "pca_components": component_count, "labels_used_in_fit": False},
    )
    write_json(
        root / "metrics" / "registry.json",
        {
            key: {**value, "representation": "median imputation + StandardScaler + unsupervised PCA", "selection": "silhouette except Gaussian mixture uses BIC"}
            for key, value in SPEECH_CLUSTERERS.items()
        },
    )
    rows: list[dict[str, Any]] = []
    for key in SPEECH_CLUSTERERS:
        LOGGER.info("Fitting speech clusterer %s without labels", key)
        start = perf_counter()
        fitted = fit_cluster_family(key, matrix, config.seed, int(section["cluster_k_min"]), int(section["cluster_k_max"]))
        fit_seconds = perf_counter() - start
        # Target labels enter only after fit_cluster_family has returned assignments.
        metrics = cluster_metrics(matrix, fitted.assignments, dataset.labels)
        result = {
            "model_key": key,
            "family": SPEECH_CLUSTERERS[key]["family"],
            "labels_used_in_fit": False,
            "selected_parameters": fitted.selected_parameters,
            "selection_method": fitted.selection_method,
            "fit_seconds": fit_seconds,
            "metrics": metrics,
        }
        write_json(root / "metrics" / f"{key}.json", result)
        write_json(root / "models_or_configs" / f"{key}_candidates.json", fitted.candidates)
        joblib.dump(fitted.estimator, root / "models_or_configs" / f"{key}.joblib")
        assignments = pd.DataFrame({"sample_id": dataset.sample_ids, "true_language_post_fit": dataset.labels, "cluster": fitted.assignments})
        write_csv(root / "assignments" / f"{key}.csv", assignments)
        sizes = assignments.groupby("cluster", dropna=False).size().rename("count").reset_index()
        write_csv(root / "assignments" / f"{key}_sizes.csv", sizes)
        composition = assignments.groupby(["cluster", "true_language_post_fit"], dropna=False).size().rename("count").reset_index()
        write_csv(root / "assignments" / f"{key}_language_composition.csv", composition)
        plot_clusters(matrix[:, :2], fitted.assignments, root / "plots" / f"{key}_clusters.png", f"Speech clustering: {SPEECH_CLUSTERERS[key]['family']}")
        rows.append(
            {
                "model_key": key,
                "family": SPEECH_CLUSTERERS[key]["family"],
                "silhouette": metric_number(metrics["silhouette"]),
                "silhouette_status": "available" if metric_number(metrics["silhouette"]) is not None else metrics["silhouette"]["reason"],
                "calinski_harabasz": metric_number(metrics["calinski_harabasz"]),
                "davies_bouldin": metric_number(metrics["davies_bouldin"]),
                "ari_post_fit": metrics["ari_post_fit"],
                "nmi_post_fit": metrics["nmi_post_fit"],
                "purity_descriptive": metrics["purity_descriptive"],
                "cluster_count_excluding_noise": metrics["cluster_count_excluding_noise"],
                "noise_rate": metrics["noise_rate"],
                "fit_seconds": fit_seconds,
            }
        )
    comparison = pd.DataFrame(rows).sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)
    write_csv(root / "metrics" / "comparison.csv", comparison)
    available = comparison.dropna(subset=["silhouette"])
    best = available.iloc[0] if not available.empty else comparison.iloc[0]
    write_json(
        root / "metrics" / "selection.json",
        {
            "best_model_key": best["model_key"],
            "best_family": best["family"],
            "criterion": "highest label-free silhouette among valid results",
            "labels_used_for_selection": False,
            "interpretation_note": "ARI, NMI, and purity were computed only after fitting and were not selection inputs.",
        },
    )
    return comparison

