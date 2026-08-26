"""Classification and post-fit clustering evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, list[str]]:
    labels = sorted({str(value) for value in y_true} | {str(value) for value in y_pred})
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).transpose()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, matrix, labels


def common_confusions(matrix: np.ndarray, labels: list[str], limit: int = 5) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row, true_label in enumerate(labels):
        for column, predicted_label in enumerate(labels):
            if row != column and int(matrix[row, column]) > 0:
                pairs.append({"true": true_label, "predicted": predicted_label, "count": int(matrix[row, column])})
    return sorted(pairs, key=lambda item: (-item["count"], item["true"], item["predicted"]))[:limit]


def not_applicable(reason: str) -> dict[str, str]:
    return {"status": "not_applicable", "reason": reason}


def _internal_cluster_metrics(matrix: np.ndarray, assignments: np.ndarray) -> dict[str, Any]:
    mask = assignments != -1
    used = assignments[mask]
    unique = np.unique(used)
    if mask.sum() < 3 or len(unique) < 2 or len(unique) >= mask.sum():
        reason = "requires at least two non-noise clusters and fewer clusters than evaluated samples"
        unavailable = not_applicable(reason)
        return {"silhouette": unavailable, "calinski_harabasz": unavailable, "davies_bouldin": unavailable}
    subset = matrix[mask]
    return {
        "silhouette": float(silhouette_score(subset, used)),
        "calinski_harabasz": float(calinski_harabasz_score(subset, used)),
        "davies_bouldin": float(davies_bouldin_score(subset, used)),
    }


def cluster_purity(y_true: np.ndarray, assignments: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(assignments):
        values = y_true[assignments == cluster]
        if len(values):
            total += Counter(values).most_common(1)[0][1]
    return float(total / len(y_true))


def cluster_metrics(matrix: np.ndarray, assignments: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """Evaluate assignments; labels enter only in this post-fit function."""
    metrics = _internal_cluster_metrics(matrix, assignments)
    metrics.update(
        {
            "ari_post_fit": float(adjusted_rand_score(y_true, assignments)),
            "nmi_post_fit": float(normalized_mutual_info_score(y_true, assignments)),
            "purity_descriptive": cluster_purity(y_true, assignments),
            "cluster_count_excluding_noise": int(len(set(assignments)) - (1 if -1 in assignments else 0)),
            "noise_count": int(np.sum(assignments == -1)),
            "noise_rate": float(np.mean(assignments == -1)),
        }
    )
    return metrics


def metric_number(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        return float(value)
    return None

