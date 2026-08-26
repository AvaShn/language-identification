"""Label-free parameter selection for the four clustering families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterFit:
    estimator: Any
    assignments: np.ndarray
    selected_parameters: dict[str, Any]
    selection_method: str
    candidates: list[dict[str, Any]]


def _silhouette_or_none(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels != -1
    unique = np.unique(labels[mask])
    if mask.sum() < 3 or len(unique) < 2 or len(unique) >= mask.sum():
        return None
    return float(silhouette_score(matrix[mask], labels[mask]))


def fit_cluster_family(
    family: str,
    matrix: np.ndarray,
    seed: int,
    k_min: int,
    k_max: int,
) -> ClusterFit:
    """Fit and choose one clustering family without accepting target labels."""
    if matrix.ndim != 2 or len(matrix) < 3:
        raise ValueError("Clustering requires a 2D matrix with at least three samples")
    k_values = range(k_min, min(k_max, len(matrix) - 1) + 1)
    candidates: list[dict[str, Any]] = []

    if family == "kmeans":
        best: tuple[float, Any, np.ndarray, int] | None = None
        for k in k_values:
            estimator = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = estimator.fit_predict(matrix)
            score = _silhouette_or_none(matrix, labels)
            candidates.append({"n_clusters": k, "silhouette": score, "inertia": float(estimator.inertia_)})
            if score is not None and (best is None or score > best[0]):
                best = (score, estimator, labels, k)
        if best is None:
            raise RuntimeError("K-Means produced no valid silhouette candidate")
        return ClusterFit(best[1], best[2], {"n_clusters": best[3], "n_init": 20}, "maximum silhouette", candidates)

    if family == "agglomerative":
        best_agg: tuple[float, Any, np.ndarray, int, str] | None = None
        for linkage in ("ward", "average"):
            for k in k_values:
                estimator = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = estimator.fit_predict(matrix)
                score = _silhouette_or_none(matrix, labels)
                candidates.append({"n_clusters": k, "linkage": linkage, "silhouette": score})
                if score is not None and (best_agg is None or score > best_agg[0]):
                    best_agg = (score, estimator, labels, k, linkage)
        if best_agg is None:
            raise RuntimeError("Agglomerative clustering produced no valid silhouette candidate")
        return ClusterFit(best_agg[1], best_agg[2], {"n_clusters": best_agg[3], "linkage": best_agg[4]}, "maximum silhouette", candidates)

    if family == "gaussian_mixture":
        best_gmm: tuple[float, Any, np.ndarray, int] | None = None
        for k in k_values:
            estimator = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=seed, reg_covar=1e-6)
            estimator.fit(matrix)
            labels = estimator.predict(matrix)
            bic = float(estimator.bic(matrix))
            score = _silhouette_or_none(matrix, labels)
            candidates.append({"n_components": k, "covariance_type": "diag", "bic": bic, "silhouette": score})
            if best_gmm is None or bic < best_gmm[0]:
                best_gmm = (bic, estimator, labels, k)
        if best_gmm is None:
            raise RuntimeError("Gaussian mixture produced no candidates")
        return ClusterFit(best_gmm[1], best_gmm[2], {"n_components": best_gmm[3], "covariance_type": "diag"}, "minimum BIC", candidates)

    if family == "dbscan":
        best_db: tuple[tuple[float, float, int], Any, np.ndarray, float, int] | None = None
        for min_samples in (5, 10):
            neighbours = NearestNeighbors(n_neighbors=min(min_samples, len(matrix))).fit(matrix)
            distances, _ = neighbours.kneighbors(matrix)
            kth = distances[:, -1]
            for quantile in (0.70, 0.80, 0.90, 0.95):
                eps = float(np.quantile(kth, quantile))
                if eps <= 0:
                    continue
                estimator = DBSCAN(eps=eps, min_samples=min_samples)
                labels = estimator.fit_predict(matrix)
                score = _silhouette_or_none(matrix, labels)
                noise_rate = float(np.mean(labels == -1))
                cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
                candidates.append({"eps": eps, "min_samples": min_samples, "quantile": quantile, "silhouette": score, "noise_rate": noise_rate, "cluster_count": cluster_count})
                rank = (score if score is not None else -2.0, -noise_rate, cluster_count)
                if best_db is None or rank > best_db[0]:
                    best_db = (rank, estimator, labels, eps, min_samples)
        if best_db is None:
            raise RuntimeError("DBSCAN produced no candidates")
        return ClusterFit(best_db[1], best_db[2], {"eps": best_db[3], "min_samples": best_db[4]}, "maximum valid silhouette, then lower noise", candidates)

    raise KeyError(f"Unknown clustering family: {family}")

