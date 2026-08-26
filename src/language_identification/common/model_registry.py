"""The declared, auditable model-family registries."""

from __future__ import annotations

from typing import Any


SPEECH_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"family": "Logistic regression", "scaled": True},
    "rbf_svm": {"family": "Support vector machine", "scaled": True},
    "random_forest": {"family": "Random forest", "scaled": False},
    "knn": {"family": "k-nearest neighbours", "scaled": True},
}

TEXT_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "knn": {"family": "k-nearest neighbours"},
    "decision_tree": {"family": "Decision tree"},
    "logistic_regression": {"family": "Logistic regression"},
    "multinomial_nb": {"family": "Multinomial Naive Bayes"},
}

SPEECH_CLUSTERERS: dict[str, dict[str, Any]] = {
    "kmeans": {"family": "K-Means"},
    "agglomerative": {"family": "Agglomerative clustering"},
    "gaussian_mixture": {"family": "Gaussian mixture"},
    "dbscan": {"family": "DBSCAN"},
}

TEXT_CLUSTERERS: dict[str, dict[str, Any]] = {
    "kmeans": {"family": "K-Means"},
    "dbscan": {"family": "DBSCAN"},
    "agglomerative": {"family": "Agglomerative clustering"},
     "gaussian_mixture": {"family": "Gaussian Mixture"},
}


def assert_exact_model_counts() -> None:
    expected = {
        "speech classification": (SPEECH_CLASSIFIERS, 4),
        "speech clustering": (SPEECH_CLUSTERERS, 4),
        "text classification": (TEXT_CLASSIFIERS, 4),
        "text clustering": (TEXT_CLUSTERERS, 4),
    }
    failures = {name: len(registry) for name, (registry, count) in expected.items() if len(registry) != count}
    if failures:
        raise AssertionError(f"Model registry counts differ from the declared project design: {failures}")
