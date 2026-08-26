"""Non-interactive plotting helpers for generated experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from language_identification.common.io import ensure_dir


def plot_confusion(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    ensure_dir(path.parent)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted language")
    axis.set_ylabel("True language")
    axis.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_clusters(points: np.ndarray, assignments: np.ndarray, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    if points.shape[1] < 2:
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(points[:, 0], points[:, 1], c=assignments, cmap="tab10", s=18, alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Unsupervised component 1")
    axis.set_ylabel("Unsupervised component 2")
    figure.colorbar(scatter, ax=axis, label="Cluster (-1 = noise)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

