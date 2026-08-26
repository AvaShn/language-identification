"""Leakage-aware held-out and cross-validation split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray
    strategy: str


def stratified_group_holdout(
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> SplitIndices:
    """Create one deterministic stratified, group-disjoint held-out fold."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have identical lengths")
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two language classes are required")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n_splits = max(2, int(round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(labels)), labels, groups))
    overlap = set(groups[train]) & set(groups[test])
    if overlap:
        raise RuntimeError(f"Group leakage detected in held-out split: {sorted(overlap)[:3]}")
    missing_train = set(np.unique(labels)) - set(np.unique(labels[train]))
    missing_test = set(np.unique(labels)) - set(np.unique(labels[test]))
    if missing_train or missing_test:
        raise RuntimeError(
            f"Stratified group split lost classes; train_missing={missing_train}, test_missing={missing_test}"
        )
    return SplitIndices(train=train, test=test, strategy=f"StratifiedGroupKFold first fold ({n_splits} folds)")


def group_cv(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

