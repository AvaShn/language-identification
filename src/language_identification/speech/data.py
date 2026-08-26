"""Read-only speech feature ingestion, alignment, grouping, and audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from language_identification.common.io import ensure_dir, write_csv, write_json
from language_identification.common.splitting import SplitIndices, stratified_group_holdout
from language_identification.config import ProjectConfig


@dataclass
class SpeechDataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    feature_columns: list[str]
    target_column: str
    group_rule: str


def speech_paths(config: ProjectConfig) -> tuple[Path, Path | None]:
    section = config.section("speech")
    source = config.resolve_path(section["feature_source"])
    feature_path = source / section["features_file"]
    metadata_path = source / section["metadata_file"] if section.get("metadata_file") else None
    if not source.is_dir():
        raise FileNotFoundError(
            f"Speech feature source directory does not exist: {source}. "
            "Update configs/speech.yaml; raw-audio extraction will not be used as a fallback."
        )
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Required supplied speech feature table is missing: {feature_path}. "
            "The project intentionally will not rebuild speech features."
        )
    return feature_path, metadata_path if metadata_path and metadata_path.is_file() else None


def _align_metadata(features: pd.DataFrame, metadata: pd.DataFrame, identifier: str) -> pd.DataFrame:
    if identifier not in features or identifier not in metadata:
        raise ValueError(f"Both features and metadata must contain stable identifier column '{identifier}'")
    if features[identifier].duplicated().any() or metadata[identifier].duplicated().any():
        raise ValueError(f"Duplicate '{identifier}' values prevent safe speech metadata alignment")
    missing = set(features[identifier]) - set(metadata[identifier])
    extra = set(metadata[identifier]) - set(features[identifier])
    if missing or extra:
        raise ValueError(
            f"Speech metadata alignment failed: {len(missing)} feature IDs missing from metadata and "
            f"{len(extra)} extra metadata IDs"
        )
    aligned = metadata.set_index(identifier).loc[features[identifier]].reset_index()
    result = features.copy()
    for column in aligned.columns:
        if column == identifier:
            continue
        if column in result:
            unequal = result[column].astype(str).to_numpy() != aligned[column].astype(str).to_numpy()
            if unequal.any():
                raise ValueError(f"Speech features and metadata disagree in column '{column}' after ID alignment")
        else:
            result[column] = aligned[column].to_numpy()
    return result


def _infer_groups(paths: pd.Series, pattern: str) -> tuple[np.ndarray, str]:
    regex = re.compile(pattern)
    groups: list[str] = []
    matched = 0
    for value in paths.astype(str):
        name = Path(value).name
        match = regex.match(name)
        if match:
            groups.append(match.group(1))
            matched += 1
        else:
            groups.append(f"unique:{value.replace(chr(92), '/')}")
    rule = f"regex {pattern!r} on filepath basename; {matched}/{len(paths)} rows matched, unmatched paths treated as unique"
    return np.asarray(groups, dtype=object), rule


def load_speech_data(config: ProjectConfig) -> SpeechDataset:
    section = config.section("speech")
    feature_path, metadata_path = speech_paths(config)
    frame = pd.read_csv(feature_path)
    if frame.empty:
        raise ValueError(f"Speech feature table contains no rows: {feature_path}")
    identifier = str(section["identifier_column"])
    target = str(section["target_column"])
    if metadata_path:
        frame = _align_metadata(frame, pd.read_csv(metadata_path), identifier)
    required = [identifier, target]
    missing_required = [column for column in required if column not in frame]
    if missing_required:
        raise ValueError(f"Speech feature table is missing required columns: {missing_required}")
    if frame[identifier].isna().any() or frame[target].isna().any():
        raise ValueError("Speech identifier and target columns may not contain null values")
    excluded = {identifier, target, *section.get("metadata_columns", [])}
    candidate = frame.drop(columns=[column for column in excluded if column in frame])
    non_numeric = candidate.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric speech predictor candidates require an explicit metadata exclusion: {non_numeric}"
        )
    feature_columns = candidate.columns.tolist()
    if not feature_columns:
        raise ValueError("No numerical speech predictors remain after metadata/target exclusion")
    matrix = candidate.replace([np.inf, -np.inf], np.nan)
    groups, group_rule = _infer_groups(frame[identifier], str(section["group_regex"]))
    return SpeechDataset(
        frame=frame,
        features=matrix,
        labels=frame[target].astype(str).to_numpy(),
        groups=groups,
        sample_ids=frame[identifier].astype(str).to_numpy(),
        feature_columns=feature_columns,
        target_column=target,
        group_rule=group_rule,
    )


def audit_speech_data(config: ProjectConfig) -> tuple[SpeechDataset, SplitIndices, dict[str, Any]]:
    dataset = load_speech_data(config)
    output = ensure_dir(config.artifacts_dir / "speech" / "audit")
    split = stratified_group_holdout(dataset.labels, dataset.groups, config.test_size, config.seed)
    numeric = dataset.features
    constant = [column for column in numeric if numeric[column].nunique(dropna=False) <= 1]
    near_constant = [
        column for column in numeric
        if column not in constant and numeric[column].value_counts(dropna=False, normalize=True).iloc[0] >= 0.99
    ]
    suspicious = [column for column in numeric if any(token in column.lower() for token in ("lang", "label", "target", "class"))]
    audit: dict[str, Any] = {
        "source_role": "supplied read-only extracted speech features",
        "rows": int(len(dataset.frame)),
        "columns_total": int(dataset.frame.shape[1]),
        "target_column": dataset.target_column,
        "identifier_columns": [config.section("speech")["identifier_column"]],
        "metadata_columns": config.section("speech").get("metadata_columns", []),
        "numeric_predictor_count": len(dataset.feature_columns),
        "numeric_predictors": dataset.feature_columns,
        "class_distribution": dataset.frame[dataset.target_column].value_counts().sort_index().to_dict(),
        "duplicate_rows": int(dataset.frame.duplicated().sum()),
        "duplicate_identifiers": int(pd.Series(dataset.sample_ids).duplicated().sum()),
        "missing_values_by_column": {key: int(value) for key, value in dataset.frame.isna().sum().items() if value},
        "infinite_numeric_values": int(np.isinf(dataset.features.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "constant_features": constant,
        "near_constant_features_99pct": near_constant,
        "suspicious_label_like_numeric_columns": suspicious,
        "metadata_alignment": "aligned by filepath and equality-checked" if speech_paths(config)[1] else "metadata.csv unavailable; feature table fields used",
        "group_rule": dataset.group_rule,
        "group_count": int(len(np.unique(dataset.groups))),
        "repeated_group_count": int(sum(value > 1 for value in pd.Series(dataset.groups).value_counts())),
        "split_strategy": split.strategy,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "group_overlap_count": int(len(set(dataset.groups[split.train]) & set(dataset.groups[split.test]))),
        "leakage_controls": [
            "target, filepath, and gender excluded from predictors",
            "metadata aligned by stable filepath rather than row order",
            "speaker/source groups kept disjoint in holdout and CV",
            "preprocessing fitted inside supervised training pipelines",
        ],
    }
    write_json(output / "speech_audit.json", audit)
    write_csv(output / "class_distribution.csv", dataset.frame[dataset.target_column].value_counts().sort_index().rename_axis("language").reset_index(name="count"))
    statistics = dataset.features.describe(include="all").transpose().reset_index(names="feature")
    write_csv(output / "feature_statistics.csv", statistics)
    split_frame = pd.DataFrame({
        "sample_id": dataset.sample_ids,
        "language": dataset.labels,
        "group": dataset.groups,
        "partition": np.where(np.isin(np.arange(len(dataset.labels)), split.test), "test", "train"),
    })
    write_csv(config.artifacts_dir / "speech" / "splits" / "holdout_split.csv", split_frame)
    return dataset, split, audit

