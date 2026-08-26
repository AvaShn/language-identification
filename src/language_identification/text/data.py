"""Independent raw text discovery, schema resolution, grouping, and audit."""

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
from language_identification.text.cleaning import clean_text


@dataclass
class TextDataset:
    frame: pd.DataFrame
    raw_texts: np.ndarray
    clean_texts: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    group_rule: str
    discovery: dict[str, Any]


def text_source(config: ProjectConfig) -> Path:
    source = config.resolve_path(config.section("text")["raw_source"])
    if not source.is_dir():
        raise FileNotFoundError(
            f"Independent text source directory does not exist: {source}. Update configs/text.yaml; "
            "the speech source will not be used as a substitute."
        )
    return source


def load_text_data(config: ProjectConfig) -> TextDataset:
    section = config.section("text")
    source = text_source(config)
    language_names = {str(key): str(value) for key, value in section.get("language_names", {}).items()}
    if language_names:
        files = [
            path
            for folder_code in language_names
            for path in sorted((source / folder_code).rglob("*.txt"))
            if path.is_file()
        ]
    else:
        files = sorted(path for path in source.glob(section["glob"]) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No text files matching {section['glob']!r} were found under {source}")
    regex = re.compile(str(section["group_regex"]))
    rows: list[dict[str, Any]] = []
    decode_errors: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(source)
        if len(relative.parts) < 2:
            raise ValueError(f"Cannot infer a directory language label for text file: {relative}")
        source_language = relative.parts[0]
        language = language_names.get(source_language, source_language)
        try:
            raw = path.read_text(encoding=str(section["encoding"]))
        except UnicodeDecodeError as exc:
            decode_errors.append({"sample_id": relative.as_posix(), "error": str(exc)})
            continue
        normalized = clean_text(raw)
        match = regex.match(path.stem)
        group = f"{language}:{match.group(1)}" if match else f"unique:{language}:{relative.as_posix()}"
        rows.append(
            {
                "sample_id": relative.as_posix(),
                "language": language,
                "gender": relative.parts[1] if len(relative.parts) > 2 else None,
                "raw_text": raw,
                "clean_text": normalized,
                "group": group,
                "normalization_changed": raw != normalized,
            }
        )
    if decode_errors:
        examples = decode_errors[:3]
        raise ValueError(f"{len(decode_errors)} text files failed {section['encoding']} decoding; examples: {examples}")
    raw_frame = pd.DataFrame(rows)
    empty_count = int((raw_frame["clean_text"].str.len() == 0).sum())
    nonempty = raw_frame.loc[raw_frame["clean_text"].str.len() > 0].copy()
    duplicate_mask = nonempty.duplicated(subset=["clean_text"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    frame = nonempty.loc[~duplicate_mask].reset_index(drop=True)
    if frame["language"].nunique() < 2:
        raise ValueError("Text dataset must contain at least two directory language labels")
    group_rule = (
        f"language-prefixed regex {section['group_regex']!r} on filename; "
        "unmatched files receive unique path groups"
    )
    discovery = {
        "source_format": "UTF-8 directory-per-language .txt files",
        "source_file_count": len(files),
        "decoded_file_count": len(raw_frame),
        "text_field": "complete file contents",
        "label_field": "first relative directory",
        "sample_id_field": "relative path",
        "optional_metadata": ["second relative directory (gender)"],
        "encoding": section["encoding"],
        "empty_removed": empty_count,
        "exact_duplicate_rows_removed": duplicate_count,
        "clean_rows": len(frame),
    }
    return TextDataset(
        frame=frame,
        raw_texts=frame["raw_text"].astype(str).to_numpy(),
        clean_texts=frame["clean_text"].astype(str).to_numpy(),
        labels=frame["language"].astype(str).to_numpy(),
        groups=frame["group"].astype(str).to_numpy(),
        sample_ids=frame["sample_id"].astype(str).to_numpy(),
        group_rule=group_rule,
        discovery=discovery,
    )


def audit_text_data(config: ProjectConfig) -> tuple[TextDataset, SplitIndices, dict[str, Any]]:
    dataset = load_text_data(config)
    text_test_size = float(config.section("text").get("test_size", config.test_size))
    split = stratified_group_holdout(dataset.labels, dataset.groups, text_test_size, config.seed)
    output = ensure_dir(config.artifacts_dir / "text" / "audit")
    lengths = dataset.frame["clean_text"].str.len()
    audit = {
        **dataset.discovery,
        "dataset_independence": "independent directory corpus; no speech row/sample pairing assumed",
        "class_distribution": dataset.frame["language"].value_counts().sort_index().to_dict(),
        "gender_distribution": dataset.frame.groupby(["language", "gender"], dropna=False).size().to_dict(),
        "normalization_changed_rows": int(dataset.frame["normalization_changed"].sum()),
        "replacement_character_count": int(dataset.frame["clean_text"].str.count("\ufffd").sum()),
        "text_length_characters": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
        "duplicate_policy": "after whitespace normalization, keep lexicographically first path for exact duplicate text",
        "cleaning": [
            "collapse consecutive whitespace (including line breaks and tabs) to one space",
            "trim surrounding whitespace",
            "preserve case, punctuation, digits, scripts, and diacritics",
            "no stemming, lemmatization, transliteration, or English-only tokenization",
        ],
        "group_rule": dataset.group_rule,
        "group_count": int(len(np.unique(dataset.groups))),
        "repeated_group_count": int(sum(value > 1 for value in pd.Series(dataset.groups).value_counts())),
        "split_strategy": split.strategy,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "group_overlap_count": int(len(set(dataset.groups[split.train]) & set(dataset.groups[split.test]))),
        "normalized_text_overlap_count": int(len(set(dataset.clean_texts[split.train]) & set(dataset.clean_texts[split.test]))),
    }
    # JSON does not support tuple keys from the grouped distribution.
    audit["gender_distribution"] = {
        f"{language}/{gender}": int(count)
        for (language, gender), count in dataset.frame.groupby(["language", "gender"], dropna=False).size().items()
    }
    write_json(output / "text_audit.json", audit)
    write_csv(output / "class_distribution.csv", dataset.frame["language"].value_counts().sort_index().rename_axis("language").reset_index(name="count"))
    cleaning_counts = pd.DataFrame(
        [
            {"event": "source_files", "count": dataset.discovery["source_file_count"]},
            {"event": "empty_removed", "count": dataset.discovery["empty_removed"]},
            {"event": "exact_duplicates_removed", "count": dataset.discovery["exact_duplicate_rows_removed"]},
            {"event": "normalization_changed", "count": audit["normalization_changed_rows"]},
            {"event": "final_rows", "count": len(dataset.frame)},
        ]
    )
    write_csv(config.artifacts_dir / "text" / "preprocessing" / "cleaning_counts.csv", cleaning_counts)
    split_frame = pd.DataFrame({
        "sample_id": dataset.sample_ids,
        "language": dataset.labels,
        "group": dataset.groups,
        "partition": np.where(np.isin(np.arange(len(dataset.labels)), split.test), "test", "train"),
    })
    write_csv(config.artifacts_dir / "text" / "splits" / "holdout_split.csv", split_frame)
    return dataset, split, audit
