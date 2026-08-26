"""Inference from one already-extracted speech feature row."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


def load_speech_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "feature_columns"}.issubset(bundle):
        raise ValueError(f"Invalid speech classifier bundle: {path}")
    return bundle


def predict_speech_row(row: Mapping[str, Any], bundle_path: Path) -> str:
    bundle = load_speech_bundle(bundle_path)
    columns = list(bundle["feature_columns"])
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Speech feature row is missing {len(missing)} required fields; first missing: {missing[:5]}")
    frame = pd.DataFrame([{column: row[column] for column in columns}], columns=columns)
    prediction = bundle["pipeline"].predict(frame)[0]

    if "label_encoder" in bundle:
        prediction = bundle["label_encoder"].inverse_transform([prediction])[0]

    return str(prediction)

