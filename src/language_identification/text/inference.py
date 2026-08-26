"""Raw-text language inference from the saved full pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_text_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"Invalid text classifier bundle: {path}")
    return bundle


def predict_text(text: str, bundle_path: Path) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Raw text inference requires a non-empty string")
    bundle = load_text_bundle(bundle_path)
    return str(bundle["pipeline"].predict([text])[0])

