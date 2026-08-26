"""CLI for one already-extracted speech feature JSON object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from language_identification.config import load_config
from language_identification.speech.inference import predict_speech_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict language from one extracted speech feature row")
    parser.add_argument("--input-json", required=True, type=Path, help="JSON file containing required numeric feature fields")
    parser.add_argument("--bundle", type=Path, default=None, help="Optional saved bundle path")
    args = parser.parse_args()
    config = load_config()
    bundle = args.bundle or config.artifacts_dir / "speech" / "classification" / "models" / "best_classifier.joblib"
    row = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError("Speech input JSON must contain one object")
    print(json.dumps({"prediction": predict_speech_row(row, bundle)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

