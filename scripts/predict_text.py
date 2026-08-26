"""CLI for raw multilingual text inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from language_identification.config import load_config
from language_identification.text.inference import predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict language from raw Unicode text")
    parser.add_argument("--text", required=True)
    parser.add_argument("--bundle", type=Path, default=None, help="Optional saved bundle path")
    args = parser.parse_args()
    config = load_config()
    bundle = args.bundle or config.artifacts_dir / "text" / "classification" / "models" / "best_classifier.joblib"
    print(json.dumps({"prediction": predict_text(args.text, bundle)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

