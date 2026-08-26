"""Run the required environment and immutable-input preflight."""

from __future__ import annotations

import json

from language_identification.common.preflight import run_preflight
from language_identification.config import load_config
from language_identification.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    report = run_preflight(load_config())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

