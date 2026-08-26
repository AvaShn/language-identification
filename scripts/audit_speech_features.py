"""Audit the supplied read-only speech feature table and create a group split."""

from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.speech.data import audit_speech_data


def main() -> None:
    configure_logging()
    _, _, audit = audit_speech_data(load_config())
    print(f"Speech audit passed: {audit['rows']} rows, {audit['numeric_predictor_count']} predictors")


if __name__ == "__main__":
    main()

