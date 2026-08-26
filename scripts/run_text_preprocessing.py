"""Discover, clean, split, and fit the training-only text feature pipeline."""

from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.text.features import run_text_preprocessing


def main() -> None:
    configure_logging()
    statistics = run_text_preprocessing(load_config())
    print(f"Text preprocessing passed: train shape {statistics['train_shape']}, vocabulary {statistics['vocabulary_size']}")


if __name__ == "__main__":
    main()

