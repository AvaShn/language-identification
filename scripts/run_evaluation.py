"""Build cross-modal summaries, final report, and strict acceptance audit."""

from language_identification.common.report import generate_report
from language_identification.config import load_config
from language_identification.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    summary = generate_report(load_config())
    print(f"Report generated with model counts: {summary['model_counts']}")


if __name__ == "__main__":
    main()

