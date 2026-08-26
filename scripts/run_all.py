"""Run the complete, ordered 16-family project workflow and strict QA."""

from __future__ import annotations

from datetime import datetime, timezone

from language_identification.common.io import write_json
from language_identification.common.model_registry import assert_exact_model_counts
from language_identification.common.preflight import run_preflight
from language_identification.common.report import generate_report
from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.reproducibility import set_global_seed
from language_identification.speech.classification import run_speech_classification
from language_identification.speech.clustering import run_speech_clustering
from language_identification.speech.data import audit_speech_data
from language_identification.text.classification import run_text_classification
from language_identification.text.clustering import run_text_clustering
from language_identification.text.features import run_text_preprocessing

from execute_notebooks import execute_required_notebooks
from run_quality import run_tests


def main() -> None:
    configure_logging()
    config = load_config()
    set_global_seed(config.seed)
    assert_exact_model_counts()
    run_preflight(config)
    audit_speech_data(config)
    run_speech_classification(config)
    run_speech_clustering(config)
    run_text_preprocessing(config)
    run_text_classification(config)
    run_text_clustering(config)
    execute_required_notebooks()
    run_tests()
    run_preflight(config)
    generate_report(config)
    write_json(
        config.artifacts_dir / "qa" / "full_run.json",
        {
            "status": "passed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "required_model_family_results": 16,
            "notebooks_executed": 7,
        },
    )
    print("Complete workflow passed: 16 model families, 7 notebooks, tests, report, and acceptance audit")


if __name__ == "__main__":
    main()

