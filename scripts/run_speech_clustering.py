"""Execute the four registered speech clustering families without labels in fit."""

from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.reproducibility import set_global_seed
from language_identification.speech.clustering import run_speech_clustering


def main() -> None:
    configure_logging()
    config = load_config()
    set_global_seed(config.seed)
    print(run_speech_clustering(config).to_string(index=False))


if __name__ == "__main__":
    main()

