"""Execute the four registered speech classifier families."""

from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.reproducibility import set_global_seed
from language_identification.speech.classification import run_speech_classification


def main() -> None:
    configure_logging()
    config = load_config()
    set_global_seed(config.seed)
    print(run_speech_classification(config).to_string(index=False))


if __name__ == "__main__":
    main()

