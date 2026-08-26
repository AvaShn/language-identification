"""Execute the four registered text classifier families."""

from language_identification.config import load_config
from language_identification.logging_utils import configure_logging
from language_identification.reproducibility import set_global_seed
from language_identification.text.classification import run_text_classification


def main() -> None:
    configure_logging()
    config = load_config()
    set_global_seed(config.seed)
    print(run_text_classification(config).to_string(index=False))


if __name__ == "__main__":
    main()

