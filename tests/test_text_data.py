from language_identification.config import load_config
from language_identification.text.cleaning import clean_text
from language_identification.text.data import load_text_data


def test_cleaning_preserves_multilingual_information() -> None:
    value = "  Àrvore\r\n日本語! 한국어  "
    cleaned = clean_text(value)
    assert cleaned == "Àrvore 日本語! 한국어"


def test_actual_text_schema_is_independent_directory_corpus() -> None:
    dataset = load_text_data(load_config())
    assert len(dataset.frame) == 608
    assert set(dataset.labels) == {"german", "spanish", "italian", "japanese", "korean", "portuguese"}
    assert dataset.frame["clean_text"].str.len().min() > 0
    assert not dataset.frame["clean_text"].duplicated().any()
