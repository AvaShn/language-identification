from language_identification.config import load_config
from language_identification.text.data import load_text_data
from language_identification.text.features import make_feature_selector, make_text_feature_pipeline


def test_character_tfidf_fits_and_reloads_shape() -> None:
    config = load_config()
    dataset = load_text_data(config)
    pipeline = make_text_feature_pipeline(config)
    matrix = pipeline.fit_transform(dataset.raw_texts[:60])
    transformed = pipeline.transform(dataset.raw_texts[60:63])
    assert matrix.shape[0] == 60
    assert transformed.shape == (3, matrix.shape[1])
    union = pipeline.named_steps["features"]
    vectorizers = dict(union.transformer_list)
    assert vectorizers["char"].analyzer == "char_wb"
    assert vectorizers["char"].ngram_range == (2, 4)
    assert vectorizers["word"].analyzer == "word"
    assert vectorizers["word"].ngram_range == (1, 2)
    selector = make_feature_selector(config)
    assert selector.k == 50
