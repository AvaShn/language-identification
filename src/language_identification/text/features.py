"""Training-only combined character/word TF-IDF feature artifacts."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from language_identification.common.io import ensure_dir, write_json
from language_identification.config import ProjectConfig
from language_identification.text.cleaning import MultilingualTextCleaner
from language_identification.text.data import audit_text_data


def _make_vectorizer(settings: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=str(settings["analyzer"]),
        ngram_range=tuple(int(value) for value in settings["ngram_range"]),
        min_df=int(settings["min_df"]),
        max_features=int(settings["max_features"]),
        sublinear_tf=bool(settings["sublinear_tf"]),
        lowercase=bool(settings["lowercase"]),
        dtype=np.float64,
    )


def make_feature_union(config: ProjectConfig) -> FeatureUnion:
    """Build the exact character + word TF-IDF representation from the reference project."""
    settings = config.section("text")["tfidf"]
    return FeatureUnion(
        [
            ("char", _make_vectorizer(settings["character"])),
            ("word", _make_vectorizer(settings["word"])),
        ]
    )


def make_feature_selector(config: ProjectConfig) -> SelectKBest:
    """Build supervised Chi-square selection for the configured top features."""
    return SelectKBest(score_func=chi2, k=int(config.section("text")["select_k_best"]))


def make_text_feature_pipeline(config: ProjectConfig) -> Pipeline:
    return Pipeline([("cleaner", MultilingualTextCleaner()), ("features", make_feature_union(config))])


def run_text_preprocessing(config: ProjectConfig) -> dict[str, Any]:
    dataset, split, audit = audit_text_data(config)
    pipeline = make_text_feature_pipeline(config)
    train_matrix = pipeline.fit_transform(dataset.raw_texts[split.train])
    test_matrix = pipeline.transform(dataset.raw_texts[split.test])
    selector = make_feature_selector(config)
    selected_train = selector.fit_transform(train_matrix, dataset.labels[split.train])
    selected_test = selector.transform(test_matrix)
    output = ensure_dir(config.artifacts_dir / "text" / "features")
    joblib.dump(pipeline, output / "training_vectorizer.joblib")
    joblib.dump(selector, output / "training_feature_selector.joblib")
    feature_union = pipeline.named_steps["features"]
    char_vectorizer = dict(feature_union.transformer_list)["char"]
    word_vectorizer = dict(feature_union.transformer_list)["word"]
    statistics = {
        "representation": "FeatureUnion(character-boundary TF-IDF + word TF-IDF) followed by supervised Chi-square selection",
        "character": {
            "analyzer": char_vectorizer.analyzer,
            "ngram_range": list(char_vectorizer.ngram_range),
            "min_df": char_vectorizer.min_df,
            "max_features": char_vectorizer.max_features,
            "sublinear_tf": char_vectorizer.sublinear_tf,
            "lowercase": char_vectorizer.lowercase,
        },
        "word": {
            "analyzer": word_vectorizer.analyzer,
            "ngram_range": list(word_vectorizer.ngram_range),
            "min_df": word_vectorizer.min_df,
            "max_features": word_vectorizer.max_features,
            "sublinear_tf": word_vectorizer.sublinear_tf,
            "lowercase": word_vectorizer.lowercase,
        },
        "selected_feature_count": int(selected_train.shape[1]),
        "train_shape": list(train_matrix.shape),
        "test_shape": list(test_matrix.shape),
        "selected_train_shape": list(selected_train.shape),
        "selected_test_shape": list(selected_test.shape),
        "vocabulary_size": int(len(char_vectorizer.vocabulary_) + len(word_vectorizer.vocabulary_)),
        "train_nonzero": int(train_matrix.nnz),
        "test_nonzero": int(test_matrix.nnz),
        "fit_scope": "TF-IDF and Chi-square selection fitted on training partition only",
        "raw_text_transform_smoke_test": bool(pipeline.transform([dataset.raw_texts[split.test[0]]]).shape[0] == 1),
        "data_audit_rows": audit["clean_rows"],
    }
    write_json(output / "feature_statistics.json", statistics)
    return statistics
