from language_identification.common.model_registry import (
    SPEECH_CLASSIFIERS,
    SPEECH_CLUSTERERS,
    TEXT_CLASSIFIERS,
    TEXT_CLUSTERERS,
    assert_exact_model_counts,
)


def test_exactly_four_distinct_families_per_task() -> None:
    assert_exact_model_counts()
    for registry in (SPEECH_CLASSIFIERS, SPEECH_CLUSTERERS, TEXT_CLASSIFIERS):
        assert len(registry) == 4
        assert len({entry["family"] for entry in registry.values()}) == 4
    assert len(TEXT_CLUSTERERS) == 4
    assert set(TEXT_CLUSTERERS) == {"kmeans", "dbscan", "agglomerative", "gaussian_mixture",}
