import numpy as np

from language_identification.common.splitting import stratified_group_holdout
from language_identification.config import load_config
from language_identification.speech.data import load_speech_data
from language_identification.text.data import load_text_data


def _assert_group_safe(labels: np.ndarray, groups: np.ndarray) -> None:
    split = stratified_group_holdout(labels, groups, 0.2, 42)
    assert not (set(groups[split.train]) & set(groups[split.test]))
    assert set(labels[split.train]) == set(labels)
    assert set(labels[split.test]) == set(labels)


def test_speech_and_text_group_splits_are_disjoint() -> None:
    config = load_config()
    speech = load_speech_data(config)
    text = load_text_data(config)
    _assert_group_safe(speech.labels, speech.groups)
    _assert_group_safe(text.labels, text.groups)

