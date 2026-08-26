import numpy as np

from language_identification.config import load_config
from language_identification.speech.data import load_speech_data


def test_actual_speech_schema_and_metadata_boundary() -> None:
    dataset = load_speech_data(load_config())
    assert len(dataset.frame) == 720
    assert len(dataset.feature_columns) == 133
    assert dataset.target_column == "lang"
    assert {"filepath", "lang", "gender"}.isdisjoint(dataset.feature_columns)
    assert np.isfinite(dataset.features.to_numpy()).all()
    assert len(set(dataset.groups)) < len(dataset.groups)

