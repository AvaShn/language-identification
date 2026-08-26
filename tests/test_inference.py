from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language_identification.config import load_config
from language_identification.speech.inference import predict_speech_row
from language_identification.text.cleaning import MultilingualTextCleaner
from language_identification.text.features import make_feature_union
from language_identification.text.inference import predict_text


def test_speech_feature_row_inference(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [0.0, 0.1, 1.0, 1.1], "b": [0.0, 0.2, 1.0, 1.2]})
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())]).fit(frame, ["x", "x", "y", "y"])
    path = tmp_path / "speech.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": ["a", "b"]}, path)
    assert predict_speech_row({"a": 0.05, "b": 0.1}, path) in {"x", "y"}


def test_raw_text_inference_bundle(tmp_path: Path) -> None:
    config = load_config()
    pipeline = Pipeline(
        [
            ("cleaner", MultilingualTextCleaner()),
            ("features", make_feature_union(config)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    texts = ["bonjour le monde", "bonjour encore", "salut le monde", "hallo welt", "guten morgen welt", "hallo noch einmal"]
    labels = ["fr", "fr", "fr", "de", "de", "de"]
    pipeline.fit(texts, labels)
    path = tmp_path / "text.joblib"
    joblib.dump({"pipeline": pipeline}, path)
    assert predict_text("bonjour à tous", path) in {"fr", "de"}
