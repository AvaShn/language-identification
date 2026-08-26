from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language_identification.common.splitting import stratified_group_holdout
from language_identification.config import load_config
from language_identification.speech.data import load_speech_data
from language_identification.text.data import load_text_data
from language_identification.text.features import make_text_feature_pipeline


def test_reduced_end_to_end_data_to_predictions() -> None:
    config = load_config()
    speech = load_speech_data(config)
    speech_split = stratified_group_holdout(speech.labels, speech.groups, config.test_size, config.seed)
    model = Pipeline(
        [("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
    columns = speech.feature_columns[:12]
    model.fit(speech.features.iloc[speech_split.train][columns], speech.labels[speech_split.train])
    assert len(model.predict(speech.features.iloc[speech_split.test[:4]][columns])) == 4

    text = load_text_data(config)
    text_split = stratified_group_holdout(text.labels, text.groups, config.test_size, config.seed)
    features = make_text_feature_pipeline(config)
    features.fit(text.raw_texts[text_split.train])
    assert features.transform(text.raw_texts[text_split.test[:4]]).shape[0] == 4

