"""
Unit tests for src/predict.py (Stage 9: Sentiment Prediction and Model Persistence).
"""

from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import SentimentTfidfExtractor
from src.predict import (
    SentimentPredictor,
    load_predictor,
    PredictionResult,
    BatchPredictionResult,
    DEFAULT_MODEL_PATH,
)


@pytest.fixture
def predictor():
    """Load default trained SentimentPredictor artifact."""
    assert DEFAULT_MODEL_PATH.exists(), f"Default model missing at {DEFAULT_MODEL_PATH}"
    return load_predictor()


def test_load_predictor_missing_file_raises_error():
    """Verify FileNotFoundError when model path does not exist."""
    fake_path = Path("models/non_existent_model_xyz.pkl")
    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        SentimentPredictor(model_path=fake_path)


def test_load_predictor_corrupt_file_raises_error(tmp_path):
    """Verify RuntimeError when loading an unparseable/corrupt model file."""
    corrupt_file = tmp_path / "corrupt_model.pkl"
    corrupt_file.write_text("NOT_A_VALID_PICKLE_FILE_CONTENT")

    with pytest.raises(RuntimeError, match="Failed to load model artifact"):
        SentimentPredictor(model_path=corrupt_file)


def test_predict_single_sentiment_and_probabilities(predictor):
    """Verify single comment inference returns accurate labels and probability scores."""
    # Positive sample
    pos_res = predictor.predict_single("Loved the flight! Super friendly crew and arrived early ❤️ 😊")
    assert isinstance(pos_res, PredictionResult)
    assert pos_res.predicted_sentiment == "Positive"
    assert pos_res.predicted_probability is not None
    assert pos_res.predicted_probability > 0.5
    assert pos_res.class_probabilities is not None
    assert "Positive" in pos_res.class_probabilities
    assert "Negative" in pos_res.class_probabilities
    assert "Neutral" in pos_res.class_probabilities
    assert abs(sum(pos_res.class_probabilities.values()) - 1.0) < 0.01

    # Negative sample
    neg_res = predictor.predict_single("Flight was delayed 4 hours and bags were lost! Horrible service 😡")
    assert neg_res.predicted_sentiment == "Negative"
    assert neg_res.predicted_probability is not None
    assert neg_res.predicted_probability > 0.5


def test_predict_single_empty_and_malformed_inputs(predictor):
    """Verify graceful fallback for empty, whitespace, None, and numeric inputs."""
    # None
    res_none = predictor.predict_single(None)
    assert res_none.predicted_sentiment == "Neutral"
    assert res_none.predicted_probability is None

    # Empty string
    res_empty = predictor.predict_single("")
    assert res_empty.predicted_sentiment == "Neutral"
    assert res_empty.predicted_probability is None

    # Whitespace only
    res_spaces = predictor.predict_single("     ")
    assert res_spaces.predicted_sentiment == "Neutral"
    assert res_spaces.predicted_probability is None

    # Symbols only
    res_punct = predictor.predict_single("???!!!! ...")
    assert res_punct.predicted_sentiment == "Neutral"

    # Numeric input
    res_num = predictor.predict_single(123456)
    assert res_num.predicted_sentiment in ["Positive", "Negative", "Neutral"]


def test_predict_batch_list_and_series(predictor):
    """Verify batch classification across Python lists and pandas Series."""
    raw_list = [
        "Great flight, thank you so much!",
        "Delayed 6 hours without any food or water.",
        "What time does boarding start for Denver?",
    ]

    res_batch = predictor.predict_batch(raw_list)

    assert isinstance(res_batch, BatchPredictionResult)
    assert res_batch.total_records == 3
    assert len(res_batch.predictions) == 3
    assert "predicted_sentiment" in res_batch.dataframe.columns
    assert "predicted_probability" in res_batch.dataframe.columns

    # Verify predictions
    assert res_batch.predictions[0].predicted_sentiment == "Positive"
    assert res_batch.predictions[1].predicted_sentiment == "Negative"


def test_predict_batch_dataframe_preserves_metadata(predictor):
    """Verify DataFrame batch processing retains extra metadata columns intact."""
    df_input = pd.DataFrame({
        "text": [
            "Smooth flight and great wifi!",
            "Terrible customer service on hold for 2 hours.",
        ],
        "airline": ["Delta", "United"],
        "date": ["2026-02-01", "2026-02-02"],
        "topic": ["in-flight", "customer-service"],
    })

    batch_res = predictor.predict_batch(df_input, text_col="text")

    out_df = batch_res.dataframe
    # Verify metadata columns preserved
    assert "airline" in out_df.columns
    assert "date" in out_df.columns
    assert "topic" in out_df.columns
    assert "original_text" in out_df.columns
    assert "clean_text" in out_df.columns
    assert "predicted_sentiment" in out_df.columns
    assert "predicted_probability" in out_df.columns
    assert len(out_df) == 2


def test_predict_batch_empty_collections(predictor):
    """Verify empty list and empty DataFrame inputs return empty batch results without error."""
    empty_list_res = predictor.predict_batch([])
    assert empty_list_res.total_records == 0
    assert len(empty_list_res.dataframe) == 0

    empty_df_res = predictor.predict_batch(pd.DataFrame(columns=["text"]))
    assert empty_df_res.total_records == 0


def test_predict_model_without_probabilities(tmp_path):
    """
    CRITICAL: Verify that when a model does NOT support probabilities (e.g. LinearSVC),
    predicted_probability is gracefully set to None as specified in requirements.
    """
    # Create simple LinearSVC pipeline without probability calibration
    extractor = SentimentTfidfExtractor(ngram_range=(1, 2), min_df=1)
    train_texts = pd.Series([
        "good service fast flight",
        "bad service delayed flight",
        "what is flight schedule",
    ])
    train_labels = pd.Series(["Positive", "Negative", "Neutral"])

    extractor.fit(train_texts)
    svm = LinearSVC(random_state=42)
    X_vec = extractor.transform(train_texts)
    svm.fit(X_vec, train_labels)

    svm_pipeline = Pipeline([
        ("tfidf", extractor.vectorizer),
        ("classifier", svm),
    ])

    svm_path = tmp_path / "svm_pipeline.pkl"
    joblib.dump(svm_pipeline, svm_path)

    svm_predictor = SentimentPredictor(model_path=svm_path)
    assert svm_predictor.supports_probabilities is False

    # Single prediction
    single_res = svm_predictor.predict_single("good service")
    assert single_res.predicted_sentiment == "Positive"
    assert single_res.predicted_probability is None
    assert single_res.class_probabilities is None

    # Batch prediction
    batch_res = svm_predictor.predict_batch(["bad service delayed", "good flight"])
    assert batch_res.total_records == 2
    assert all(p.predicted_probability is None for p in batch_res.predictions)
    assert batch_res.dataframe["predicted_probability"].isna().all()
