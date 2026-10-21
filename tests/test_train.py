"""
Unit tests for src/train.py (Stage 7: Model Training and Comparison).
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import split_dataset, SentimentTfidfExtractor
from src.train import (
    get_default_candidate_models,
    evaluate_model,
    train_and_compare_models,
    ModelMetrics,
    ExperimentComparison,
)


SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "validated_sample.csv"


@pytest.fixture
def sample_df():
    """Load validated sample dataset."""
    assert SAMPLE_PATH.exists()
    return pd.read_csv(SAMPLE_PATH)


def test_get_default_candidate_models():
    """Verify standard candidate models are instantiated correctly."""
    models = get_default_candidate_models()
    assert len(models) == 3
    assert "Multinomial Naive Bayes" in models
    assert "Logistic Regression" in models
    assert "Linear SVM" in models


def test_evaluate_model_single_candidate(sample_df):
    """Verify evaluation metrics computation for an individual candidate."""
    split = split_dataset(sample_df, random_state=42)
    extractor = SentimentTfidfExtractor(ngram_range=(1, 2), min_df=2)
    X_train_vec = extractor.fit_transform(split.X_train)
    X_val_vec = extractor.transform(split.X_val)

    models = get_default_candidate_models()
    nb_model = models["Multinomial Naive Bayes"]

    metrics, fitted_model = evaluate_model(
        model=nb_model,
        X_train=X_train_vec,
        y_train=split.y_train,
        X_val=X_val_vec,
        y_val=split.y_val,
        model_name="Multinomial Naive Bayes",
        cv=3,
        random_state=42,
    )

    assert isinstance(metrics, ModelMetrics)
    assert metrics.model_name == "Multinomial Naive Bayes"
    assert 0.0 <= metrics.val_accuracy <= 1.0
    assert 0.0 <= metrics.val_macro_f1 <= 1.0
    assert 0.0 <= metrics.val_macro_precision <= 1.0
    assert 0.0 <= metrics.val_macro_recall <= 1.0
    assert metrics.train_time_sec >= 0.0
    assert metrics.infer_time_ms >= 0.0
    assert "Positive" in metrics.class_metrics
    assert "Negative" in metrics.class_metrics
    assert "Neutral" in metrics.class_metrics
    assert metrics.supports_probabilities is True


def test_train_and_compare_models_sample(sample_df, tmp_path):
    """Verify training, comparison table generation, and artifact saving across all models."""
    exp = train_and_compare_models(
        df=sample_df,
        text_col="text",
        sentiment_col="sentiment",
        cv=3,
        random_state=42,
        output_dir=tmp_path,
    )

    assert isinstance(exp, ExperimentComparison)
    assert len(exp.metrics_list) == 3

    # Check comparison table
    comp_df = exp.comparison_table
    assert len(comp_df) == 3
    assert "Model" in comp_df.columns
    assert "Accuracy" in comp_df.columns
    assert "Macro F1" in comp_df.columns
    assert "Train Time (s)" in comp_df.columns
    assert "Infer Latency (ms)" in comp_df.columns

    # Verify best model selection
    assert exp.best_model_name in [m.model_name for m in exp.metrics_list]
    assert exp.best_macro_f1 > 0.0
    assert isinstance(exp.selection_rationale, str)

    # Verify artifacts were persisted on disk
    assert (tmp_path / "vectorizer.pkl").exists()
    assert (tmp_path / "sentiment_model.pkl").exists()
    assert (tmp_path / "model_comparison.json").exists()

    # Verify JSON content
    with open(tmp_path / "model_comparison.json", "r") as f:
        saved_json = json.load(f)
    assert saved_json["best_model_name"] == exp.best_model_name
    assert len(saved_json["models"]) == 3


def test_saved_deployment_pipeline(sample_df, tmp_path):
    """Verify that the persisted end-to-end Pipeline predicts on raw text directly."""
    exp = train_and_compare_models(
        df=sample_df,
        text_col="text",
        sentiment_col="sentiment",
        cv=3,
        random_state=42,
        output_dir=tmp_path,
    )

    pipeline_path = tmp_path / "sentiment_model.pkl"
    loaded_pipeline = joblib.load(pipeline_path)

    assert isinstance(loaded_pipeline, Pipeline)
    assert "tfidf" in loaded_pipeline.named_steps
    assert "classifier" in loaded_pipeline.named_steps

    raw_test_comments = pd.Series([
        "Flight delayed 5 hours, terrible service!",
        "Thank you for the fantastic flight! ❤️",
        "What is the gate number for Dallas?",
    ])

    predictions = loaded_pipeline.predict(raw_test_comments)
    assert len(predictions) == 3
    for pred in predictions:
        assert pred in ["Positive", "Negative", "Neutral"]


def test_model_training_reproducibility(sample_df, tmp_path):
    """Verify that identical random seeds yield identical comparison metrics."""
    exp1 = train_and_compare_models(sample_df, cv=3, random_state=42, output_dir=tmp_path / "run1")
    exp2 = train_and_compare_models(sample_df, cv=3, random_state=42, output_dir=tmp_path / "run2")

    assert exp1.best_model_name == exp2.best_model_name
    assert exp1.best_macro_f1 == exp2.best_macro_f1
    for m1, m2 in zip(exp1.metrics_list, exp2.metrics_list):
        assert m1.val_accuracy == m2.val_accuracy
        assert m1.val_macro_f1 == m2.val_macro_f1
