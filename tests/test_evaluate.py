"""
Unit tests for src/evaluate.py (Stage 8: Evaluation and Error Analysis).
"""

from pathlib import Path
import json
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.features import split_dataset
from src.train import train_and_compare_models
from src.evaluate import (
    categorize_error,
    evaluate_model_performance,
    perform_error_analysis,
    create_confusion_matrix_chart,
    evaluate_and_audit_model,
    EvaluationReport,
    CANONICAL_CLASSES,
)


SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "validated_sample.csv"


@pytest.fixture
def sample_df():
    """Load validated sample dataset."""
    assert SAMPLE_PATH.exists()
    return pd.read_csv(SAMPLE_PATH)


def test_categorize_error():
    """Verify linguistic error categorization logic."""
    # Sarcasm test
    arch_sarcasm, exp_sarcasm = categorize_error(
        text="@Airline Thanks for cancelling our flight, fantastic job!",
        true_label="Negative",
        pred_label="Positive",
    )
    assert arch_sarcasm == "Sarcasm / Irony"
    assert "superficial praise" in exp_sarcasm.lower() or "sarcasm" in exp_sarcasm.lower()

    # Complex Negation test
    arch_neg, _ = categorize_error(
        text="We haven't boarded yet so let's not get excited, but on time so far",
        true_label="Positive",
        pred_label="Negative",
    )
    assert arch_neg == "Complex Negation"

    # Ambiguous Inquiry test
    arch_inq, _ = categorize_error(
        text="Will flight 769 be delayed due to weather?",
        true_label="Neutral",
        pred_label="Negative",
    )
    assert arch_inq == "Ambiguous Inquiry"

    # Mixed Sentiment test
    arch_mixed, _ = categorize_error(
        text="Great flight crew and awesome seats, but flight delayed 4 hours",
        true_label="Negative",
        pred_label="Positive",
    )
    assert arch_mixed == "Mixed Sentiment"

    # Short / Low Context test
    arch_short, _ = categorize_error(
        text="smh delayed again",
        true_label="Negative",
        pred_label="Neutral",
    )
    assert arch_short == "Short / Low Context"


def test_evaluate_model_performance_metrics(sample_df, tmp_path):
    """Verify accuracy, macro metrics, and confusion matrix shapes."""
    exp = train_and_compare_models(sample_df, output_dir=tmp_path, cv=3, random_state=42)
    pipeline_path = tmp_path / "sentiment_model.pkl"
    import joblib
    model = joblib.load(pipeline_path)

    split = split_dataset(sample_df, random_state=42)
    perf = evaluate_model_performance(model, split.X_test, split.y_test)

    assert 0.0 <= perf["accuracy"] <= 1.0
    assert 0.0 <= perf["macro_f1"] <= 1.0
    assert 0.0 <= perf["macro_precision"] <= 1.0
    assert 0.0 <= perf["macro_recall"] <= 1.0

    cm = perf["confusion_matrix"]
    assert len(cm) == 3
    assert len(cm[0]) == 3
    assert sum(sum(row) for row in cm) == len(split.X_test)

    clf_rep = perf["classification_report"]
    for cls in CANONICAL_CLASSES:
        assert cls in clf_rep
        assert "f1-score" in clf_rep[cls]
        assert "precision" in clf_rep[cls]
        assert "recall" in clf_rep[cls]


def test_perform_error_analysis(sample_df, tmp_path):
    """Verify error audit breakdown, archetypes, and example extraction."""
    exp = train_and_compare_models(sample_df, output_dir=tmp_path, cv=3, random_state=42)
    pipeline_path = tmp_path / "sentiment_model.pkl"
    import joblib
    model = joblib.load(pipeline_path)

    split = split_dataset(sample_df, random_state=42)
    audit = perform_error_analysis(model, split.test_df, text_col="text", sentiment_col="sentiment")

    assert audit["total_test_samples"] == len(split.test_df)
    assert 0 <= audit["total_errors"] <= len(split.test_df)
    assert 0.0 <= audit["test_error_rate"] <= 1.0

    if audit["total_errors"] > 0:
        assert len(audit["confusion_breakdown"]) > 0
        assert len(audit["error_archetypes_count"]) > 0
        assert len(audit["sample_errors"]) > 0
        first_err = audit["sample_errors"][0]
        assert "text" in first_err
        assert "true_sentiment" in first_err
        assert "predicted_sentiment" in first_err
        assert "error_archetype" in first_err


def test_create_confusion_matrix_chart():
    """Verify Plotly heatmap figure generation for confusion matrix."""
    mock_cm = [[70, 10, 5], [8, 65, 7], [4, 6, 75]]
    fig = create_confusion_matrix_chart(mock_cm, class_names=CANONICAL_CLASSES)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_evaluate_and_audit_model_end_to_end(sample_df, tmp_path):
    """Verify end-to-end evaluation pipeline and JSON report persistence."""
    train_and_compare_models(sample_df, output_dir=tmp_path, cv=3, random_state=42)
    pipeline_path = tmp_path / "sentiment_model.pkl"

    report = evaluate_and_audit_model(
        model_path=pipeline_path,
        df=sample_df,
        text_col="text",
        sentiment_col="sentiment",
        output_dir=tmp_path,
        random_state=42,
    )

    assert isinstance(report, EvaluationReport)
    assert report.total_test_samples == 45
    assert report.total_errors >= 0
    assert len(report.limitations) >= 3

    # Check JSON was saved and matches
    json_path = tmp_path / "evaluation_results.json"
    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["test_accuracy"] == report.test_accuracy
    assert data["test_macro_f1"] == report.test_macro_f1
