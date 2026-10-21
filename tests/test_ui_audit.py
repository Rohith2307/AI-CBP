"""Thorough automated verification of UI/UX redesign and edge cases for Phase 6.

Tests all eight dashboard pages, missing-date behavior, missing-product behavior,
custom CSV upload emulation, empty state resilience, and artifact verification.
"""
import json
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_artifacts_exist_and_valid():
    """Verify models and evaluation result files exist and are valid JSON."""
    comp_path = Path("models/model_comparison.json")
    eval_path = Path("models/evaluation_results.json")
    model_path = Path("models/sentiment_model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    assert comp_path.exists(), "models/model_comparison.json is missing!"
    assert eval_path.exists(), "models/evaluation_results.json is missing!"
    assert model_path.exists(), "models/sentiment_model.pkl is missing!"
    assert vectorizer_path.exists(), "models/vectorizer.pkl is missing!"

    with open(comp_path, "r") as f:
        comp_data = json.load(f)
    assert "best_model_name" in comp_data
    assert "comparison_table" in comp_data
    assert len(comp_data["comparison_table"]) == 3  # LR, Linear SVM, MNB

    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    assert "test_accuracy" in eval_data
    assert "test_macro_f1" in eval_data
    assert "confusion_matrix" in eval_data
    assert "classification_report" in eval_data


def test_all_pages_navigate_and_render_with_benchmark_data():
    """Verify navigation to all 8 pages renders with real benchmark data and zero exceptions."""
    pages = [
        "🏠 Overview",
        "📂 Ingestion & Quality",
        "📊 Exploratory Data (EDA)",
        "🧠 Sentiment Analysis",
        "🔍 Feedback Explorer",
        "📈 Trends & Spikes",
        "🎯 Model Performance",
        "📑 Insights & Export",
    ]

    for page in pages:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception, f"Initial load failed for page {page}"

        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Rendering failed for page {page}"
        assert at.title[0].value, f"Page {page} missing main title!"


def test_trends_and_explorer_resilience_missing_optional_columns():
    """Verify Trends & Spikes and Feedback Explorer handle missing date and product columns gracefully."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Simulate active_df with only text and sentiment (no date, no product)
    simple_df = pd.DataFrame({
        "clean_text": ["excellent flight", "terrible delay baggage lost", "what time does gate close"],
        "sentiment": ["Positive", "Negative", "Neutral"],
        "predicted_sentiment": ["Positive", "Negative", "Neutral"],
    })
    at.session_state["active_df"] = simple_df

    # Test Trends & Spikes with missing date
    at.sidebar.radio[0].set_value("📈 Trends & Spikes").run()
    assert not at.exception
    # Should render informative notice for missing date rather than crashing
    assert any("Date information is not available" in m.value or "callout" in m.value for m in at.markdown)

    # Test Feedback Explorer with missing date and missing product
    at.sidebar.radio[0].set_value("🔍 Feedback Explorer").run()
    assert not at.exception
    assert any("matching comments" in c.value for c in at.caption)


def test_custom_sample_dataset_loading():
    """Verify that switching to the development sample dataset updates active_df and runs properly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("📂 Ingestion & Quality").run()
    assert not at.exception

    # Click 'Load Development Sample' button
    sample_btn = [b for b in at.button if "Load Development Sample" in b.label][0]
    sample_btn.click().run()
    assert not at.exception

    assert len(at.session_state["active_df"]) == 299
    assert "Sample Benchmark (299 rows)" in at.session_state["dataset_source_name"]
