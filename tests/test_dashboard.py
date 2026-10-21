"""Tests for the Streamlit sentiment analysis dashboard.

Verifies page navigation, interactive inference, empty-state resilience,
and data preparation helpers using Streamlit's AppTest testing framework.
"""

from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.app import prepare_df_with_sentiment

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_prepare_df_with_sentiment_basic():
    """Verify renaming a custom sentiment column to 'sentiment'."""
    df = pd.DataFrame({
        "text": ["Great flight", "Bad service"],
        "predicted_sentiment": ["Positive", "Negative"],
    })
    res = prepare_df_with_sentiment(df, "predicted_sentiment")
    assert "sentiment" in res.columns
    assert "predicted_sentiment" not in res.columns
    assert list(res["sentiment"]) == ["Positive", "Negative"]


def test_prepare_df_with_sentiment_duplicate_column_collision():
    """Verify that an existing 'sentiment' column is safely dropped when a different column is selected."""
    df = pd.DataFrame({
        "text": ["Flight was fine"],
        "sentiment": ["Neutral"],
        "predicted_sentiment": ["Positive"],
    })
    res = prepare_df_with_sentiment(df, "predicted_sentiment")
    assert list(res.columns).count("sentiment") == 1
    assert res["sentiment"].iloc[0] == "Positive"


def test_dashboard_initial_load():
    """Verify dashboard loads default state without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on initial load: {at.exception}"
    assert len(at.title) > 0


def test_dashboard_all_pages_render():
    """Verify all 8 navigation tabs render cleanly without uncaught exceptions."""
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

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Uncaught exception on page '{page}': {at.exception}"


def test_sentiment_analysis_single_prediction_interactive():
    """Verify single-comment interactive classification via AppTest."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("🧠 Sentiment Analysis").run()
    assert not at.exception

    # Enter positive text
    at.text_area[0].set_value("The flight attendants were exceptionally polite and helpful!").run()

    predict_btn = [b for b in at.button if b.label == "Predict Sentiment"][0]
    predict_btn.click().run()
    assert not at.exception

    # Check for sentiment badge rendering
    assert any("Positive" in m.value for m in at.markdown)


def test_empty_state_resilience():
    """Verify data-dependent pages display informative empty states when no data is loaded."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Clear active DataFrame
    at.session_state.active_df = pd.DataFrame()

    dependent_pages = [
        "🏠 Overview",
        "📊 Exploratory Data (EDA)",
        "🔍 Feedback Explorer",
        "📈 Trends & Spikes",
        "📑 Insights & Export",
    ]

    for page in dependent_pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Page '{page}' crashed on empty dataset"
        assert len(at.warning) > 0, f"Page '{page}' missing empty dataset warning"


def test_overview_page_redesign_components():
    """Verify redesigned Overview telemetry header, KPI cards, and charts render properly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("🏠 Overview").run()
    assert not at.exception
    assert any("Executive Telemetry" in m.value for m in at.markdown)
    assert any("Total Feedback Volume" in m.value for m in at.markdown)
    assert any("Net Sentiment Score" in m.value for m in at.markdown)


def test_ingestion_page_redesign_components():
    """Verify redesigned Ingestion & Quality page headers, tabs, and audit components."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("📂 Ingestion & Quality").run()
    assert not at.exception
    assert any("Data Hygiene Engine" in m.value for m in at.markdown)
    assert any("Data Quality & Hygiene Audit" in m.value for m in at.markdown)
    # Check benchmark buttons
    btn_labels = [b.label for b in at.button]
    assert any("Load Development Sample" in l for l in btn_labels)
