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


def test_eda_page_redesign_components():
    """Verify redesigned EDA page headers and lexical breakdown sections."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("📊 Exploratory Data (EDA)").run()
    assert not at.exception
    assert any("Linguistic Telemetry" in m.value for m in at.markdown)
    assert any("Lexical Frequency Breakdown" in m.value for m in at.markdown)


def test_sentiment_analysis_batch_inference_interactive():
    """Verify batch inference interface runs and computes summary metrics."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("🧠 Sentiment Analysis").run()
    assert not at.exception

    # Locate the batch text area (second text area on page)
    assert len(at.text_area) >= 2
    at.text_area[1].set_value("Flight cancelled with zero notice!\nCabin crew was wonderful and attentive.\nIs gate 4 open?").run()

    batch_btn = [b for b in at.button if b.label == "Run Batch Inference"][0]
    batch_btn.click().run()
    assert not at.exception

    # Verify batch completion message and metrics
    assert any("Successfully classified" in m.value for m in at.success)
    assert any("Batch Volume" in m.value for m in at.markdown)


def test_trends_page_redesign_components():
    """Verify redesigned Trends & Spikes headers and causality notice."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("📈 Trends & Spikes").run()
    assert not at.exception
    assert any("Grievance Surge Detection" in m.value for m in at.markdown)
    assert any("Methodological Causality Notice" in m.value for m in at.markdown)


def test_trends_page_with_active_spike_alert_rendering():
    """Verify Trends & Spikes executes render_spike_alert without NameError when spikes exist."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Construct synthetic multi-day dataset where Day 4 exhibits a major negative grievance spike
    dates = []
    sentiments = []
    # Days 1-3: 50 comments/day, 20% negative baseline
    for day in ["2025-01-01", "2025-01-02", "2025-01-03"]:
        dates.extend([f"{day} 12:00:00"] * 50)
        sentiments.extend(["Negative"] * 10 + ["Positive"] * 40)
    # Day 4: 50 comments, 90% negative (triggers spike alert)
    dates.extend(["2025-01-04 12:00:00"] * 50)
    sentiments.extend(["Negative"] * 45 + ["Positive"] * 5)

    spike_df = pd.DataFrame({
        "text": ["Flight review sample"] * 200,
        "clean_text": ["Flight review sample"] * 200,
        "sentiment": sentiments,
        "predicted_sentiment": sentiments,
        "date": dates,
        "product": ["AirlineX"] * 200,
    })
    at.session_state["active_df"] = spike_df

    at.sidebar.radio[0].set_value("📈 Trends & Spikes").run()
    assert not at.exception
    # Ensure spike alert HTML rendered successfully by render_spike_alert
    assert any("Statistical Spike Alert" in m.value or "spike-card" in m.value for m in at.markdown)


def test_feedback_explorer_search_and_filter():
    """Verify Feedback Explorer search and sentiment filters update output."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("🔍 Feedback Explorer").run()
    assert not at.exception
    assert any("Granular Comment Search & Filter Telemetry" in m.value for m in at.markdown)

    # Apply search filter
    search_input = at.text_input[0]
    search_input.set_value("flight").run()
    assert not at.exception
    assert any("matching comments" in c.value for c in at.caption)


def test_model_performance_leaderboard():
    """Verify Candidate Benchmark leaderboard, winning model card, and test set audit."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("🎯 Model Performance").run()
    assert not at.exception
    assert any("Candidate Benchmark & Test Set Audit" in m.value for m in at.markdown)
    assert any("Selected Production Model" in m.value for m in at.markdown)
    assert any("Unbiased Held-Out Test Split Audit" in m.value for m in at.markdown)
    assert len(at.table) >= 2  # Comparison table + Per-class metrics table


def test_insights_export_downloads():
    """Verify Executive Insights KPIs, driver cards, and download buttons."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("📑 Insights & Export").run()
    assert not at.exception
    assert any("Executive Intelligence Dossier" in m.value for m in at.markdown)
    assert any("Executive Summary" in m.value for m in at.markdown)
    assert any("Top Complaint Drivers" in m.value for m in at.markdown)
    assert any("Top Praise Drivers" in m.value for m in at.markdown)
    labels = [b.label for b in at.download_button]
    assert any("Download Predictions CSV" in l for l in labels)
    assert any("Download Executive Report (.md)" in l for l in labels)

