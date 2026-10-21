"""
Unit tests for src/insights.py (Stage 10: Analytics and Insight Generation).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.insights import (
    compute_kpis,
    extract_sentiment_keywords,
    compute_product_comparison,
    compute_daily_sentiment_trends,
    detect_negative_spikes,
    generate_insights,
    InsightReport,
    CAUSALITY_DISCLAIMER,
)
from src.predict import load_predictor


SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "validated_sample.csv"


@pytest.fixture
def predicted_sample_df():
    """Load sample dataset and run prediction to supply actual predicted_sentiment."""
    df = pd.read_csv(SAMPLE_PATH)
    predictor = load_predictor()
    batch_res = predictor.predict_batch(df, text_col="text")
    return batch_res.dataframe


def test_compute_kpis_calculation():
    """Verify calculation of counts, percentages, and Net Sentiment Score (NSS)."""
    df = pd.DataFrame({
        "predicted_sentiment": ["Positive", "Positive", "Positive", "Negative", "Neutral"]
    })
    kpis = compute_kpis(df)

    assert kpis["total_comments"] == 5
    assert kpis["positive_count"] == 3
    assert kpis["negative_count"] == 1
    assert kpis["neutral_count"] == 1
    assert kpis["positive_rate"] == 60.0
    assert kpis["negative_rate"] == 20.0
    assert kpis["neutral_rate"] == 20.0
    # Net Sentiment Score = 60.0 - 20.0 = +40.0
    assert kpis["net_sentiment_score"] == 40.0


def test_compute_kpis_empty_and_error():
    """Verify empty DataFrame returns zero KPIs and missing column raises ValueError."""
    empty_df = pd.DataFrame({"predicted_sentiment": []})
    kpis = compute_kpis(empty_df)
    assert kpis["total_comments"] == 0
    assert kpis["net_sentiment_score"] == 0.0

    bad_df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError, match="Column 'predicted_sentiment' not found"):
        compute_kpis(bad_df)


def test_extract_sentiment_keywords(predicted_sample_df):
    """Verify keyword extraction for negative complaints and positive praise."""
    complaints = extract_sentiment_keywords(predicted_sample_df, "Negative", top_n=5)
    praise = extract_sentiment_keywords(predicted_sample_df, "Positive", top_n=5)

    assert len(complaints) > 0
    assert len(praise) > 0

    # Praise should contain positive sentiment words
    praise_terms = [t[0] for t in praise]
    assert any(term in praise_terms for term in ["thank", "great", "love", "emoji_pos", "best"])


def test_compute_product_comparison(predicted_sample_df):
    """Verify product-level sentiment comparison and graceful missing handling."""
    # With product column
    res = compute_product_comparison(predicted_sample_df, product_col="product")
    assert res is not None
    assert "products_evaluated" in res
    assert "highest_negative_product" in res

    # Without product column
    df_no_prod = predicted_sample_df.drop(columns=["product"])
    assert compute_product_comparison(df_no_prod, product_col="product") is None


def test_compute_daily_sentiment_trends(predicted_sample_df):
    """Verify daily aggregation and missing date graceful fallback."""
    trends = compute_daily_sentiment_trends(predicted_sample_df, date_col="date_parsed")
    assert trends is not None
    assert "negative_rate" in trends.columns
    assert "positive_rate" in trends.columns
    assert "total" in trends.columns

    # Missing date column
    df_no_date = predicted_sample_df.drop(columns=["date", "date_parsed"])
    assert compute_daily_sentiment_trends(df_no_date) is None


def test_detect_negative_spikes_guards_against_tiny_samples():
    """
    CRITICAL: Verify spike detection ignores days with low sample volume
    to avoid triggering false alarms on small sample sizes (e.g. 2/2 = 100%).
    """
    mock_trends = pd.DataFrame({
        "total": [2, 100, 120, 110, 150],  # Day 0 has only 2 comments
        "negative": [2, 45, 50, 48, 120],  # Day 4 has genuine surge (120/150 = 80%)
        "positive": [0, 30, 40, 35, 15],
        "neutral": [0, 25, 30, 27, 15],
        "negative_rate": [100.0, 45.0, 41.67, 43.64, 80.0],
        "positive_rate": [0.0, 30.0, 33.33, 31.82, 10.0],
    }, index=["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])

    # Require minimum 30 comments per day
    alerts = detect_negative_spikes(
        daily_trends=mock_trends,
        min_daily_volume=30,
        min_days=3,
        std_multiplier=1.5,
        absolute_threshold_pct=60.0,
    )

    # Day 0 (2 comments at 100%) must NOT trigger an alert due to small sample guard
    alert_dates = [a.date for a in alerts]
    assert "2026-01-01" not in alert_dates

    # Day 4 (150 comments at 80%) MUST trigger an alert
    assert "2026-01-05" in alert_dates
    assert alerts[0].negative_rate == 80.0
    assert alerts[0].total_count == 150


def test_causality_disclaimer_present():
    """Verify that the framework contains an explicit correlation-not-causation disclaimer."""
    assert "Statistical correlation does not establish causation" in CAUSALITY_DISCLAIMER


def test_generate_insights_end_to_end(predicted_sample_df):
    """Verify end-to-end analytics report generation."""
    report = generate_insights(
        df=predicted_sample_df,
        text_col="text",
        sentiment_col="predicted_sentiment",
        product_col="product",
        date_col="date_parsed",
        min_daily_volume=5,
    )

    assert isinstance(report, InsightReport)
    assert report.kpis["total_comments"] == len(predicted_sample_df)
    assert len(report.executive_summary) > 50
    assert report.causality_disclaimer == CAUSALITY_DISCLAIMER
    assert isinstance(report.top_complaint_terms, list)
    assert isinstance(report.top_positive_terms, list)
