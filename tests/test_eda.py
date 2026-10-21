"""
Unit tests for src/eda.py (Stage 5: Exploratory Data Analysis).
"""

from pathlib import Path
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.eda import (
    compute_sentiment_summary,
    compute_length_statistics,
    compute_top_terms,
    compute_product_insights,
    compute_temporal_insights,
    compute_complaint_topics,
    create_sentiment_bar_chart,
    create_sentiment_donut_chart,
    create_length_distribution_chart,
    create_top_words_bar_chart,
    create_product_sentiment_chart,
    create_temporal_trend_chart,
    create_complaint_topics_chart,
)


SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "validated_sample.csv"


@pytest.fixture
def sample_df():
    """Load validated sample dataset."""
    assert SAMPLE_PATH.exists()
    return pd.read_csv(SAMPLE_PATH)


def test_compute_sentiment_summary(sample_df):
    """Verify sentiment distribution and class balance metrics."""
    summary = compute_sentiment_summary(sample_df)

    assert summary["total_records"] == 299
    assert summary["counts"]["Negative"] == 100
    assert summary["counts"]["Neutral"] == 100
    assert summary["counts"]["Positive"] == 99

    pct_sum = sum(summary["percentages"].values())
    assert abs(pct_sum - 100.0) < 0.2  # Rounded percentages sum to ~100%
    assert summary["imbalance_ratio"] == 1.01  # Near perfectly balanced sample
    assert isinstance(summary["explanation"], str)
    assert len(summary["explanation"]) > 20


def test_compute_sentiment_summary_missing_column():
    """Verify ValueError when sentiment column is absent."""
    df = pd.DataFrame({"text": ["hello", "world"]})
    with pytest.raises(ValueError, match="must contain a 'sentiment' column"):
        compute_sentiment_summary(df)


def test_compute_length_statistics(sample_df):
    """Verify character and word length calculation."""
    stats = compute_length_statistics(sample_df)

    assert "overall" in stats
    assert "by_sentiment" in stats
    assert stats["overall"]["mean_char_length"] > 0
    assert stats["overall"]["mean_word_count"] > 0

    assert "Negative" in stats["by_sentiment"]
    assert "Positive" in stats["by_sentiment"]
    assert "Neutral" in stats["by_sentiment"]
    assert stats["by_sentiment"]["Negative"]["mean_words"] > 0
    assert stats["by_sentiment"]["Positive"]["mean_words"] > 0
    assert isinstance(stats["explanation"], str)


def test_compute_top_terms(sample_df):
    """Verify top keywords extraction for specific sentiments."""
    pos_terms = compute_top_terms(sample_df, sentiment="Positive", top_n=5)
    neg_terms = compute_top_terms(sample_df, sentiment="Negative", top_n=5)

    assert len(pos_terms) == 5
    assert len(neg_terms) == 5

    # Check structure
    assert isinstance(pos_terms[0], tuple)
    assert isinstance(pos_terms[0][0], str)
    assert isinstance(pos_terms[0][1], int)

    # Positive feedback should feature praise terms
    pos_words = [t[0] for t in pos_terms]
    assert any(term in pos_words for term in ["thank", "great", "love", "emoji_pos", "best"])


def test_compute_product_insights(sample_df):
    """Verify product-level sentiment calculations and graceful handling when missing."""
    # When product column exists
    prod_insights = compute_product_insights(sample_df)
    assert prod_insights is not None
    assert "highest_neg_product" in prod_insights
    assert isinstance(prod_insights["explanation"], str)

    # When product column is missing
    df_no_prod = sample_df.drop(columns=["product"])
    assert compute_product_insights(df_no_prod) is None


def test_compute_temporal_insights(sample_df):
    """Verify temporal trend calculations and graceful handling when dates are missing."""
    temp_insights = compute_temporal_insights(sample_df)
    assert temp_insights is not None
    assert "daily_breakdown" in temp_insights
    assert "peak_negative_day" in temp_insights
    assert isinstance(temp_insights["explanation"], str)

    # When date columns are missing
    df_no_date = sample_df.drop(columns=["date", "date_parsed"])
    assert compute_temporal_insights(df_no_date) is None


def test_compute_complaint_topics(sample_df):
    """Verify complaint topic extraction and graceful fallback when missing."""
    topics = compute_complaint_topics(sample_df)
    assert topics is not None
    assert "counts" in topics
    assert "percentages" in topics
    assert "top_issue" in topics
    assert isinstance(topics["explanation"], str)

    # When topic column is missing
    df_no_topic = sample_df.drop(columns=["topic"])
    assert compute_complaint_topics(df_no_topic) is None


def test_plotly_chart_creators(sample_df):
    """Verify that all visualization generators construct valid Plotly Figures."""
    fig_bar = create_sentiment_bar_chart(sample_df)
    assert isinstance(fig_bar, go.Figure)

    fig_donut = create_sentiment_donut_chart(sample_df)
    assert isinstance(fig_donut, go.Figure)

    fig_len = create_length_distribution_chart(sample_df)
    assert isinstance(fig_len, go.Figure)

    fig_words = create_top_words_bar_chart(sample_df, sentiment="Positive")
    assert isinstance(fig_words, go.Figure)

    fig_prod = create_product_sentiment_chart(sample_df)
    assert isinstance(fig_prod, go.Figure)

    fig_temp = create_temporal_trend_chart(sample_df)
    assert isinstance(fig_temp, go.Figure)

    fig_topics = create_complaint_topics_chart(sample_df)
    assert isinstance(fig_topics, go.Figure)
