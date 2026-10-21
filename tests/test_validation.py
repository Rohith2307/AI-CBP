"""
Unit tests for src/validation.py (Stage 3: Data Validation).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.validation import (
    validate_dataset,
    normalize_sentiment_label,
    save_validated_dataset,
    ValidationResult,
    CANONICAL_SENTIMENTS,
)


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_airline_sentiment.csv"


def test_normalize_sentiment_label():
    """Verify various sentiment variations map to canonical Positive, Negative, Neutral."""
    # Positive variations
    assert normalize_sentiment_label("positive") == "Positive"
    assert normalize_sentiment_label("POSITIVE") == "Positive"
    assert normalize_sentiment_label("pos") == "Positive"
    assert normalize_sentiment_label("1") == "Positive"
    assert normalize_sentiment_label("good") == "Positive"

    # Negative variations
    assert normalize_sentiment_label("negative") == "Negative"
    assert normalize_sentiment_label("NEGATIVE") == "Negative"
    assert normalize_sentiment_label("neg") == "Negative"
    assert normalize_sentiment_label("-1") == "Negative"
    assert normalize_sentiment_label("bad") == "Negative"

    # Neutral variations
    assert normalize_sentiment_label("neutral") == "Neutral"
    assert normalize_sentiment_label("NEUTRAL") == "Neutral"
    assert normalize_sentiment_label("neu") == "Neutral"
    assert normalize_sentiment_label("0") == "Neutral"

    # Unrecognized / NaN
    assert normalize_sentiment_label(None) is None
    assert normalize_sentiment_label(np.nan) is None
    assert normalize_sentiment_label("irrelevant") is None
    assert normalize_sentiment_label("unknown") is None


def test_validate_clean_sample_dataset():
    """Verify validation on the clean sample airline dataset."""
    df, _ = load_dataset(SAMPLE_DATA_PATH)
    result = validate_dataset(df)

    assert isinstance(result, ValidationResult)
    # The real sample contains 1 duplicate tweet ("@VirginAmerica Thanks!"), so 299 valid and 1 duplicate
    assert len(result.valid_df) == 299
    assert len(result.invalid_df) == 1
    assert result.invalid_df.iloc[0]["removal_reason"] == "duplicate_text"
    assert result.report.duplicate_count == 1

    # Verify canonical sentiments
    assert set(result.valid_df["sentiment"].unique()) == CANONICAL_SENTIMENTS
    assert result.report.class_distribution == {
        "Positive": 99,
        "Negative": 100,
        "Neutral": 100,
    }

    # Verify metrics computed
    assert result.report.valid_records == 299
    assert result.report.total_records == 300
    assert result.report.avg_char_length > 0
    assert result.report.avg_word_count > 0
    assert "char_length" in result.valid_df.columns
    assert "word_count" in result.valid_df.columns
    assert "original_text" in result.valid_df.columns


def test_validate_flags_missing_and_empty_text():
    """Verify missing (NaN) and empty/whitespace texts are cleanly isolated and logged."""
    raw_data = pd.DataFrame({
        "text": ["Valid good text", None, "   ", "Another valid text"],
        "sentiment": ["positive", "neutral", "negative", "neutral"],
    })

    result = validate_dataset(raw_data)

    assert len(result.valid_df) == 2
    assert len(result.invalid_df) == 2

    # Check reasons in invalid_df
    reasons = result.invalid_df["removal_reason"].tolist()
    assert "missing_text" in reasons
    assert "empty_text" in reasons

    assert result.report.missing_text_count == 1
    assert result.report.empty_text_count == 1
    assert result.report.valid_records == 2


def test_validate_duplicate_detection():
    """Verify duplicate text records are detected and audited without data loss."""
    raw_data = pd.DataFrame({
        "text": [
            "Great service today!",
            "Flight was delayed by 3 hours.",
            "Great service today!",  # duplicate
            "Everything was okay.",
        ],
        "sentiment": ["positive", "negative", "positive", "neutral"],
    })

    result = validate_dataset(raw_data, drop_duplicates=True)

    assert len(result.valid_df) == 3
    assert len(result.invalid_df) == 1
    assert result.invalid_df.iloc[0]["removal_reason"] == "duplicate_text"
    assert result.report.duplicate_count == 1
    assert result.report.valid_records == 3


def test_validate_unrecognized_sentiment_labels():
    """Verify invalid sentiment labels are rejected with descriptive reasons."""
    raw_data = pd.DataFrame({
        "text": ["Good item", "Spam post", "Neutral item"],
        "sentiment": ["positive", "irrelevant_ad_content", "neutral"],
    })

    result = validate_dataset(raw_data, require_sentiment=True)

    assert len(result.valid_df) == 2
    assert len(result.invalid_df) == 1
    assert "unrecognized_sentiment_label" in result.invalid_df.iloc[0]["removal_reason"]
    assert result.report.invalid_sentiment_count == 1


def test_missing_text_column_raises_error():
    """Verify error when text column is absent."""
    raw_data = pd.DataFrame({"body": ["Some comment"], "sentiment": ["positive"]})
    with pytest.raises(ValueError, match="DataFrame must contain a 'text' column"):
        validate_dataset(raw_data)


def test_save_validated_dataset(tmp_path):
    """Verify persisting validated dataset to disk."""
    df, _ = load_dataset(SAMPLE_DATA_PATH)
    result = validate_dataset(df)

    out_file = tmp_path / "validated_data.csv"
    saved_path = save_validated_dataset(result.valid_df, out_file)

    assert saved_path.exists()
    reloaded = pd.read_csv(saved_path)
    assert len(reloaded) == 299
    assert "char_length" in reloaded.columns
    assert set(reloaded["sentiment"].unique()) == CANONICAL_SENTIMENTS
