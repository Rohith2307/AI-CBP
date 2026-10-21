"""
Unit tests for src/data_loader.py (Stage 2: Dataset Acquisition & Ingestion).
"""

from pathlib import Path
import io
import pytest
import pandas as pd

from src.data_loader import load_dataset, auto_detect_columns, IngestionReport


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_airline_sentiment.csv"


def test_auto_detect_columns():
    """Verify that common column names are automatically recognized."""
    cols = ["tweet_id", "airline_sentiment", "text", "airline", "tweet_created", "negativereason"]
    mapping = auto_detect_columns(cols)

    assert mapping["text"] == "text"
    assert mapping["sentiment"] == "airline_sentiment"
    assert mapping["product"] == "airline"
    assert mapping["date"] == "tweet_created"
    assert mapping["topic"] == "negativereason"


def test_load_sample_dataset_success():
    """Verify that sample dataset loads correctly with accurate metrics."""
    assert SAMPLE_DATA_PATH.exists(), f"Sample dataset file not found at {SAMPLE_DATA_PATH}"

    df, report = load_dataset(SAMPLE_DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(report, IngestionReport)
    assert len(df) == 300
    assert "text" in df.columns
    assert "sentiment" in df.columns

    # Verify class distribution has 100 of each
    assert report.class_distribution == {
        "neutral": 100,
        "negative": 100,
        "positive": 100,
    }
    assert report.total_records == 300
    assert report.total_columns >= 2


def test_load_with_explicit_column_mapping():
    """Verify loading a custom dataset with non-standard column headers."""
    csv_content = """review_body,customer_feeling,post_timestamp
"Great product and fast delivery!",positive,2026-01-10
"Horrible experience, broken item.",negative,2026-01-11
"Average item, nothing special.",neutral,2026-01-12
"""
    buffer = io.StringIO(csv_content)
    custom_mapping = {
        "text": "review_body",
        "sentiment": "customer_feeling",
        "date": "post_timestamp",
    }

    df, report = load_dataset(buffer, column_mapping=custom_mapping)

    assert "text" in df.columns
    assert "sentiment" in df.columns
    assert "date" in df.columns
    assert len(df) == 3
    assert df.loc[0, "text"] == "Great product and fast delivery!"
    assert df.loc[0, "sentiment"] == "positive"
    assert report.class_distribution == {"positive": 1, "negative": 1, "neutral": 1}


def test_load_missing_file_raises_error():
    """Verify that non-existent file path raises FileNotFoundError."""
    non_existent = Path("data/raw/does_not_exist_xyz123.csv")
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_dataset(non_existent)


def test_load_empty_file_raises_error(tmp_path):
    """Verify that an empty file raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")

    with pytest.raises(ValueError, match="Dataset file is empty"):
        load_dataset(empty_file)


def test_load_empty_dataframe_raises_error():
    """Verify that an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset contains no rows or records"):
        load_dataset(empty_df)


def test_load_missing_text_column_raises_error():
    """Verify that missing text column raises descriptive ValueError."""
    csv_content = """id,rating,score
1,5,100
2,1,20
"""
    buffer = io.StringIO(csv_content)
    with pytest.raises(ValueError, match="Missing required 'text' column"):
        load_dataset(buffer)


def test_load_missing_sentiment_for_training_raises_error():
    """Verify that missing sentiment column raises ValueError when require_sentiment=True."""
    csv_content = """text,author
"Just received my order today!",user_1
"""
    buffer = io.StringIO(csv_content)
    with pytest.raises(ValueError, match="Missing required 'sentiment' column"):
        load_dataset(buffer, require_sentiment=True)

    # When require_sentiment=False (inference mode), it should succeed
    buffer.seek(0)
    df, report = load_dataset(buffer, require_sentiment=False)
    assert len(df) == 1
    assert "text" in df.columns


def test_load_encoding_fallback_latin1(tmp_path):
    """Verify graceful handling and fallback when encountering non-UTF8 encodings."""
    latin1_file = tmp_path / "latin1_sample.csv"
    # Write characters with latin-1 encoding that fail strict UTF-8 decoding
    # e.g., 'café' encoded in latin-1 (0xe9)
    latin1_bytes = 'text,sentiment\n"Super café & crêpes! Très bien.",positive\n'.encode("latin-1")
    latin1_file.write_bytes(latin1_bytes)

    df, report = load_dataset(latin1_file)

    assert len(df) == 1
    assert "café" in df.loc[0, "text"]
    assert report.encoding_used == "latin-1"
    assert len(report.warnings) > 0
    assert "Falling back to 'latin-1'" in report.warnings[0]
