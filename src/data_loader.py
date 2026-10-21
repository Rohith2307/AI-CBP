"""
Data Ingestion Module for Customer Sentiment Analysis Framework.

Provides dataset-agnostic ingestion with flexible column mapping,
encoding fallbacks, schema checks, and ingestion summary generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import pandas as pd


# Default candidate aliases for auto-detecting common column names (case-insensitive)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "text": ["text", "tweettext", "comment", "review", "content", "message", "body", "tweet", "sentence"],
    "sentiment": ["sentiment", "airline_sentiment", "label", "target", "score", "rating", "class", "polarity"],
    "date": ["date", "tweetdate", "tweet_created", "created_at", "timestamp", "time", "datetime"],
    "product": ["product", "airline", "brand", "category", "item", "entity", "company"],
    "source": ["source", "platform", "channel", "origin", "site"],
    "topic": ["topic", "negativereason", "issue", "aspect", "reason", "subject"],
}


@dataclass
class IngestionReport:
    """Summary of data ingestion, schema detection, and raw record metrics."""
    total_records: int
    total_columns: int
    detected_columns: List[str]
    mapped_columns: Dict[str, str]
    encoding_used: str
    class_distribution: Optional[Dict[str, int]] = None
    warnings: List[str] = field(default_factory=list)


def auto_detect_columns(available_columns: List[str]) -> Dict[str, str]:
    """
    Automatically detect standard framework fields from available DataFrame columns.

    Args:
        available_columns: List of column names in the raw dataset.

    Returns:
        Dict mapping standard field names ('text', 'sentiment', etc.) to actual column names.
    """
    col_map: Dict[str, str] = {}
    normalized_cols = {col.strip().lower().replace("_", "").replace("-", ""): col for col in available_columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            norm_alias = alias.lower().replace("_", "").replace("-", "")
            if norm_alias in normalized_cols:
                col_map[standard_name] = normalized_cols[norm_alias]
                break

    return col_map


def load_dataset(
    source: Union[str, Path, io.StringIO, io.BytesIO, pd.DataFrame, Any],
    column_mapping: Optional[Dict[str, str]] = None,
    require_sentiment: bool = True,
    encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, IngestionReport]:
    """
    Load, standardize, and validate customer text data from CSV, buffer, or DataFrame.

    Args:
        source: File path, open file buffer, or pandas DataFrame.
        column_mapping: Optional explicit mapping of standard fields to source column names
                        (e.g., {'text': 'review_text', 'sentiment': 'user_rating'}).
        require_sentiment: If True, raises ValueError if a sentiment column is missing.
                           Set to False when loading unlabelled data for prediction.
        encoding: Preferred character encoding (defaults to attempting 'utf-8', then 'latin-1').

    Returns:
        Tuple of (standardized DataFrame, IngestionReport).

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If file is empty, missing required text column, or invalid.
    """
    warnings: List[str] = []
    used_encoding = encoding or "utf-8"

    # Step 1: Read raw input
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        used_encoding = "memory (DataFrame)"
    else:
        # Check path existence if string/Path
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"Dataset file is empty (0 bytes): {path}")

        # Attempt reading with specified or default encoding, with fallback
        try:
            df = pd.read_csv(source, encoding=used_encoding)
        except (UnicodeDecodeError, LookupError) as e:
            fallback_encoding = "latin-1"
            warnings.append(
                f"Failed to read with encoding '{used_encoding}' ({e}). "
                f"Falling back to '{fallback_encoding}'."
            )
            # If source was a buffer, reset pointer if possible
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=fallback_encoding)
            used_encoding = fallback_encoding

    if df.empty:
        raise ValueError("Dataset contains no rows or records.")

    raw_columns = list(df.columns)

    # Step 2: Establish column mapping
    final_mapping: Dict[str, str] = {}
    detected = auto_detect_columns(raw_columns)

    # Merge auto-detected mapping with user overrides
    if column_mapping:
        final_mapping.update(column_mapping)
        # Fill in any missing standard fields from detected
        for std_key, actual_col in detected.items():
            if std_key not in final_mapping and actual_col not in final_mapping.values():
                final_mapping[std_key] = actual_col
    else:
        final_mapping = detected

    # Step 3: Schema validation for required columns
    if "text" not in final_mapping or final_mapping["text"] not in df.columns:
        raise ValueError(
            f"Missing required 'text' column. Available columns are: {raw_columns}. "
            f"Provide an explicit column_mapping with 'text': '<your_column>'."
        )

    if require_sentiment and ("sentiment" not in final_mapping or final_mapping["sentiment"] not in df.columns):
        raise ValueError(
            f"Missing required 'sentiment' column for training/evaluation. Available columns are: {raw_columns}. "
            f"Provide an explicit column_mapping with 'sentiment': '<your_column>' or set require_sentiment=False."
        )

    # Step 4: Standardize columns in output DataFrame
    # Invert mapping for pandas rename: {actual_col: standard_name}
    rename_dict = {actual_col: std_name for std_name, actual_col in final_mapping.items() if actual_col in df.columns}
    standardized_df = df.rename(columns=rename_dict).copy()

    # Calculate class distribution if sentiment is present
    class_dist = None
    if "sentiment" in standardized_df.columns:
        class_dist = standardized_df["sentiment"].dropna().astype(str).str.lower().value_counts().to_dict()

    report = IngestionReport(
        total_records=len(standardized_df),
        total_columns=len(standardized_df.columns),
        detected_columns=raw_columns,
        mapped_columns=final_mapping,
        encoding_used=used_encoding,
        class_distribution=class_dist,
        warnings=warnings,
    )

    return standardized_df, report
