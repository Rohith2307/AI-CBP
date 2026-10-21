"""
Data Validation Module for Customer Sentiment Analysis Framework.

Performs schema verification, text integrity checks, duplicate detection,
sentiment label normalization, and transparent data-quality reporting
without silent data deletion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Canonical sentiment classes
CANONICAL_SENTIMENTS = {"Positive", "Negative", "Neutral"}

# Mapping of common alias variations to canonical classes
SENTIMENT_NORMALIZATION_MAP: Dict[str, str] = {
    # Positive aliases
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "1.0": "Positive",
    "good": "Positive",
    "great": "Positive",
    # Negative aliases
    "negative": "Negative",
    "neg": "Negative",
    "-1": "Negative",
    "-1.0": "Negative",
    "bad": "Negative",
    "poor": "Negative",
    # Neutral aliases
    "neutral": "Neutral",
    "neu": "Neutral",
    "0": "Neutral",
    "0.0": "Neutral",
    "mixed": "Neutral",
    "none": "Neutral",
}


@dataclass
class ValidationReport:
    """Comprehensive data quality and validation summary."""
    total_records: int
    valid_records: int
    missing_text_count: int
    empty_text_count: int
    duplicate_count: int
    invalid_sentiment_count: int
    class_distribution: Dict[str, int]
    avg_char_length: float
    avg_word_count: float
    unique_products: Optional[int] = None
    unique_sources: Optional[int] = None
    removal_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for UI and serialization."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "missing_text_count": self.missing_text_count,
            "empty_text_count": self.empty_text_count,
            "duplicate_count": self.duplicate_count,
            "invalid_sentiment_count": self.invalid_sentiment_count,
            "class_distribution": self.class_distribution,
            "avg_char_length": round(self.avg_char_length, 2),
            "avg_word_count": round(self.avg_word_count, 2),
            "unique_products": self.unique_products,
            "unique_sources": self.unique_sources,
            "removal_summary": self.removal_summary,
            "warnings": self.warnings,
        }


@dataclass
class ValidationResult:
    """Container holding validated dataset, flagged/invalid dataset, and report."""
    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame
    report: ValidationReport


def normalize_sentiment_label(raw_label: Any) -> Optional[str]:
    """
    Standardize a raw sentiment value into 'Positive', 'Negative', or 'Neutral'.

    Args:
        raw_label: Raw label string, number, or NaN.

    Returns:
        Standardized label ('Positive', 'Negative', 'Neutral') or None if unclassifiable.
    """
    if pd.isna(raw_label):
        return None

    norm_str = str(raw_label).strip().lower()
    return SENTIMENT_NORMALIZATION_MAP.get(norm_str, None)


def validate_dataset(
    df: pd.DataFrame,
    require_sentiment: bool = True,
    drop_duplicates: bool = True,
) -> ValidationResult:
    """
    Validate customer comment records, log invalid/removed records with reasons,
    standardize sentiment labels, and compute quality metrics.

    Args:
        df: Input DataFrame (must contain 'text', optionally 'sentiment', 'date', 'product', etc.).
        require_sentiment: Whether valid records must have a recognized sentiment label.
        drop_duplicates: Whether duplicate text records should be flagged and excluded from valid_df.

    Returns:
        ValidationResult containing (valid_df, invalid_df, report).

    Raises:
        ValueError: If 'text' column is missing entirely from the DataFrame.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column for validation.")

    # Work on a copy and preserve original text explicitly
    working_df = df.copy()
    if "original_text" not in working_df.columns:
        working_df["original_text"] = working_df["text"]

    invalid_records: List[Dict[str, Any]] = []
    valid_indices: List[Any] = []

    # Track distinct issue counts
    missing_text_indices = set()
    empty_text_indices = set()
    invalid_sentiment_indices = set()
    duplicate_indices = set()

    # Step 1: Check text integrity
    for idx, row in working_df.iterrows():
        raw_val = row["text"]
        if pd.isna(raw_val):
            missing_text_indices.add(idx)
            rec = row.to_dict()
            rec["removal_reason"] = "missing_text"
            rec["original_index"] = idx
            invalid_records.append(rec)
            continue

        text_str = str(raw_val).strip()
        if len(text_str) == 0:
            empty_text_indices.add(idx)
            rec = row.to_dict()
            rec["removal_reason"] = "empty_text"
            rec["original_index"] = idx
            invalid_records.append(rec)
            continue

        # Step 2: Check sentiment label if required
        if require_sentiment:
            if "sentiment" not in row or pd.isna(row["sentiment"]):
                invalid_sentiment_indices.add(idx)
                rec = row.to_dict()
                rec["removal_reason"] = "missing_sentiment"
                rec["original_index"] = idx
                invalid_records.append(rec)
                continue

            normalized_sent = normalize_sentiment_label(row["sentiment"])
            if normalized_sent is None:
                invalid_sentiment_indices.add(idx)
                rec = row.to_dict()
                rec["removal_reason"] = f"unrecognized_sentiment_label: {row['sentiment']}"
                rec["original_index"] = idx
                invalid_records.append(rec)
                continue
            else:
                working_df.at[idx, "sentiment"] = normalized_sent
        elif "sentiment" in working_df.columns and not pd.isna(row["sentiment"]):
            # Normalize if present even if not strictly required
            normalized_sent = normalize_sentiment_label(row["sentiment"])
            if normalized_sent:
                working_df.at[idx, "sentiment"] = normalized_sent

        valid_indices.append(idx)

    # Filter candidates before duplicate check
    candidate_df = working_df.loc[valid_indices].copy()
    candidate_df["text"] = candidate_df["text"].astype(str)

    # Step 3: Check duplicates
    if drop_duplicates and not candidate_df.empty:
        is_dup = candidate_df.duplicated(subset=["text"], keep="first")
        dup_mask_indices = candidate_df[is_dup].index
        for dup_idx in dup_mask_indices:
            duplicate_indices.add(dup_idx)
            rec = candidate_df.loc[dup_idx].to_dict()
            rec["removal_reason"] = "duplicate_text"
            rec["original_index"] = dup_idx
            invalid_records.append(rec)

        valid_df = candidate_df[~is_dup].copy()
    else:
        valid_df = candidate_df

    # Step 4: Metadata cleanup and text length calculations
    if not valid_df.empty:
        valid_df["char_length"] = valid_df["text"].str.len()
        valid_df["word_count"] = valid_df["text"].str.split().str.len()

        # Parse date if present without crashing on malformed dates
        if "date" in valid_df.columns:
            valid_df["date_parsed"] = pd.to_datetime(valid_df["date"], errors="coerce", format="mixed")

        avg_char_length = float(valid_df["char_length"].mean())
        avg_word_count = float(valid_df["word_count"].mean())
    else:
        avg_char_length = 0.0
        avg_word_count = 0.0

    # Step 5: Build reports and summaries
    invalid_df = pd.DataFrame(invalid_records) if invalid_records else pd.DataFrame(columns=list(df.columns) + ["removal_reason", "original_index"])

    class_distribution = {}
    if "sentiment" in valid_df.columns and not valid_df.empty:
        class_distribution = valid_df["sentiment"].value_counts().to_dict()

    unique_products = int(valid_df["product"].nunique()) if "product" in valid_df.columns and not valid_df.empty else None
    unique_sources = int(valid_df["source"].nunique()) if "source" in valid_df.columns and not valid_df.empty else None

    removal_summary = {
        "missing_text": len(missing_text_indices),
        "empty_text": len(empty_text_indices),
        "invalid_sentiment": len(invalid_sentiment_indices),
        "duplicate_text": len(duplicate_indices),
        "total_removed": len(invalid_records),
    }

    warnings = []
    if len(invalid_records) > 0:
        warnings.append(
            f"Validation removed {len(invalid_records)} invalid/duplicate records. "
            f"Details preserved in invalid_df and removal_summary."
        )

    # Check for severe class imbalance if classes are present
    if class_distribution and len(class_distribution) > 1:
        max_c = max(class_distribution.values())
        min_c = min(class_distribution.values())
        if min_c > 0 and (max_c / min_c) > 5.0:
            warnings.append(
                f"Severe class imbalance detected: majority class ({max_c}) is over 5x "
                f"the minority class ({min_c}). Stratification recommended."
            )

    report = ValidationReport(
        total_records=len(df),
        valid_records=len(valid_df),
        missing_text_count=len(missing_text_indices),
        empty_text_count=len(empty_text_indices),
        duplicate_count=len(duplicate_indices),
        invalid_sentiment_count=len(invalid_sentiment_indices),
        class_distribution=class_distribution,
        avg_char_length=avg_char_length,
        avg_word_count=avg_word_count,
        unique_products=unique_products,
        unique_sources=unique_sources,
        removal_summary=removal_summary,
        warnings=warnings,
    )

    return ValidationResult(valid_df=valid_df, invalid_df=invalid_df, report=report)


def save_validated_dataset(valid_df: pd.DataFrame, target_path: Union[str, Path]) -> Path:
    """
    Save the validated dataset to target CSV path for downstream pipeline stages.

    Args:
        valid_df: Validated DataFrame.
        target_path: Path where CSV should be stored.

    Returns:
        Path of the saved file.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(path, index=False, encoding="utf-8")
    return path
