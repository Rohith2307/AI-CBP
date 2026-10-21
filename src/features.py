"""
Feature Extraction Module for Customer Sentiment Analysis Framework.

Implements reproducible stratified train/validation/test dataset splitting
and anti-leakage TF-IDF feature extraction using scikit-learn.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


DEFAULT_RANDOM_STATE = 42
DEFAULT_NGRAM_RANGE = (1, 2)  # Unigram + Bigram
DEFAULT_MAX_FEATURES = 5000   # Sufficient for high discriminability while fast
DEFAULT_MIN_DF = 2            # Exclude single-occurrence terms to prevent noise
DEFAULT_MAX_DF = 0.95         # Exclude terms appearing in >95% of documents


@dataclass
class SplitSummary:
    """Summary of train/validation/test partition sizes and stratification."""
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    stratified: bool
    train_class_distribution: Dict[str, int]
    val_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    strategy_notes: str


@dataclass
class SplitResult:
    """Container for split data arrays, DataFrames, and summary metrics."""
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    summary: SplitSummary


def split_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitResult:
    """
    Partition dataset into reproducible, stratified train (approx 70%),
    validation (approx 15%), and test (approx 15%) sets.

    Handles small datasets by gracefully adjusting split ratios when needed.

    Args:
        df: Validated DataFrame.
        text_col: Column containing text features (clean_text or text).
        sentiment_col: Column containing target sentiment labels.
        test_size: Proportion of dataset allocated to the test set.
        val_size: Proportion of dataset allocated to the validation set.
        random_state: Seed for reproducible random splits.

    Returns:
        SplitResult dataclass holding train/val/test splits and summary.
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame.")
    if sentiment_col not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_col}' not found in DataFrame.")

    total = len(df)
    if total < 6:
        raise ValueError(f"Dataset too small to split ({total} records, minimum 6 required).")

    # Check minimum class representation for stratification
    class_counts = df[sentiment_col].value_counts()
    min_class_count = class_counts.min()

    # Determine whether stratification is viable
    can_stratify = min_class_count >= 3
    stratify_target = df[sentiment_col] if can_stratify else None

    # Handle tiny datasets (< 20 records) with a simplified 80/20 train/test split
    if total < 20 or min_class_count < 3:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            stratify=stratify_target,
            random_state=random_state,
        )
        val_df = test_df.copy()  # For tiny datasets, val mirrors test for evaluation
        notes = "Dataset too small for 3-way split; used 80/20 train/test partition."
    else:
        # Step 1: Split out test set (15%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state,
        )

        # Step 2: Split remaining 85% into train (70% overall) and val (15% overall)
        # Ratio of val inside train_val: 0.15 / (1.0 - test_size) = 0.15 / 0.85 = 0.17647
        val_relative_size = val_size / (1.0 - test_size)
        stratify_tv = train_val_df[sentiment_col] if can_stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            stratify=stratify_tv,
            random_state=random_state,
        )
        notes = "Standard stratified 70% Train, 15% Validation, 15% Test partition."

    # Extract Series
    X_train = train_df[text_col]
    X_val = val_df[text_col]
    X_test = test_df[text_col]

    y_train = train_df[sentiment_col]
    y_val = val_df[sentiment_col]
    y_test = test_df[sentiment_col]

    summary = SplitSummary(
        total_samples=total,
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
        train_ratio=round(len(train_df) / total, 4),
        val_ratio=round(len(val_df) / total, 4),
        test_ratio=round(len(test_df) / total, 4),
        random_state=random_state,
        stratified=can_stratify,
        train_class_distribution=y_train.value_counts().to_dict(),
        val_class_distribution=y_val.value_counts().to_dict(),
        test_class_distribution=y_test.value_counts().to_dict(),
        strategy_notes=notes,
    )

    return SplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        summary=summary,
    )


class SentimentTfidfExtractor:
    """
    TF-IDF feature extractor configured for sentiment analysis.

    Ensures zero data leakage by fitting solely on training splits,
    then projecting validation and test data into the learned feature space.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
        max_features: Optional[int] = DEFAULT_MAX_FEATURES,
        min_df: Union[int, float] = DEFAULT_MIN_DF,
        max_df: Union[int, float] = DEFAULT_MAX_DF,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check whether the vectorizer has been fitted."""
        return self._is_fitted

    def fit(self, X_train: pd.Series) -> "SentimentTfidfExtractor":
        """
        Fit vectorizer vocabulary and IDF weights ONLY on training text.

        Args:
            X_train: Training text series.

        Returns:
            self
        """
        # Ensure strings and no NaNs
        clean_X = X_train.fillna("").astype(str)

        # Handle very small corpora where min_df > num_samples or max_df filters out shared terms
        if len(clean_X) < self.min_df:
            self.vectorizer.set_params(min_df=1)
        if len(clean_X) < 20 and isinstance(self.max_df, float) and self.max_df < 1.0:
            self.vectorizer.set_params(max_df=1.0)

        self.vectorizer.fit(clean_X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.Series) -> csr_matrix:
        """
        Transform unseen or validation text into TF-IDF vector matrix.
        Does NOT refit or modify vocabulary/weights.

        Args:
            X: Text series to transform.

        Returns:
            Scipy sparse CSR matrix.
        """
        if not self._is_fitted:
            raise NotFittedError("SentimentTfidfExtractor must be fitted before transforming data.")

        clean_X = X.fillna("").astype(str)
        return self.vectorizer.transform(clean_X)

    def fit_transform(self, X_train: pd.Series) -> csr_matrix:
        """
        Fit on training text and transform it in a single step.

        Args:
            X_train: Training text series.

        Returns:
            TF-IDF sparse feature matrix for training data.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def get_feature_names(self) -> List[str]:
        """Return list of learned feature strings."""
        if not self._is_fitted:
            raise NotFittedError("Extractor is not fitted yet.")
        return self.vectorizer.get_feature_names_out().tolist()

    def inspect_vocabulary(self) -> Dict[str, Any]:
        """Return metadata summary of learned feature vocabulary."""
        if not self._is_fitted:
            raise NotFittedError("Extractor is not fitted yet.")

        feature_names = self.get_feature_names()
        unigrams = [f for f in feature_names if " " not in f]
        bigrams = [f for f in feature_names if " " in f]

        return {
            "vocab_size": len(feature_names),
            "unigram_count": len(unigrams),
            "bigram_count": len(bigrams),
            "sample_unigrams": unigrams[:10],
            "sample_bigrams": bigrams[:10],
            "config": {
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "sublinear_tf": self.sublinear_tf,
            },
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize fitted extractor artifact using joblib."""
        if not self._is_fitted:
            raise NotFittedError("Cannot save an unfitted extractor.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SentimentTfidfExtractor":
        """Load serialized extractor from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found: {path}")
        extractor = joblib.load(path)
        if not isinstance(extractor, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance.")
        return extractor


@dataclass
class FeaturePipelineResult:
    """Result of feature extraction containing matrices, extractor, and split result."""
    X_train_vec: csr_matrix
    X_val_vec: csr_matrix
    X_test_vec: csr_matrix
    extractor: SentimentTfidfExtractor
    split_result: SplitResult


def extract_features(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
    extractor: Optional[SentimentTfidfExtractor] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FeaturePipelineResult:
    """
    End-to-end feature extraction pipeline:
    1. Stratified split (70% train, 15% val, 15% test)
    2. Fit extractor ONLY on training text
    3. Transform train, validation, and test sets without data leakage

    Args:
        df: Validated DataFrame.
        text_col: Source text column.
        sentiment_col: Sentiment class target column.
        extractor: Optional pre-configured SentimentTfidfExtractor.
        test_size: Test proportion.
        val_size: Validation proportion.
        random_state: Random state seed.

    Returns:
        FeaturePipelineResult container.
    """
    split = split_dataset(
        df=df,
        text_col=text_col,
        sentiment_col=sentiment_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    tfidf_extractor = extractor or SentimentTfidfExtractor()

    # STRICT ANTI-LEAKAGE: Fit solely on X_train
    X_train_vec = tfidf_extractor.fit_transform(split.X_train)

    # Transform val and test using fitted vectorizer
    X_val_vec = tfidf_extractor.transform(split.X_val)
    X_test_vec = tfidf_extractor.transform(split.X_test)

    return FeaturePipelineResult(
        X_train_vec=X_train_vec,
        X_val_vec=X_val_vec,
        X_test_vec=X_test_vec,
        extractor=tfidf_extractor,
        split_result=split,
    )
