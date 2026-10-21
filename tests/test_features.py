"""
Unit tests for src/features.py (Stage 6: TF-IDF Feature Extraction & Dataset Splitting).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from src.features import (
    split_dataset,
    SentimentTfidfExtractor,
    extract_features,
    SplitResult,
    FeaturePipelineResult,
)


SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "validated_sample.csv"


@pytest.fixture
def sample_df():
    """Load validated sample dataset."""
    assert SAMPLE_PATH.exists()
    return pd.read_csv(SAMPLE_PATH)


def test_split_dataset_integrity(sample_df):
    """Verify train/val/test split partition sizes, zero overlap, and stratification."""
    result = split_dataset(sample_df, test_size=0.15, val_size=0.15, random_state=42)

    assert isinstance(result, SplitResult)
    total = len(sample_df)
    n_train = len(result.X_train)
    n_val = len(result.X_val)
    n_test = len(result.X_test)

    # 1. Total samples conserved exactly
    assert n_train + n_val + n_test == total

    # 2. Ratios match approximately 70 / 15 / 15
    assert 0.68 <= (n_train / total) <= 0.72
    assert 0.14 <= (n_val / total) <= 0.16
    assert 0.14 <= (n_test / total) <= 0.16

    # 3. Disjoint partitions (zero index leakage across splits)
    train_idx = set(result.train_df.index)
    val_idx = set(result.val_df.index)
    test_idx = set(result.test_df.index)

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0

    # 4. Stratification: all 3 classes preserved in every split
    for dist in [result.summary.train_class_distribution,
                 result.summary.val_class_distribution,
                 result.summary.test_class_distribution]:
        assert "Positive" in dist
        assert "Negative" in dist
        assert "Neutral" in dist
        assert all(count > 0 for count in dist.values())


def test_split_dataset_reproducibility(sample_df):
    """Verify that using the same random seed produces identical splits."""
    split_1 = split_dataset(sample_df, random_state=42)
    split_2 = split_dataset(sample_df, random_state=42)
    split_diff = split_dataset(sample_df, random_state=99)

    assert split_1.train_df.index.tolist() == split_2.train_df.index.tolist()
    assert split_1.val_df.index.tolist() == split_2.val_df.index.tolist()
    assert split_1.test_df.index.tolist() == split_2.test_df.index.tolist()

    # Different seed should yield different partition ordering
    assert split_1.train_df.index.tolist() != split_diff.train_df.index.tolist()


def test_small_dataset_split_fallback():
    """Verify small dataset handling falls back gracefully to 80/20 train/test."""
    tiny_df = pd.DataFrame({
        "text": [f"comment {i}" for i in range(12)],
        "sentiment": ["Positive", "Negative", "Neutral"] * 4,
    })

    result = split_dataset(tiny_df, random_state=42)
    assert len(result.X_train) == 9  # ~80%
    assert len(result.X_test) == 3   # ~20%
    assert "80/20" in result.summary.strategy_notes

    # Dataset with < 6 rows should raise descriptive error
    with pytest.raises(ValueError, match="Dataset too small to split"):
        split_dataset(tiny_df.head(4))


def test_tfidf_extractor_unigrams_and_bigrams():
    """Verify unigrams and bigrams are extracted into vocabulary."""
    corpus = pd.Series([
        "flight was delayed and customer service was terrible",
        "great flight and wonderful customer service",
        "delayed flight again with terrible service",
        "excellent flight wonderful crew",
    ])

    extractor = SentimentTfidfExtractor(ngram_range=(1, 2), min_df=1)
    X_vec = extractor.fit_transform(corpus)

    assert isinstance(X_vec, csr_matrix)
    assert extractor.is_fitted

    vocab_meta = extractor.inspect_vocabulary()
    assert vocab_meta["vocab_size"] > 0
    assert vocab_meta["unigram_count"] > 0
    assert vocab_meta["bigram_count"] > 0

    feature_names = extractor.get_feature_names()
    assert "flight" in feature_names
    assert "customer service" in feature_names or "delayed flight" in feature_names


def test_strict_anti_leakage():
    """
    CRITICAL: Verify that the vocabulary and IDF weights are determined
    SOLEY by the training split. Unseen words in test data must not leak.
    """
    train_texts = pd.Series([
        "good service fast flight",
        "poor service delayed flight",
    ])
    test_texts = pd.Series([
        "exceptional onboard breakfast champagne",  # completely novel words
    ])

    extractor = SentimentTfidfExtractor(min_df=1)
    extractor.fit(train_texts)

    train_vocab = extractor.get_feature_names()
    assert "good" in train_vocab
    assert "flight" in train_vocab

    # Novel test tokens must NOT be in the vocabulary
    assert "champagne" not in train_vocab
    assert "breakfast" not in train_vocab
    assert "exceptional" not in train_vocab

    # Transforming unseen text should succeed without error or vocabulary alteration
    test_vec = extractor.transform(test_texts)
    assert test_vec.shape == (1, len(train_vocab))
    assert test_vec.sum() == 0.0  # none of the novel words match, vector is all-zeros cleanly


def test_not_fitted_error():
    """Verify NotFittedError is raised when transforming or saving before fit."""
    extractor = SentimentTfidfExtractor()
    assert not extractor.is_fitted

    with pytest.raises(NotFittedError, match="must be fitted before transforming"):
        extractor.transform(pd.Series(["some text"]))

    with pytest.raises(NotFittedError, match="Cannot save an unfitted extractor"):
        extractor.save("models/vectorizer.pkl")


def test_transformation_of_unseen_comments(sample_df):
    """Verify transforming novel unseen comments produces valid non-zero sparse vectors."""
    split = split_dataset(sample_df, random_state=42)
    extractor = SentimentTfidfExtractor(ngram_range=(1, 2), min_df=2)
    extractor.fit(split.X_train)

    unseen_comments = pd.Series([
        "Flight was delayed by four hours, horrible customer service!",
        "Thank you so much for the amazing flight and wonderful crew ❤️",
        "What is the baggage allowance for international travel?",
    ])

    vec = extractor.transform(unseen_comments)
    assert isinstance(vec, csr_matrix)
    assert vec.shape[0] == 3
    assert vec.shape[1] == len(extractor.get_feature_names())

    # At least known common terms (flight, delayed, customer, service, thank, crew) should produce positive weights
    assert vec[0].sum() > 0.0
    assert vec[1].sum() > 0.0


def test_extractor_save_and_load(sample_df, tmp_path):
    """Verify serialization and deserialization of the fitted vectorizer."""
    split = split_dataset(sample_df, random_state=42)
    extractor = SentimentTfidfExtractor(ngram_range=(1, 2), min_df=2)
    extractor.fit(split.X_train)

    artifact_path = tmp_path / "vectorizer.pkl"
    saved_path = extractor.save(artifact_path)
    assert saved_path.exists()

    loaded_extractor = SentimentTfidfExtractor.load(saved_path)
    assert loaded_extractor.is_fitted
    assert loaded_extractor.get_feature_names() == extractor.get_feature_names()

    # Compare transformation outputs
    sample_text = pd.Series(["Flight delayed again!"])
    orig_vec = extractor.transform(sample_text)
    loaded_vec = loaded_extractor.transform(sample_text)
    assert np.allclose(orig_vec.toarray(), loaded_vec.toarray())


def test_extract_features_pipeline(sample_df):
    """Verify end-to-end feature extraction pipeline function."""
    result = extract_features(sample_df, random_state=42)

    assert isinstance(result, FeaturePipelineResult)
    assert result.X_train_vec.shape[0] == result.split_result.summary.train_samples
    assert result.X_val_vec.shape[0] == result.split_result.summary.val_samples
    assert result.X_test_vec.shape[0] == result.split_result.summary.test_samples

    # All feature matrices must share the exact same number of feature columns
    n_features = result.X_train_vec.shape[1]
    assert result.X_val_vec.shape[1] == n_features
    assert result.X_test_vec.shape[1] == n_features
