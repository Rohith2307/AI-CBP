"""
Sentiment Prediction Module for Customer Sentiment Analysis Framework.

Provides high-throughput inference for single comments and batch inputs (lists,
Series, DataFrames). Extracts predicted probabilities when supported by the model,
preserves original text, and handles empty or malformed inputs gracefully.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import SentimentTfidfExtractor
from src.preprocessing import TextPreprocessor


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.pkl"
DEFAULT_VECTORIZER_PATH = Path(__file__).resolve().parent.parent / "models" / "vectorizer.pkl"


@dataclass
class PredictionResult:
    """Inference output for a single comment."""
    original_text: str
    clean_text: str
    predicted_sentiment: str
    predicted_probability: Optional[float]
    class_probabilities: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert single prediction to dictionary."""
        return asdict(self)


@dataclass
class BatchPredictionResult:
    """Inference output for a batch of comments."""
    dataframe: pd.DataFrame
    total_records: int
    sentiment_counts: Dict[str, int]
    predictions: List[PredictionResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch summary to dictionary."""
        return {
            "total_records": self.total_records,
            "sentiment_counts": self.sentiment_counts,
            "predictions": [p.to_dict() for p in self.predictions],
        }


class SentimentPredictor:
    """
    Unified Inference Engine for Customer Sentiment Analysis.

    Attributes:
        model: Loaded scikit-learn Pipeline or estimator.
        vectorizer: Optional standalone vectorizer if model is not a self-contained Pipeline.
        preprocessor: TextPreprocessor instance for normalizing noisy input text.
        classes: Ordered list of sentiment class labels.
        supports_probabilities: Whether the underlying model provides predict_proba.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        vectorizer_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        vectorizer: Optional[Any] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.preprocessor = preprocessor or TextPreprocessor()

        # Step 1: Initialize model
        if model is not None:
            self.model = model
        else:
            path = Path(model_path or DEFAULT_MODEL_PATH)
            if not path.exists():
                raise FileNotFoundError(
                    f"Trained model artifact not found at: {path}. "
                    f"Please run model training (Stage 7) first or specify a valid model_path."
                )
            try:
                self.model = joblib.load(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model artifact from {path}: {e}") from e

        # Step 2: Determine pipeline structure and vectorizer
        if isinstance(self.model, Pipeline) and "tfidf" in self.model.named_steps:
            self.is_pipeline = True
            self.vectorizer = None
        else:
            self.is_pipeline = False
            if vectorizer is not None:
                self.vectorizer = vectorizer
            elif vectorizer_path or DEFAULT_VECTORIZER_PATH.exists():
                v_path = Path(vectorizer_path or DEFAULT_VECTORIZER_PATH)
                if not v_path.exists():
                    raise FileNotFoundError(f"Vectorizer artifact not found at: {v_path}")
                try:
                    loaded_vec = joblib.load(v_path)
                    # If loaded object is SentimentTfidfExtractor, extract internal vectorizer
                    self.vectorizer = loaded_vec.vectorizer if hasattr(loaded_vec, "vectorizer") else loaded_vec
                except Exception as e:
                    raise RuntimeError(f"Failed to load vectorizer artifact from {v_path}: {e}") from e
            else:
                raise ValueError("Model is not an end-to-end Pipeline and no vectorizer was provided.")

        # Step 3: Inspect probability support and class names
        clf_estimator = self.model.named_steps.get("classifier", self.model) if isinstance(self.model, Pipeline) else self.model
        self.supports_probabilities = hasattr(clf_estimator, "predict_proba")
        self.classes = list(getattr(clf_estimator, "classes_", ["Negative", "Neutral", "Positive"]))

    def predict_single(self, text: Optional[str]) -> PredictionResult:
        """
        Classify sentiment for a single comment.

        Args:
            text: Raw comment text or None/empty string.

        Returns:
            PredictionResult dataclass with original text, clean text, predicted sentiment,
            and predicted probability (if supported).
        """
        # Handle empty, None, or whitespace-only inputs gracefully
        if text is None or pd.isna(text) or str(text).strip() == "":
            return PredictionResult(
                original_text="" if text is None else str(text),
                clean_text="",
                predicted_sentiment="Neutral",
                predicted_probability=None,
                class_probabilities=None,
            )

        orig_str = str(text)
        cleaned_str = self.preprocessor.process(orig_str)

        # If preprocessing strips all tokens (e.g. symbols only like "???"), fallback gracefully
        if not cleaned_str.strip():
            return PredictionResult(
                original_text=orig_str,
                clean_text="",
                predicted_sentiment="Neutral",
                predicted_probability=None,
                class_probabilities=None,
            )

        # Inference
        if self.is_pipeline:
            pred_label = str(self.model.predict([cleaned_str])[0])
            if self.supports_probabilities:
                raw_probas = self.model.predict_proba([cleaned_str])[0]
                prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes, raw_probas)}
                max_prob = round(float(np.max(raw_probas)), 4)
            else:
                prob_dict = None
                max_prob = None
        else:
            vec = self.vectorizer.transform([cleaned_str])
            pred_label = str(self.model.predict(vec)[0])
            if self.supports_probabilities:
                raw_probas = self.model.predict_proba(vec)[0]
                prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes, raw_probas)}
                max_prob = round(float(np.max(raw_probas)), 4)
            else:
                prob_dict = None
                max_prob = None

        return PredictionResult(
            original_text=orig_str,
            clean_text=cleaned_str,
            predicted_sentiment=pred_label,
            predicted_probability=max_prob,
            class_probabilities=prob_dict,
        )

    def predict_batch(
        self,
        data: Union[List[str], pd.Series, pd.DataFrame],
        text_col: str = "text",
    ) -> BatchPredictionResult:
        """
        Classify a batch of comments. Supports Python lists, pandas Series,
        and pandas DataFrames with metadata preservation.

        Args:
            data: Input collection of texts or DataFrame.
            text_col: Name of the text column if data is a DataFrame.

        Returns:
            BatchPredictionResult containing enhanced DataFrame and individual predictions.
        """
        # Handle empty batch input
        if (isinstance(data, (list, tuple)) and len(data) == 0) or (isinstance(data, (pd.Series, pd.DataFrame)) and data.empty):
            empty_df = pd.DataFrame(columns=[
                "original_text", "clean_text", "predicted_sentiment", "predicted_probability"
            ])
            return BatchPredictionResult(
                dataframe=empty_df,
                total_records=0,
                sentiment_counts={},
                predictions=[],
            )

        # Normalize data into working DataFrame
        if isinstance(data, pd.DataFrame):
            actual_col = text_col if text_col in data.columns else ("clean_text" if "clean_text" in data.columns else None)
            if actual_col is None:
                raise ValueError(f"Text column '{text_col}' not found in DataFrame columns: {list(data.columns)}")
            df = data.copy()
            if "original_text" not in df.columns:
                df["original_text"] = df[actual_col]
            raw_texts = df[actual_col]
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({"original_text": data.values})
            raw_texts = data
        else:
            df = pd.DataFrame({"original_text": list(data)})
            raw_texts = df["original_text"]

        # Step 1: Preprocess texts
        cleaned_texts = [self.preprocessor.process(t) for t in raw_texts]
        df["clean_text"] = cleaned_texts

        # Replace completely empty cleaned texts with fallback token to avoid vectorizer empty errors
        safe_texts = [c if c.strip() else "neutral" for c in cleaned_texts]

        # Step 2: Batch Inference
        if self.is_pipeline:
            preds = self.model.predict(safe_texts)
            if self.supports_probabilities:
                probas = self.model.predict_proba(safe_texts)
            else:
                probas = None
        else:
            vecs = self.vectorizer.transform(safe_texts)
            preds = self.model.predict(vecs)
            if self.supports_probabilities:
                probas = self.model.predict_proba(vecs)
            else:
                probas = None

        df["predicted_sentiment"] = preds

        # Step 3: Populate probabilities if supported
        individual_results: List[PredictionResult] = []
        max_probas_list = []

        for i, (orig, clean, pred) in enumerate(zip(df["original_text"], df["clean_text"], preds)):
            # If original input was empty or whitespace, enforce Neutral and None probability
            if not str(orig).strip():
                pred = "Neutral"
                df.at[df.index[i], "predicted_sentiment"] = "Neutral"
                max_prob = None
                prob_dict = None
            elif probas is not None:
                row_prob = probas[i]
                max_prob = round(float(np.max(row_prob)), 4)
                prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes, row_prob)}
            else:
                max_prob = None
                prob_dict = None

            max_probas_list.append(max_prob)
            individual_results.append(PredictionResult(
                original_text=str(orig),
                clean_text=clean,
                predicted_sentiment=pred,
                predicted_probability=max_prob,
                class_probabilities=prob_dict,
            ))

        df["predicted_probability"] = max_probas_list

        counts = df["predicted_sentiment"].value_counts().to_dict()

        return BatchPredictionResult(
            dataframe=df,
            total_records=len(df),
            sentiment_counts=counts,
            predictions=individual_results,
        )


def load_predictor(
    model_path: Optional[Union[str, Path]] = None,
    vectorizer_path: Optional[Union[str, Path]] = None,
) -> SentimentPredictor:
    """
    Convenience factory to instantiate a ready-to-use SentimentPredictor.

    Args:
        model_path: Path to serialized Pipeline.
        vectorizer_path: Path to serialized vectorizer.

    Returns:
        SentimentPredictor instance.
    """
    return SentimentPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
