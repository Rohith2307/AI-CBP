"""
Model Evaluation and Error Analysis Module for Customer Sentiment Analysis Framework.

Performs unbiased final evaluation strictly on the held-out test split,
generates the confusion matrix, computes macro-F1 and per-class metrics,
and performs systematic error analysis (sarcasm, negation, mixed sentiment).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import joblib
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features import DEFAULT_RANDOM_STATE, split_dataset


CANONICAL_CLASSES = ["Negative", "Neutral", "Positive"]


# Lexical heuristic indicators for categorizing failure patterns in error analysis
SARCASM_INDICATORS = {
    "thanks for", "thank you for", "great job", "awesome job", "fantastic job",
    "love waiting", "so glad", "wonderful service", "brilliant", "slowest",
    "if only", "good perspective", "appreciate being stranded",
}
NEGATION_INDICATORS = {
    "not", "never", "no", "haven't", "hasn't", "didn't", "don't", "won't",
    "couldn't", "shouldn't", "cannot", "can't", "hardly", "barely",
}
INQUIRY_INDICATORS = {
    "is it", "can i", "will flight", "how do i", "where is", "what time",
    "status of", "flight 769", "ticket", "baggage claim", "gate number",
}


@dataclass
class ErrorExample:
    """Detailed metadata for an individual misclassified comment."""
    index: int
    text: str
    true_sentiment: str
    predicted_sentiment: str
    predicted_probability: Optional[float]
    probabilities: Optional[Dict[str, float]]
    error_archetype: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Full evaluation and error analysis report on held-out test set."""
    model_name: str
    test_accuracy: float
    test_macro_f1: float
    test_macro_precision: float
    test_macro_recall: float
    test_weighted_f1: float
    confusion_matrix: List[List[int]]
    class_names: List[str]
    classification_report: Dict[str, Any]
    total_test_samples: int
    total_errors: int
    test_error_rate: float
    confusion_breakdown: Dict[str, int]
    error_archetypes_count: Dict[str, int]
    sample_errors: List[Dict[str, Any]]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def categorize_error(
    text: str,
    true_label: str,
    pred_label: str,
) -> Tuple[str, str]:
    """
    Categorize a misclassification into an interpretable error archetype
    based on lexical patterns and linguistic nuances.

    Args:
        text: Original comment text.
        true_label: Ground truth sentiment ('Positive', 'Negative', 'Neutral').
        pred_label: Model prediction.

    Returns:
        Tuple of (archetype_name, explanation).
    """
    text_lower = text.lower()

    # Archetype 1: Sarcasm / Irony
    # Customer uses positive praise words to complain about delays or lost luggage
    if true_label == "Negative" and any(ind in text_lower for ind in SARCASM_INDICATORS):
        return (
            "Sarcasm / Irony",
            "Customer used superficial praise phrasing ('thanks for', 'great job') to express frustration with a service failure.",
        )

    # Archetype 2: Subtle / Complex Negation
    # Customer employs double negations or cautious wording ('not bad', 'haven't departed yet so let's not get too high hopes')
    if any(neg in text_lower for neg in NEGATION_INDICATORS):
        return (
            "Complex Negation",
            "Multi-clause negation or conditional statement altered sentence polarity beyond bag-of-words representation.",
        )

    # Archetype 3: Neutral Inquiry Mistaken for Complaint
    # Objective customer questions containing problem terms like 'delay', 'cancel', 'gate'
    if true_label == "Neutral" and pred_label == "Negative":
        if any(inq in text_lower for inq in INQUIRY_INDICATORS) or "?" in text:
            return (
                "Ambiguous Inquiry",
                "Informational customer question regarding flight status or policy was mistaken for a grievance.",
            )

    # Archetype 4: Mixed Sentiment
    # Comment conveys both satisfaction and frustration (e.g. great crew but 4h delay)
    has_pos_lexicon = any(w in text_lower for w in ["great", "good", "love", "thank", "nice", "awesome"])
    has_neg_lexicon = any(w in text_lower for w in ["delay", "late", "cancel", "wait", "bad", "hour", "lost"])
    if has_pos_lexicon and has_neg_lexicon:
        return (
            "Mixed Sentiment",
            "Comment contains opposing positive compliments and operational complaints within the same sentence.",
        )

    # Archetype 5: Short / Slang / Implicit Expression
    if len(text.split()) <= 4:
        return (
            "Short / Low Context",
            "Very short comment with minimal lexical context for statistical unigram/bigram discrimination.",
        )

    return (
        "Semantic Ambiguity",
        f"Subtle vocabulary overlap between {true_label} sentiment context and {pred_label} training patterns.",
    )


def evaluate_model_performance(
    model: BaseEstimator,
    X_test: Any,
    y_test: pd.Series,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute unbiased performance metrics on the held-out test split.

    Args:
        model: Fitted estimator or Pipeline.
        X_test: Held-out test features (text Series or TF-IDF matrix).
        y_test: Held-out test ground truth targets.
        class_names: Target class labels (default: ['Negative', 'Neutral', 'Positive']).

    Returns:
        Dict with accuracy, macro metrics, confusion matrix, and classification report.
    """
    classes = class_names or CANONICAL_CLASSES

    y_pred = model.predict(X_test)

    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    macro_p = round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    macro_r = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    macro_f1 = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    clf_report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "y_pred": y_pred,
    }


def perform_error_analysis(
    model: BaseEstimator,
    test_df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
    class_names: Optional[List[str]] = None,
    max_examples: int = 15,
) -> Dict[str, Any]:
    """
    Perform deep error analysis on test predictions, categorizing failure patterns.

    Args:
        model: Fitted pipeline or estimator.
        test_df: Test split DataFrame with text and sentiment columns.
        text_col: Text column name.
        sentiment_col: Ground truth sentiment column name.
        class_names: Canonical class order.
        max_examples: Max representative misclassified examples to retain.

    Returns:
        Dict containing error breakdowns, archetypes count, and representative examples.
    """
    classes = class_names or CANONICAL_CLASSES
    X_test = test_df[text_col]
    y_test = test_df[sentiment_col]

    y_pred = model.predict(X_test)
    supports_probas = hasattr(model, "predict_proba")
    probas = model.predict_proba(X_test) if supports_probas else None

    # Identify errors
    misclassified_mask = (y_test != y_pred)
    total_samples = len(test_df)
    total_errors = int(misclassified_mask.sum())
    error_rate = round(total_errors / total_samples, 4) if total_samples > 0 else 0.0

    # Confusion pair breakdown
    confusion_breakdown: Dict[str, int] = {}
    archetype_counts: Dict[str, int] = {}
    error_examples: List[ErrorExample] = []

    # Get class index mapping if probabilities exist
    model_classes = list(getattr(model, "classes_", classes))

    error_indices = test_df[misclassified_mask].index

    for idx in error_indices:
        loc_pos = test_df.index.get_loc(idx)
        true_cls = str(y_test.loc[idx])
        pred_cls = str(y_pred[loc_pos])
        orig_text = str(test_df.loc[idx, "original_text"] if "original_text" in test_df.columns else test_df.loc[idx, text_col])

        # Track confusion pair
        pair_key = f"{true_cls} -> {pred_cls}"
        confusion_breakdown[pair_key] = confusion_breakdown.get(pair_key, 0) + 1

        # Categorize archetype
        archetype, explanation = categorize_error(orig_text, true_cls, pred_cls)
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

        # Probability diagnostics
        top_prob = None
        prob_dict = None
        if probas is not None:
            row_probas = probas[loc_pos]
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(model_classes, row_probas)}
            top_prob = round(float(np.max(row_probas)), 4)

        if len(error_examples) < max_examples:
            error_examples.append(ErrorExample(
                index=int(idx),
                text=orig_text,
                true_sentiment=true_cls,
                predicted_sentiment=pred_cls,
                predicted_probability=top_prob,
                probabilities=prob_dict,
                error_archetype=archetype,
                explanation=explanation,
            ))

    # Sort archetypes by frequency
    sorted_archetypes = dict(sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_confusion = dict(sorted(confusion_breakdown.items(), key=lambda x: x[1], reverse=True))

    return {
        "total_test_samples": total_samples,
        "total_errors": total_errors,
        "test_error_rate": error_rate,
        "confusion_breakdown": sorted_confusion,
        "error_archetypes_count": sorted_archetypes,
        "sample_errors": [e.to_dict() for e in error_examples],
    }


def create_confusion_matrix_chart(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate an annotated Plotly heatmap visualization of the confusion matrix.

    Args:
        cm: Confusion matrix 2D list/array.
        class_names: List of class labels (default: ['Negative', 'Neutral', 'Positive']).

    Returns:
        Plotly Figure.
    """
    classes = class_names or CANONICAL_CLASSES
    cm_array = np.array(cm)

    # Compute row-normalized percentages for recall transparency
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_array, row_sums, out=np.zeros_like(cm_array, dtype=float), where=row_sums != 0) * 100

    # Annotation text: 'Count (Recall %)'
    annot_text = []
    for r in range(len(classes)):
        row_annot = []
        for c in range(len(classes)):
            val = cm_array[r, c]
            pct = cm_norm[r, c]
            row_annot.append(f"<b>{val:,}</b><br>({pct:.1f}%)")
        row_annot.append(row_annot)
        annot_text.append([f"<b>{cm_array[r, c]:,}</b><br>({cm_norm[r, c]:.1f}%)" for c in range(len(classes))])

    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=annot_text,
        texttemplate="%{text}",
        textfont={"size": 13},
        hoverinfo="x+y+z",
        showscale=True,
    ))

    fig.update_layout(
        title="Test Set Confusion Matrix (Counts & Class Recall %)",
        xaxis_title="<b>Predicted Sentiment</b>",
        yaxis_title="<b>True Sentiment</b>",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def evaluate_and_audit_model(
    model_path: Union[str, Path] = "models/sentiment_model.pkl",
    df: Optional[pd.DataFrame] = None,
    text_col: str = "clean_text",
    sentiment_col: str = "sentiment",
    output_dir: Union[str, Path] = "models",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> EvaluationReport:
    """
    Run full end-to-end evaluation and error audit on the held-out test split.

    Args:
        model_path: Path to serialized Pipeline artifact.
        df: Input DataFrame (will split into held-out test split).
        text_col: Text column name.
        sentiment_col: Target sentiment column name.
        output_dir: Directory to save evaluation_results.json.
        random_state: Seed ensuring exact match to training splits.

    Returns:
        EvaluationReport object.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model pipeline
    model = joblib.load(model_path)
    model_name = type(model.named_steps.get("classifier", model)).__name__

    # 2. Partition dataset to get strictly held-out test set
    if df is None:
        raise ValueError("DataFrame required to extract held-out test split.")

    actual_col = text_col if text_col in df.columns else "text"
    split = split_dataset(
        df=df,
        text_col=actual_col,
        sentiment_col=sentiment_col,
        test_size=0.15,
        val_size=0.15,
        random_state=random_state,
    )

    test_df = split.test_df.copy()

    # 3. Compute Test Performance
    perf = evaluate_model_performance(model, test_df[actual_col], test_df[sentiment_col])

    # 4. Perform Detailed Error Analysis
    err_audit = perform_error_analysis(model, test_df, text_col=actual_col, sentiment_col=sentiment_col)

    limitations = [
        "Sarcasm & Irony: Praise keywords ('thanks for cancelling', 'great job losing my bag') can deceive n-gram TF-IDF representations.",
        "Subtle & Double Negation: Long-distance negations ('we haven't departed yet so let's not get too high hopes, but on schedule so far') alter polarity in ways bag-of-words models cannot always resolve.",
        "Mixed Sentiments: Comments expressing simultaneous praise and grievance are mapped to a single discrete label without aspect-level granularity.",
        "Neutral Inquiries: Objective questions containing airline problem keywords ('flight delayed', 'cancel flight') are occasionally misclassified as complaints.",
        "Short Slang & Typos: Social media abbreviations ('smh', 'fml') and typos degrade TF-IDF matching if unigrams are out of vocabulary.",
    ]

    report = EvaluationReport(
        model_name=model_name,
        test_accuracy=perf["accuracy"],
        test_macro_f1=perf["macro_f1"],
        test_macro_precision=perf["macro_precision"],
        test_macro_recall=perf["macro_recall"],
        test_weighted_f1=perf["weighted_f1"],
        confusion_matrix=perf["confusion_matrix"],
        class_names=CANONICAL_CLASSES,
        classification_report=perf["classification_report"],
        total_test_samples=err_audit["total_test_samples"],
        total_errors=err_audit["total_errors"],
        test_error_rate=err_audit["test_error_rate"],
        confusion_breakdown=err_audit["confusion_breakdown"],
        error_archetypes_count=err_audit["error_archetypes_count"],
        sample_errors=err_audit["sample_errors"],
        limitations=limitations,
    )

    # Persist JSON report
    report_file = out_dir / "evaluation_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return report
