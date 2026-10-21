"""
Model Training and Comparison Module for Customer Sentiment Analysis Framework.

Trains and evaluates Multinomial Naive Bayes, Logistic Regression, and Linear SVM
using TF-IDF features. Evaluates performance using Accuracy, Macro-F1, Precision,
Recall, and Cross-Validation, preventing data leakage and persisting results.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import DEFAULT_RANDOM_STATE, SentimentTfidfExtractor, split_dataset


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained classifier."""
    model_name: str
    hyperparameters: Dict[str, Any]
    train_time_sec: float
    infer_time_ms: float
    val_accuracy: float
    val_macro_precision: float
    val_macro_recall: float
    val_macro_f1: float
    val_weighted_f1: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    class_metrics: Dict[str, Dict[str, float]]
    supports_probabilities: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ExperimentComparison:
    """Comparison table and artifacts from candidate model training."""
    metrics_list: List[ModelMetrics]
    comparison_table: pd.DataFrame
    best_model_name: str
    best_macro_f1: float
    selection_rationale: str
    saved_artifacts: Dict[str, str] = field(default_factory=dict)


def get_default_candidate_models(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Define the three core candidate sentiment models specified in the build-plan:
    1. Multinomial Naive Bayes (Fast probabilistic baseline)
    2. Logistic Regression (Primary linear probabilistic classifier with class weighting)
    3. Linear SVM (LinearSVC - Maximum margin linear hyperplane)

    Args:
        random_state: Seed for reproducible fitting.

    Returns:
        Dict mapping model name to instantiated scikit-learn estimator.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def evaluate_model(
    model: BaseEstimator,
    X_train: Any,
    y_train: pd.Series,
    X_val: Any,
    y_val: pd.Series,
    model_name: str,
    cv: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[ModelMetrics, BaseEstimator]:
    """
    Train a candidate estimator on training features, record timing,
    evaluate on validation set, and compute 5-fold cross-validation.

    Args:
        model: Unfitted scikit-learn estimator.
        X_train: Training features (TF-IDF CSR matrix or series).
        y_train: Training sentiment targets.
        X_val: Validation features.
        y_val: Validation sentiment targets.
        model_name: Descriptive name of the classifier.
        cv: Number of cross-validation folds on training split.
        random_state: Seed for cross-validation splits.

    Returns:
        Tuple of (ModelMetrics, fitted_model).
    """
    estimator = clone(model)

    # 1. Measure Training Time
    t_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = round(time.perf_counter() - t_start, 4)

    # 2. Measure Validation Inference Latency
    t_infer_start = time.perf_counter()
    y_pred = estimator.predict(X_val)
    infer_time_ms = round((time.perf_counter() - t_infer_start) * 1000, 2)

    # 3. Validation Metrics
    val_acc = round(float(accuracy_score(y_val, y_pred)), 4)
    val_macro_p = round(float(precision_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    val_macro_r = round(float(recall_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    val_macro_f1 = round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    val_weighted_f1 = round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 4)

    # 4. Per-Class Precision, Recall, and F1
    classes = sorted(list(y_val.unique()))
    class_metrics = {}
    for cls in classes:
        # Binary mask for current class
        y_val_bin = (y_val == cls)
        y_pred_bin = (y_pred == cls)
        class_metrics[cls] = {
            "precision": round(float(precision_score(y_val_bin, y_pred_bin, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val_bin, y_pred_bin, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val_bin, y_pred_bin, zero_division=0)), 4),
            "support": int(y_val_bin.sum()),
        }

    # 5. K-Fold Cross Validation on Train Split
    cv_mean = 0.0
    cv_std = 0.0
    if cv > 1 and len(y_train) >= cv * 3:
        cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(estimator, X_train, y_train, cv=cv_folds, scoring="f1_macro")
        cv_mean = round(float(np.mean(scores)), 4)
        cv_std = round(float(np.std(scores)), 4)

    supports_prob = hasattr(estimator, "predict_proba")

    # Extract non-default hyperparameters safely
    params = {}
    for k, v in estimator.get_params().items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            params[k] = v

    metrics = ModelMetrics(
        model_name=model_name,
        hyperparameters=params,
        train_time_sec=train_time,
        infer_time_ms=infer_time_ms,
        val_accuracy=val_acc,
        val_macro_precision=val_macro_p,
        val_macro_recall=val_macro_r,
        val_macro_f1=val_macro_f1,
        val_weighted_f1=val_weighted_f1,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        class_metrics=class_metrics,
        supports_probabilities=supports_prob,
    )

    return metrics, estimator


def train_and_compare_models(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    sentiment_col: str = "sentiment",
    models: Optional[Dict[str, BaseEstimator]] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 2,
    cv: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    output_dir: Union[str, Path] = "models",
) -> ExperimentComparison:
    """
    Train, validate, and compare all candidate classifiers on the dataset.
    Identifies the best model based on validation Macro F1, constructs an
    end-to-end Pipeline with the fitted TF-IDF vectorizer, and persists artifacts.

    Args:
        df: Input preprocessed/validated DataFrame.
        text_col: Column containing comments.
        sentiment_col: Target sentiment column.
        models: Dictionary of models to compare.
        ngram_range: TF-IDF n-gram range.
        max_features: Maximum TF-IDF vocabulary size.
        min_df: Minimum document frequency.
        cv: Number of cross-validation folds.
        random_state: Random state seed.
        output_dir: Directory where models and metrics are saved.

    Returns:
        ExperimentComparison object containing metrics, comparison table, and artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Partition
    actual_text_col = text_col if text_col in df.columns else "text"
    split = split_dataset(
        df=df,
        text_col=actual_text_col,
        sentiment_col=sentiment_col,
        test_size=0.15,
        val_size=0.15,
        random_state=random_state,
    )

    # 2. Fit TF-IDF Extractor strictly on Train
    extractor = SentimentTfidfExtractor(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    X_train_vec = extractor.fit_transform(split.X_train)
    X_val_vec = extractor.transform(split.X_val)

    # 3. Train & Evaluate Candidates
    candidate_models = models or get_default_candidate_models(random_state=random_state)
    metrics_list: List[ModelMetrics] = []
    fitted_estimators: Dict[str, BaseEstimator] = {}

    for name, candidate in candidate_models.items():
        m_result, fitted_model = evaluate_model(
            model=candidate,
            X_train=X_train_vec,
            y_train=split.y_train,
            X_val=X_val_vec,
            y_val=split.y_val,
            model_name=name,
            cv=cv,
            random_state=random_state,
        )
        metrics_list.append(m_result)
        fitted_estimators[name] = fitted_model

    # 4. Construct Comparison Table
    table_rows = []
    for m in metrics_list:
        table_rows.append({
            "Model": m.model_name,
            "Accuracy": f"{m.val_accuracy * 100:.2f}%",
            "Macro F1": f"{m.val_macro_f1:.4f}",
            "Macro Precision": f"{m.val_macro_precision:.4f}",
            "Macro Recall": f"{m.val_macro_recall:.4f}",
            "CV Macro F1 (Train)": f"{m.cv_macro_f1_mean:.4f} (±{m.cv_macro_f1_std:.4f})",
            "Train Time (s)": f"{m.train_time_sec:.4f}s",
            "Infer Latency (ms)": f"{m.infer_time_ms:.2f}ms",
            "Probabilities": "Yes" if m.supports_probabilities else "No",
        })

    comp_df = pd.DataFrame(table_rows).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)

    # 5. Select Winning Model based on Validation Macro F1
    sorted_metrics = sorted(metrics_list, key=lambda x: x.val_macro_f1, reverse=True)
    best_candidate = sorted_metrics[0]
    best_model_name = best_candidate.model_name
    best_macro_f1 = best_candidate.val_macro_f1

    rationale = (
        f"'{best_model_name}' selected as the superior classifier with the highest Validation Macro F1 "
        f"of {best_macro_f1:.4f} (Accuracy: {best_candidate.val_accuracy * 100:.2f}%, CV F1: {best_candidate.cv_macro_f1_mean:.4f}). "
        f"It effectively handles class imbalance via balanced weighting and provides fast inference ({best_candidate.infer_time_ms:.2f}ms)."
    )

    # 6. Build End-to-End Scikit-Learn Pipeline for Deployment
    # Integrates the fitted vectorizer with the winning estimator
    winning_estimator = fitted_estimators[best_model_name]
    deployment_pipeline = Pipeline([
        ("tfidf", extractor.vectorizer),
        ("classifier", winning_estimator),
    ])

    # 7. Persist Artifacts
    saved_artifacts = {}

    # Save vectorizer
    vec_path = out_path / "vectorizer.pkl"
    extractor.save(vec_path)
    saved_artifacts["vectorizer"] = str(vec_path)

    # Save winning model pipeline
    model_path = out_path / "sentiment_model.pkl"
    joblib.dump(deployment_pipeline, model_path)
    saved_artifacts["model"] = str(model_path)

    # Save JSON comparison report
    report_data = {
        "best_model_name": best_model_name,
        "best_macro_f1": best_macro_f1,
        "selection_rationale": rationale,
        "comparison_table": comp_df.to_dict(orient="records"),
        "models": [m.to_dict() for m in metrics_list],
    }
    json_path = out_path / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    saved_artifacts["comparison_json"] = str(json_path)

    return ExperimentComparison(
        metrics_list=metrics_list,
        comparison_table=comp_df,
        best_model_name=best_model_name,
        best_macro_f1=best_macro_f1,
        selection_rationale=rationale,
        saved_artifacts=saved_artifacts,
    )
