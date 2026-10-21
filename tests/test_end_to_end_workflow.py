"""End-to-end integration test for the entire customer sentiment analysis pipeline.

Verifies the complete workflow sequentially:
1. Ingestion: Load raw data, detect columns, standardize schemas.
2. Validation: Clean text records, remove duplicates, produce audit metrics.
3. Preprocessing: Clean, tokenize, lemmatize, preserve original comment.
4. Exploratory Data Analysis: Calculate distributions, length stats, frequent n-grams.
5. Anti-Leakage Feature Extraction: Stratified split, fit TF-IDF on train only.
6. Model Training & Comparison: MultinomialNB, Logistic Regression, Linear SVM with cross-validation.
7. Held-Out Test Set Evaluation: Confusion matrix, per-class F1, error analysis.
8. Unseen Inference: Single and batch predictions with predicted probabilities.
9. Business Intelligence: KPIs, negative spike detection, comparative insights.
10. Data & Report Export: CSV and executive markdown reports.
"""

import io
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.validation import validate_dataset
from src.preprocessing import preprocess_dataframe
from src.eda import compute_sentiment_summary, compute_length_statistics, compute_top_terms
from src.train import train_and_compare_models
from src.evaluate import evaluate_and_audit_model
from src.predict import SentimentPredictor
from src.insights import generate_insights


def test_full_end_to_end_workflow(tmp_path):
    """Run complete pipeline end-to-end from raw data to export in a temporary environment."""
    data_path = Path("data/sample/sample_airline_sentiment.csv")
    assert data_path.exists(), "Sample dataset must exist for end-to-end verification."

    # 1. Ingestion
    raw_df, ingest_report = load_dataset(data_path)
    assert len(raw_df) > 0
    assert "text" in raw_df.columns
    assert "sentiment" in raw_df.columns
    assert ingest_report.total_records == len(raw_df)

    # 2. Validation
    val_res = validate_dataset(raw_df)
    assert len(val_res.valid_df) > 0
    assert val_res.report.valid_records == len(val_res.valid_df)
    assert val_res.report.empty_text_count == 0

    # 3. Preprocessing
    clean_df = preprocess_dataframe(val_res.valid_df, text_col="text", target_col="clean_text")
    assert "clean_text" in clean_df.columns
    assert "original_text" in clean_df.columns
    assert len(clean_df) == len(val_res.valid_df)

    # 4. Exploratory Data Analysis
    summary = compute_sentiment_summary(clean_df)
    assert summary["total_records"] == len(clean_df)
    assert summary["majority_class"] in ["Negative", "Neutral", "Positive"]

    len_stats = compute_length_statistics(clean_df)
    assert len_stats["overall"]["mean_char_length"] > 0
    assert len_stats["overall"]["mean_word_count"] > 0

    top_complaints = compute_top_terms(clean_df, sentiment="Negative", top_n=5)
    assert isinstance(top_complaints, list)

    # 5 & 6. Anti-Leakage Feature Extraction, Model Training & Candidate Comparison
    models_dir = tmp_path / "models"
    comparison = train_and_compare_models(
        df=clean_df,
        text_col="clean_text",
        sentiment_col="sentiment",
        cv=3,
        random_state=42,
        output_dir=models_dir,
    )
    assert comparison.best_model_name in ["Logistic Regression", "Linear SVM", "Multinomial Naive Bayes"]
    assert comparison.best_macro_f1 > 0.0
    assert Path(comparison.saved_artifacts["model"]).exists()
    assert Path(comparison.saved_artifacts["vectorizer"]).exists()
    assert Path(comparison.saved_artifacts["comparison_json"]).exists()

    # 7. Unbiased Final Test Set Evaluation (Held-out test split only)
    eval_report = evaluate_and_audit_model(
        model_path=comparison.saved_artifacts["model"],
        df=clean_df,
        text_col="clean_text",
        sentiment_col="sentiment",
        output_dir=models_dir,
        random_state=42,
    )
    assert 0.0 <= eval_report.test_accuracy <= 1.0
    assert 0.0 <= eval_report.test_macro_f1 <= 1.0
    assert eval_report.total_test_samples > 0
    assert len(eval_report.confusion_matrix) == 3

    # 8. Unseen Inference on Fresh Inputs
    predictor = SentimentPredictor(
        model_path=comparison.saved_artifacts["model"],
        vectorizer_path=comparison.saved_artifacts["vectorizer"],
    )
    unseen_single = predictor.predict_single("Flight was wonderfully smooth and on time! Amazing crew.")
    assert unseen_single.predicted_sentiment in ["Positive", "Neutral", "Negative"]
    assert unseen_single.predicted_probability is not None

    unseen_batch = predictor.predict_batch([
        "Horrible service, baggage missing and flight delayed 8 hours.",
        "Can you confirm the terminal gate for connecting flight 102?",
        "Loved the extra legroom and warm cookies!",
    ])
    assert len(unseen_batch.predictions) == 3
    assert len(unseen_batch.dataframe) == 3

    # 9. Business Analytics & Negative Spike Detection
    insights = generate_insights(clean_df, sentiment_col="sentiment", min_daily_volume=2)
    assert insights.kpis["total_comments"] == len(clean_df)
    assert 0.0 <= insights.kpis["negative_rate"] <= 100.0
    assert -100.0 <= insights.kpis["net_sentiment_score"] <= 100.0

    # 10. Data & Report Export
    csv_buf = io.StringIO()
    clean_df.to_csv(csv_buf, index=False)
    assert len(csv_buf.getvalue()) > 0

    exec_report = f"# Executive Report\n{insights.executive_summary}\n"
    assert len(exec_report) > 20
