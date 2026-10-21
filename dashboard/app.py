"""
Customer Sentiment Analysis Framework — Interactive Streamlit Dashboard.

Provides an end-to-end, dataset-agnostic dashboard integrating data ingestion,
quality validation, exploratory data analysis, real-time prediction,
customer feedback filtering, chronological spike alerts, model metrics,
and exportable executive reports.
"""

from pathlib import Path
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import auto_detect_columns, load_dataset
from src.eda import (
    compute_sentiment_summary,
    compute_length_statistics,
    compute_top_terms,
    create_sentiment_bar_chart,
    create_sentiment_donut_chart,
    create_length_distribution_chart,
    create_top_words_bar_chart,
    create_product_sentiment_chart,
    create_temporal_trend_chart,
    create_complaint_topics_chart,
)
from src.evaluate import (
    create_confusion_matrix_chart,
    CANONICAL_CLASSES,
)
from src.insights import generate_insights, CAUSALITY_DISCLAIMER
from src.predict import load_predictor
from src.validation import validate_dataset


# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis Framework",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths to default datasets and models
DEFAULT_SAMPLE_PATH = Path("data/sample/sample_airline_sentiment.csv")
DEFAULT_FULL_PATH = Path("data/processed/validated_airline_sentiment.csv")
EVAL_RESULTS_PATH = Path("models/evaluation_results.json")
COMPARISON_RESULTS_PATH = Path("models/model_comparison.json")


@st.cache_resource
def get_cached_predictor():
    """Load and cache the trained inference engine."""
    return load_predictor()


def initialize_session_state():
    """Initialize state variables for dataset and prediction caching."""
    if "active_df" not in st.session_state:
        # Load default benchmark if available
        data_path = DEFAULT_SAMPLE_PATH if DEFAULT_SAMPLE_PATH.exists() else None
        if data_path:
            raw_df, _ = load_dataset(data_path)
            val_res = validate_dataset(raw_df)
            predictor = get_cached_predictor()
            batch_res = predictor.predict_batch(val_res.valid_df, text_col="text")
            st.session_state.active_df = batch_res.dataframe
            st.session_state.validation_result = val_res
            st.session_state.dataset_source_name = f"Sample Benchmark ({len(val_res.valid_df)} rows)"
        else:
            st.session_state.active_df = pd.DataFrame()
            st.session_state.validation_result = None
            st.session_state.dataset_source_name = "None"


initialize_session_state()
predictor = get_cached_predictor()


# -------------------------------------------------------------------------
# Sidebar Navigation & Dataset Selector
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ Sentiment Framework")
    st.caption("Machine Learning for Social & Customer Text")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📂 Ingestion & Quality",
            "📊 Exploratory Data (EDA)",
            "🧠 Sentiment Analysis",
            "🔍 Feedback Explorer",
            "📈 Trends & Spikes",
            "🎯 Model Performance",
            "📑 Insights & Export",
        ],
    )

    st.markdown("---")
    st.subheader("Active Dataset")
    st.info(f"📁 **Source:** {st.session_state.get('dataset_source_name', 'None')}\n\n"
            f"📊 **Rows:** {len(st.session_state.active_df):,}")

    st.markdown("---")
    st.caption("AI Capstone Project — Stage 11")


# Helper for rendering sentiment badges
def format_sentiment_badge(sentiment: str) -> str:
    color_map = {
        "Positive": "#2ecc71",
        "Neutral": "#f39c12",
        "Negative": "#e74c3c",
    }
    color = color_map.get(sentiment, "#95a5a6")
    return f'<span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{sentiment}</span>'


def prepare_df_with_sentiment(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Safely prepare a DataFrame with a unique 'sentiment' column for charting."""
    res = data.copy()
    if col_name != "sentiment" and "sentiment" in res.columns:
        res = res.drop(columns=["sentiment"])
    if col_name in res.columns and col_name != "sentiment":
        res = res.rename(columns={col_name: "sentiment"})
    return res


# =========================================================================
# Page 1: Overview
# =========================================================================
if nav_selection == "🏠 Overview":
    st.title("Customer Sentiment Analysis Framework")
    st.markdown(
        "An end-to-end machine learning system that automates sentiment identification, "
        "validates text quality, identifies operational trends, and converts customer feedback into actionable business insights."
    )

    if st.session_state.active_df.empty:
        st.warning("No dataset is currently loaded. Go to **Ingestion & Quality** to upload or load data.")
    else:
        df = st.session_state.active_df
        sentiment_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"

        # KPIs
        total = len(df)
        counts = df[sentiment_col].value_counts().to_dict()
        neg_count = counts.get("Negative", 0)
        pos_count = counts.get("Positive", 0)
        neu_count = counts.get("Neutral", 0)

        neg_rate = round((neg_count / total) * 100, 1)
        pos_rate = round((pos_count / total) * 100, 1)
        neu_rate = round((neu_count / total) * 100, 1)
        nss = round(pos_rate - neg_rate, 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Comments", f"{total:,}")
        col2.metric("Negative Rate", f"{neg_rate}%", delta=f"-{neg_rate}%", delta_color="inverse")
        col3.metric("Positive Rate", f"{pos_rate}%", delta=f"{pos_rate}%")
        col4.metric("Net Sentiment Score", f"{nss}", delta="Positive - Negative")

        st.markdown("---")

        # Distribution Visualizations
        c1, c2 = st.columns([3, 2])
        overview_eda_df = prepare_df_with_sentiment(df, sentiment_col)
        with c1:
            st.subheader("Sentiment Class Distribution")
            bar_fig = create_sentiment_bar_chart(overview_eda_df)
            st.plotly_chart(bar_fig, use_container_width=True)

        with c2:
            st.subheader("Proportion Breakdown")
            donut_fig = create_sentiment_donut_chart(overview_eda_df)
            st.plotly_chart(donut_fig, use_container_width=True)

        st.info(
            "💡 **Framework Objective:** Convert noisy unstructured text into measurable metrics. "
            "All statistics on this dashboard are dynamically calculated from the loaded dataset."
        )


# =========================================================================
# Page 2: Ingestion & Quality
# =========================================================================
elif nav_selection == "📂 Ingestion & Quality":
    st.title("Dataset Ingestion & Quality Validation")
    st.markdown(
        "Upload a customer text CSV or switch datasets. The ingestion engine automatically detects column schemas, "
        "validates text records, removes exact duplicates, and logs all data cleaning steps transparently."
    )

    tab_upload, tab_presets = st.tabs(["📤 Upload Custom CSV", "📚 Load Benchmark Datasets"])

    with tab_upload:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                # Preview raw uploaded columns
                raw_preview = pd.read_csv(uploaded_file, nrows=5)
                uploaded_file.seek(0)
                detected = auto_detect_columns(list(raw_preview.columns))

                st.write("🔍 **Auto-Detected Column Mapping:**", detected)

                with st.expander("Adjust Column Mapping (Optional)"):
                    cols = ["(None)"] + list(raw_preview.columns)
                    custom_text = st.selectbox("Text Column", cols, index=cols.index(detected.get("text")) if detected.get("text") in cols else 0)
                    custom_sent = st.selectbox("Sentiment Column (if labelled)", cols, index=cols.index(detected.get("sentiment")) if detected.get("sentiment") in cols else 0)
                    custom_prod = st.selectbox("Product/Airline Column", cols, index=cols.index(detected.get("product")) if detected.get("product") in cols else 0)
                    custom_date = st.selectbox("Date Column", cols, index=cols.index(detected.get("date")) if detected.get("date") in cols else 0)

                if st.button("Ingest & Validate Uploaded Dataset", type="primary"):
                    col_map = {}
                    if custom_text != "(None)": col_map["text"] = custom_text
                    if custom_sent != "(None)": col_map["sentiment"] = custom_sent
                    if custom_prod != "(None)": col_map["product"] = custom_prod
                    if custom_date != "(None)": col_map["date"] = custom_date

                    uploaded_file.seek(0)
                    loaded_df, report = load_dataset(
                        uploaded_file,
                        column_mapping=col_map if col_map else None,
                        require_sentiment=False,
                    )
                    val_res = validate_dataset(loaded_df, require_sentiment=False)

                    # Run predictions
                    with st.spinner("Generating sentiment predictions on uploaded data..."):
                        batch_res = predictor.predict_batch(val_res.valid_df, text_col="text")

                    st.session_state.active_df = batch_res.dataframe
                    st.session_state.validation_result = val_res
                    st.session_state.dataset_source_name = f"Uploaded: {uploaded_file.name} ({len(val_res.valid_df)} rows)"
                    st.success(f"Successfully loaded and validated {len(val_res.valid_df):,} records!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")

    with tab_presets:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Development Sample (300 records)"):
                loaded_df, _ = load_dataset(DEFAULT_SAMPLE_PATH)
                val_res = validate_dataset(loaded_df)
                batch_res = predictor.predict_batch(val_res.valid_df, text_col="text")
                st.session_state.active_df = batch_res.dataframe
                st.session_state.validation_result = val_res
                st.session_state.dataset_source_name = f"Sample Benchmark ({len(val_res.valid_df)} rows)"
                st.success("Loaded sample dataset!")
                st.rerun()

        with c2:
            if DEFAULT_FULL_PATH.exists():
                if st.button("Load Full Benchmark (14,427 records)"):
                    loaded_df, _ = load_dataset(DEFAULT_FULL_PATH)
                    val_res = validate_dataset(loaded_df)
                    batch_res = predictor.predict_batch(val_res.valid_df, text_col="text")
                    st.session_state.active_df = batch_res.dataframe
                    st.session_state.validation_result = val_res
                    st.session_state.dataset_source_name = f"Full Benchmark ({len(val_res.valid_df):,} rows)"
                    st.success("Loaded full benchmark dataset!")
                    st.rerun()

    st.markdown("---")
    st.subheader("Data Quality Summary")

    if st.session_state.validation_result:
        vr = st.session_state.validation_result.report
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Records Ingested", f"{vr.total_records:,}")
        q2.metric("Valid Records Retained", f"{vr.valid_records:,}")
        q3.metric("Duplicate Comments Removed", f"{vr.duplicate_count:,}")
        q4.metric("Avg Character Length", f"{vr.avg_char_length:.1f} chars")

        if vr.warnings:
            for w in vr.warnings:
                st.warning(f"⚠️ {w}")

        with st.expander("Inspected Data Sample (First 10 Rows)"):
            st.dataframe(st.session_state.active_df.head(10), use_container_width=True)

        if not st.session_state.validation_result.invalid_df.empty:
            with st.expander("Audited Removal Log (Removed Records)"):
                st.dataframe(st.session_state.validation_result.invalid_df, use_container_width=True)


# =========================================================================
# Page 3: Exploratory Data Analysis (EDA)
# =========================================================================
elif nav_selection == "📊 Exploratory Data (EDA)":
    st.title("Exploratory Data Analysis")
    st.markdown("Examine distribution patterns, comment length behaviors, and frequent sentiment keywords.")

    if st.session_state.active_df.empty:
        st.warning("No active dataset loaded.")
    else:
        df = st.session_state.active_df
        sentiment_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        eda_df = prepare_df_with_sentiment(df, sentiment_col)

        summary = compute_sentiment_summary(eda_df)
        st.info(f"📊 **Data Distribution Analysis:** {summary['explanation']}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Comment Length Distribution")
            len_fig = create_length_distribution_chart(eda_df)
            st.plotly_chart(len_fig, use_container_width=True)
            length_stats = compute_length_statistics(eda_df)
            st.caption(f"📝 {length_stats['explanation']}")

        with c2:
            st.subheader("Top Complaint Keywords")
            comp_fig = create_top_words_bar_chart(eda_df, sentiment="Negative", top_n=10)
            st.plotly_chart(comp_fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Top Positive Keywords")
            pos_fig = create_top_words_bar_chart(eda_df, sentiment="Positive", top_n=10)
            st.plotly_chart(pos_fig, use_container_width=True)

        with c4:
            st.subheader("Top Neutral / Inquiry Keywords")
            neu_fig = create_top_words_bar_chart(eda_df, sentiment="Neutral", top_n=10)
            st.plotly_chart(neu_fig, use_container_width=True)


# =========================================================================
# Page 4: Sentiment Analysis (Live Inference)
# =========================================================================
elif nav_selection == "🧠 Sentiment Analysis":
    st.title("Real-Time Sentiment Analysis Engine")
    st.markdown("Classify customer comments using the trained Logistic Regression + TF-IDF model pipeline.")

    st.subheader("1. Single Comment Classifier")
    sample_options = [
        "Select a sample comment or type your own below...",
        "The flight crew was absolutely amazing and the seats were so comfortable! ❤️",
        "Delayed 5 hours, no food vouchers, and lost our bags. Worst experience ever! 😡",
        "Can someone please let me know what gate the flight from Dallas is arriving at?",
        "Good perspective. If only this concern was shared before I arrived at the airport.",
        "Not bad, but could be much better.",
    ]
    selected_sample = st.selectbox("Sample Comment Prompts", sample_options)
    initial_text = "" if selected_sample == sample_options[0] else selected_sample

    user_text = st.text_area("Customer Comment Input:", value=initial_text, height=100)

    if st.button("Predict Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a non-empty comment.")
        else:
            res = predictor.predict_single(user_text)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Predicted Sentiment")
                st.markdown(format_sentiment_badge(res.predicted_sentiment), unsafe_allow_html=True)
                if res.predicted_probability:
                    st.markdown(f"**Predicted Probability:** `{res.predicted_probability:.2%}`")
                    st.progress(res.predicted_probability)

            with c2:
                st.markdown("### Class Probabilities")
                if res.class_probabilities:
                    prob_df = pd.DataFrame(
                        list(res.class_probabilities.items()),
                        columns=["Class", "Probability"],
                    )
                    prob_df["Percentage"] = (prob_df["Probability"] * 100).round(2).astype(str) + "%"
                    st.dataframe(prob_df, use_container_width=True)

            with st.expander("Inspect NLP Preprocessing Tokens"):
                st.code(f"Original Text: {res.original_text}\nClean Tokens:  {res.clean_text}")


# =========================================================================
# Page 5: Feedback Explorer
# =========================================================================
elif nav_selection == "🔍 Feedback Explorer":
    st.title("Customer Feedback Explorer")
    st.markdown("Search, filter, and inspect customer comments across sentiment and metadata.")

    if st.session_state.active_df.empty:
        st.warning("No dataset loaded.")
    else:
        df = st.session_state.active_df.copy()
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        text_col = "original_text" if "original_text" in df.columns else "text"

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_query = st.text_input("Search keywords (e.g. 'delay', 'crew', 'lost', 'wifi'):")
        with c2:
            sentiment_filter = st.multiselect("Sentiment Filter", ["Negative", "Neutral", "Positive"], default=["Negative", "Neutral", "Positive"])
        with c3:
            product_filter = "All"
            if "product" in df.columns and df["product"].nunique() > 1:
                products = ["All"] + sorted(df["product"].dropna().unique().tolist())
                product_filter = st.selectbox("Product / Brand", products)

        filtered = df[df[sent_col].isin(sentiment_filter)]

        if search_query.strip():
            filtered = filtered[filtered[text_col].astype(str).str.contains(search_query, case=False, na=False)]

        if product_filter != "All" and "product" in filtered.columns:
            filtered = filtered[filtered["product"] == product_filter]

        st.caption(f"Showing **{len(filtered):,}** matching comments out of {len(df):,} total.")

        display_cols = [text_col, sent_col]
        if "predicted_probability" in filtered.columns:
            display_cols.append("predicted_probability")
        if "product" in filtered.columns:
            display_cols.append("product")
        if "date" in filtered.columns:
            display_cols.append("date")

        st.dataframe(filtered[display_cols], use_container_width=True)


# =========================================================================
# Page 6: Trends & Spikes
# =========================================================================
elif nav_selection == "📈 Trends & Spikes":
    st.title("Sentiment Trends & Spike Detection")
    st.markdown("Analyze chronological sentiment volume and detect statistical surges in customer grievances.")

    if st.session_state.active_df.empty:
        st.warning("No dataset loaded.")
    else:
        df = st.session_state.active_df
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        insights = generate_insights(df, sentiment_col=sent_col, min_daily_volume=10)

        # Spike Alerts
        if insights.spike_alerts:
            st.subheader("🚨 Statistical Sentiment Spike Alerts")
            for alert in insights.spike_alerts:
                st.error(
                    f"**{alert['severity']} SPIKE DETECTED on {alert['date']}**\n\n"
                    f"{alert['message']}"
                )
        else:
            st.success("✅ No statistical sentiment spike alerts triggered for the current volume threshold.")

        st.markdown(f"*{CAUSALITY_DISCLAIMER}*")
        st.markdown("---")

        c1, c2 = st.columns(2)
        trend_eda_df = prepare_df_with_sentiment(df, sent_col)
        with c1:
            st.subheader("Daily Sentiment Volume")
            t_chart = create_temporal_trend_chart(trend_eda_df)
            if t_chart:
                st.plotly_chart(t_chart, use_container_width=True)
            else:
                st.info("Date information is not available in the active dataset.")

        with c2:
            st.subheader("Product Sentiment Comparison")
            p_chart = create_product_sentiment_chart(trend_eda_df)
            if p_chart:
                st.plotly_chart(p_chart, use_container_width=True)
            else:
                st.info("Product/brand information is not available in the active dataset.")


# =========================================================================
# Page 7: Model Performance
# =========================================================================
elif nav_selection == "🎯 Model Performance":
    st.title("Machine Learning Model Evaluation")
    st.markdown("Comprehensive evaluation metrics and error audit strictly evaluated on the held-out test split.")

    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r") as f:
            eval_data = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{eval_data['test_accuracy'] * 100:.2f}%")
        m2.metric("Test Macro F1", f"{eval_data['test_macro_f1']:.4f}")
        m3.metric("Test Macro Precision", f"{eval_data['test_macro_precision']:.4f}")
        m4.metric("Test Macro Recall", f"{eval_data['test_macro_recall']:.4f}")

        st.markdown("---")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Confusion Matrix (Held-Out Test Set)")
            cm_fig = create_confusion_matrix_chart(eval_data["confusion_matrix"], eval_data["class_names"])
            st.plotly_chart(cm_fig, use_container_width=True)

        with c2:
            st.subheader("Per-Class Classification Metrics")
            clf_rep = eval_data["classification_report"]
            rows = []
            for cls in eval_data["class_names"]:
                rows.append({
                    "Class": cls,
                    "Precision": f"{clf_rep[cls]['precision']:.4f}",
                    "Recall": f"{clf_rep[cls]['recall']:.4f}",
                    "F1-Score": f"{clf_rep[cls]['f1-score']:.4f}",
                    "Support": int(clf_rep[cls]["support"]),
                })
            st.table(pd.DataFrame(rows))

            st.subheader("Error Archetypes Breakdown")
            st.caption("Distribution of misclassification causes in the test set:")
            for arch, cnt in eval_data["error_archetypes_count"].items():
                pct = (cnt / eval_data["total_errors"]) * 100
                st.write(f"- **{arch}:** {cnt} ({pct:.1f}%)")

        with st.expander("Model Limitations & Responsible AI Notice"):
            for lim in eval_data.get("limitations", []):
                st.markdown(f"- {lim}")

    else:
        st.info("Evaluation results JSON not found. Please run Stage 8 model evaluation.")


# =========================================================================
# Page 8: Insights & Export
# =========================================================================
elif nav_selection == "📑 Insights & Export":
    st.title("Executive Insights & Export")
    st.markdown("Download generated sentiment predictions and summary reports for presentation.")

    if st.session_state.active_df.empty:
        st.warning("No active dataset loaded.")
    else:
        df = st.session_state.active_df
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        insights = generate_insights(df, sentiment_col=sent_col)

        st.subheader("Executive Summary")
        st.info(f"📋 {insights.executive_summary}")

        st.markdown(f"*{CAUSALITY_DISCLAIMER}*")
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Download Prediction Dataset")
            st.caption("Exports full dataset with clean tokens, predicted sentiments, and probability scores.")
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                type="primary",
            )

        with c2:
            st.subheader("Download Summary Report")
            st.caption("Exports executive markdown summary containing all calculated metrics and findings.")
            report_md = f"""# Customer Sentiment Analysis — Executive Report

## Summary
{insights.executive_summary}

## KPIs
- **Total Comments:** {insights.kpis['total_comments']:,}
- **Negative Feedback Rate:** {insights.kpis['negative_rate']}%
- **Positive Feedback Rate:** {insights.kpis['positive_rate']}%
- **Neutral Rate:** {insights.kpis['neutral_rate']}%
- **Net Sentiment Score:** {insights.kpis['net_sentiment_score']}

## Top Complaint Keywords
{", ".join([f"{t[0]} ({t[1]})" for t in insights.top_complaint_terms])}

## Top Praise Keywords
{", ".join([f"{t[0]} ({t[1]})" for t in insights.top_positive_terms])}

## Methodological Note
{CAUSALITY_DISCLAIMER}
"""
            st.download_button(
                label="📥 Download Executive Report (.md)",
                data=report_md,
                file_name="sentiment_executive_report.md",
                mime="text/markdown",
            )
