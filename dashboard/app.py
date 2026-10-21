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

from dashboard.styles import (
    inject_custom_styles,
    PRIMARY_TEAL,
    ACCENT_VIOLET,
    CORAL_RED,
    AMBER_WARNING,
    MUTED_SLATE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from dashboard.chart_theme import apply_dark_theme
from dashboard.components import (
    render_kpi_card,
    render_section_header,
    render_sentiment_badge,
    get_sentiment_badge_html,
    render_callout,
    get_callout_html,
)


# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis Framework",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark luxury design system and scoped CSS
inject_custom_styles()

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
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">✈️</div>
        <div class="sidebar-brand-text">
            <h2>SENTIMENT.AI</h2>
            <span>Customer Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("<hr style='border-color: #23314E; margin: 16px 0 14px 0;'>", unsafe_allow_html=True)

    # Active Dataset Card
    source_label = st.session_state.get("dataset_source_name", "None")
    row_count = len(st.session_state.active_df)
    st.markdown(f"""
    <div style="background-color: #162238; border: 1px solid #23314E; border-radius: 10px; padding: 12px 14px;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Active Corpus</div>
        <div style="font-size: 0.90rem; font-weight: 700; color: #F8FAFC; word-break: break-word;">{source_label}</div>
        <div style="font-size: 0.78rem; color: #06D6A0; margin-top: 5px; font-weight: 600;">● {row_count:,} Verified Records</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.caption("AI Customer Sentiment Framework • v1.2.0")


# Helper for rendering sentiment badges using design system pills
def format_sentiment_badge(sentiment: str) -> str:
    return get_sentiment_badge_html(sentiment)


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
    render_section_header(
        "Executive Telemetry & Sentiment Health",
        "Real-time opinion metrics, net sentiment balance, and distribution analytics across customer feedback.",
        badge="Live Telemetry",
    )

    if st.session_state.active_df.empty:
        st.warning("No dataset is currently loaded. Go to **Ingestion & Quality** to upload or load data.")
    else:
        df = st.session_state.active_df
        sentiment_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"

        # KPIs Calculation
        total = len(df)
        counts = df[sentiment_col].value_counts().to_dict()
        neg_count = counts.get("Negative", 0)
        pos_count = counts.get("Positive", 0)
        neu_count = counts.get("Neutral", 0)

        neg_rate = round((neg_count / total) * 100, 1)
        pos_rate = round((pos_count / total) * 100, 1)
        neu_rate = round((neu_count / total) * 100, 1)
        nss = round(pos_rate - neg_rate, 1)

        # 4 Luxury KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                "Total Feedback Volume",
                f"{total:,}",
                "Customer comments ingested",
                badge_text="100% Ingested",
                badge_type="violet",
                icon="💬",
            )
        with k2:
            render_kpi_card(
                "Negative Grievance Rate",
                f"{neg_rate}%",
                f"{neg_count:,} dissatisfaction reports",
                badge_text=f"-{neg_rate}%",
                badge_type="negative",
                icon="⚠️",
            )
        with k3:
            render_kpi_card(
                "Positive Feedback Rate",
                f"{pos_rate}%",
                f"{pos_count:,} positive expressions",
                badge_text=f"+{pos_rate}%",
                badge_type="positive",
                icon="✨",
            )
        with k4:
            nss_badge_type = "positive" if nss > 0 else "negative"
            render_kpi_card(
                "Net Sentiment Score",
                f"{nss:+.1f}",
                "Balance index (Pos % - Neg %)",
                badge_text="Advocacy Dominant" if nss > 0 else "Grievance Dominant",
                badge_type=nss_badge_type,
                icon="📈",
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Distribution Visualizations
        c1, c2 = st.columns([3, 2])
        overview_eda_df = prepare_df_with_sentiment(df, sentiment_col)
        with c1:
            bar_fig = apply_dark_theme(
                create_sentiment_bar_chart(overview_eda_df),
                title="Sentiment Class Distribution",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        with c2:
            donut_fig = apply_dark_theme(
                create_sentiment_donut_chart(overview_eda_df),
                title="Sentiment Share Breakdown",
            )
            st.plotly_chart(donut_fig, use_container_width=True)

        # Chronological Feedback Trajectory Preview (when date information is available)
        date_col = "date_parsed" if "date_parsed" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is not None:
            trend_preview_fig = create_temporal_trend_chart(overview_eda_df)
            if trend_preview_fig is not None:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                styled_trend = apply_dark_theme(
                    trend_preview_fig,
                    title="Chronological Sentiment Trajectory (Daily Volume Trend)",
                )
                st.plotly_chart(styled_trend, use_container_width=True)

        render_callout(
            "💡 <b>Real-Time Dynamic Telemetry:</b> All metrics and proportions are calculated dynamically "
            "from the active dataset. Uploading a custom dataset updates dependent KPIs, distributions, and downstream inferences instantly.",
            callout_type="info",
        )


# =========================================================================
# Page 2: Ingestion & Quality
# =========================================================================
elif nav_selection == "📂 Ingestion & Quality":
    st.title("Dataset Ingestion & Quality Validation")
    render_section_header(
        "Corpus Ingestion & Data Hygiene Engine",
        "Upload customer text CSVs, auto-detect column schemas, and inspect automated data deduplication audit logs.",
        badge="Data Engineering",
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
            st.markdown("""
            <div class="surface-card" style="margin-bottom: 10px;">
                <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">🧪 Development Sample</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                    Balanced fixture with 300 comments (100 Positive, 100 Negative, 100 Neutral). Ideal for rapid verification.
                </p>
            </div>
            """, unsafe_allow_html=True)
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
                st.markdown("""
                <div class="surface-card" style="margin-bottom: 10px;">
                    <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">✈️ Full Benchmark Corpus</h4>
                    <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                        Full Twitter Airline Sentiment dataset (14,427 validated records across 6 major US airlines).
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Load Full Benchmark (14,427 records)"):
                    loaded_df, _ = load_dataset(DEFAULT_FULL_PATH)
                    val_res = validate_dataset(loaded_df)
                    batch_res = predictor.predict_batch(val_res.valid_df, text_col="text")
                    st.session_state.active_df = batch_res.dataframe
                    st.session_state.validation_result = val_res
                    st.session_state.dataset_source_name = f"Full Benchmark ({len(val_res.valid_df):,} rows)"
                    st.success("Loaded full benchmark dataset!")
                    st.rerun()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_section_header(
        "Data Quality & Hygiene Audit",
        "Automated validation metrics, duplicate comment purge counts, and inspected records.",
        badge="Audit Log",
    )

    if st.session_state.validation_result:
        vr = st.session_state.validation_result.report
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            render_kpi_card(
                "Total Ingested",
                f"{vr.total_records:,}",
                "Raw records parsed",
                badge_text="Input",
                badge_type="neutral",
                icon="📥",
            )
        with q2:
            retention_pct = round((vr.valid_records / max(vr.total_records, 1)) * 100, 1)
            render_kpi_card(
                "Valid Retained",
                f"{vr.valid_records:,}",
                f"{retention_pct}% retention rate",
                badge_text="Clean",
                badge_type="positive",
                icon="✅",
            )
        with q3:
            render_kpi_card(
                "Duplicates Purged",
                f"{vr.duplicate_count:,}",
                "Exact duplicates removed",
                badge_text="Deduplicated",
                badge_type="warning" if vr.duplicate_count > 0 else "neutral",
                icon="🧹",
            )
        with q4:
            render_kpi_card(
                "Mean Text Length",
                f"{vr.avg_char_length:.1f}",
                "Characters per comment",
                badge_text="Avg Length",
                badge_type="violet",
                icon="📏",
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Visual breakdown of data hygiene
        h1, h2 = st.columns([1, 1])
        with h1:
            hygiene_labels = ["Valid Records Retained"]
            hygiene_values = [vr.valid_records]
            hygiene_colors = [PRIMARY_TEAL]

            if vr.duplicate_count > 0:
                hygiene_labels.append("Duplicates Purged")
                hygiene_values.append(vr.duplicate_count)
                hygiene_colors.append(AMBER_WARNING)

            if vr.empty_text_count > 0:
                hygiene_labels.append("Empty Text Removed")
                hygiene_values.append(vr.empty_text_count)
                hygiene_colors.append(CORAL_RED)

            hygiene_fig = go.Figure(data=[
                go.Pie(
                    labels=hygiene_labels,
                    values=hygiene_values,
                    hole=0.6,
                    marker=dict(colors=hygiene_colors),
                    textinfo="label+value",
                    hoverinfo="label+value+percent",
                )
            ])
            apply_dark_theme(hygiene_fig, title="Data Hygiene & Retention Breakdown")
            st.plotly_chart(hygiene_fig, use_container_width=True)

        with h2:
            st.markdown(f"""
            <div class="surface-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="margin: 0 0 10px 0; color: #F8FAFC;">🛡️ Automated Validation Summary</h4>
                <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    <b>Schema Standardization:</b> All fields normalized into canonical representations.
                </p>
                <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    <b>Exact Duplicate Purge:</b> Found and quarantined <b>{vr.duplicate_count:,} duplicate comments</b> to protect downstream models from over-fitting.
                </p>
                <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin: 0;">
                    <b>Encoding Integrity:</b> Validated character encodings and standardized labels into <code>Positive</code>, <code>Negative</code>, and <code>Neutral</code>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if vr.warnings:
            for w in vr.warnings:
                render_callout(f"⚠️ {w}", callout_type="warning")

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
