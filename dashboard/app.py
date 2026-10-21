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
    render_section_header(
        "Corpus Distribution & Linguistic Telemetry",
        "Inspect comment length behaviors, class balance, topic drivers, and high-frequency sentiment n-grams.",
        badge="Linguistic EDA",
    )

    if st.session_state.active_df.empty:
        st.warning("No active dataset loaded.")
    else:
        df = st.session_state.active_df
        sentiment_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        eda_df = prepare_df_with_sentiment(df, sentiment_col)

        summary = compute_sentiment_summary(eda_df)
        render_callout(f"📊 <b>Corpus Distribution Telemetry:</b> {summary['explanation']}", callout_type="info")

        # Row 1: Length Distribution & Categorized Complaint Drivers / Products
        c1, c2 = st.columns(2)
        with c1:
            len_fig = apply_dark_theme(
                create_length_distribution_chart(eda_df),
                title="Character Length Distribution Across Sentiment Classes",
            )
            st.plotly_chart(len_fig, use_container_width=True)
            length_stats = compute_length_statistics(eda_df)
            st.caption(f"📝 {length_stats['explanation']}")

        with c2:
            # Check for topic column
            topics_fig = create_complaint_topics_chart(eda_df, top_n=8)
            if topics_fig is not None:
                st.plotly_chart(
                    apply_dark_theme(topics_fig, title="Top Categorized Complaint Drivers"),
                    use_container_width=True,
                )
            else:
                p_chart = create_product_sentiment_chart(eda_df)
                if p_chart is not None:
                    st.plotly_chart(
                        apply_dark_theme(p_chart, title="Product Sentiment Distribution (%)"),
                        use_container_width=True,
                    )
                else:
                    comp_bar = apply_dark_theme(
                        create_top_words_bar_chart(eda_df, sentiment="Negative", top_n=10),
                        title="Top Complaint Terms (Negative)",
                    )
                    st.plotly_chart(comp_bar, use_container_width=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        render_section_header(
            "Lexical Frequency Breakdown by Class",
            "Distinctive unigrams and bigrams extracted across customer dissatisfaction, advocacy, and neutral inquiries.",
            badge="N-Gram Analysis",
        )

        c3, c4, c5 = st.columns(3)
        with c3:
            comp_fig = apply_dark_theme(
                create_top_words_bar_chart(eda_df, sentiment="Negative", top_n=10),
                title="Top Complaint Keywords (Negative)",
            )
            st.plotly_chart(comp_fig, use_container_width=True)

        with c4:
            pos_fig = apply_dark_theme(
                create_top_words_bar_chart(eda_df, sentiment="Positive", top_n=10),
                title="Top Praise Keywords (Positive)",
            )
            st.plotly_chart(pos_fig, use_container_width=True)

        with c5:
            neu_fig = apply_dark_theme(
                create_top_words_bar_chart(eda_df, sentiment="Neutral", top_n=10),
                title="Top Inquiry Keywords (Neutral)",
            )
            st.plotly_chart(neu_fig, use_container_width=True)


# =========================================================================
# Page 4: Sentiment Analysis (Live Inference)
# =========================================================================
elif nav_selection == "🧠 Sentiment Analysis":
    st.title("Real-Time Sentiment Analysis Engine")
    render_section_header(
        "Machine Learning Inference Suite",
        "Deploy the production Logistic Regression + TF-IDF model pipeline for real-time text scoring and batch inference.",
        badge="Live Inference",
    )

    tab_single, tab_batch = st.tabs(["💬 Single Comment Classifier", "📦 Batch Text Inference"])

    with tab_single:
        sample_options = [
            "Select a sample comment or type your own below...",
            "The flight crew was absolutely amazing and the seats were so comfortable! ❤️",
            "Delayed 5 hours, no food vouchers, and lost our bags. Worst experience ever! 😡",
            "Can someone please let me know what gate the flight from Dallas is arriving at?",
            "Good perspective. If only this concern was shared before I arrived at the airport.",
            "Not bad, but could be much better.",
        ]
        selected_sample = st.selectbox("Curated Sample Prompts", sample_options)
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
                        st.markdown(f"**Confidence Score:** `{res.predicted_probability:.2%}`")
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

    with tab_batch:
        st.markdown(
            "Run bulk sentiment inference across multiple comments simultaneously using the trained model pipeline."
        )
        batch_input_mode = st.radio(
            "Batch Input Method",
            ["Paste Comments (One per line)", "Upload Batch CSV"],
            horizontal=True,
        )

        batch_texts = []
        if batch_input_mode == "Paste Comments (One per line)":
            batch_raw = st.text_area(
                "Paste comments below (one per line):",
                height=150,
                placeholder="Flight delayed 4 hours with zero updates!\nLoved the friendly cabin attendants and quick boarding ❤️\nCan you confirm if flight 204 has departed?",
            )
            if batch_raw.strip():
                batch_texts = [line.strip() for line in batch_raw.split("\n") if line.strip()]
        else:
            batch_file = st.file_uploader("Upload CSV containing comments", type=["csv"], key="batch_csv_upload")
            if batch_file is not None:
                try:
                    batch_upload_df = pd.read_csv(batch_file)
                    st.write(f"Loaded CSV with {len(batch_upload_df):,} rows.")
                    text_candidate = auto_detect_columns(list(batch_upload_df.columns)).get("text")
                    cols = list(batch_upload_df.columns)
                    selected_batch_col = st.selectbox(
                        "Select column containing comments",
                        cols,
                        index=cols.index(text_candidate) if text_candidate in cols else 0,
                    )
                    batch_texts = batch_upload_df[selected_batch_col].dropna().astype(str).tolist()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        if st.button("Run Batch Inference", type="primary"):
            if not batch_texts:
                st.warning("Please provide at least one comment for batch inference.")
            else:
                with st.spinner(f"Classifying {len(batch_texts):,} comments..."):
                    batch_res = predictor.predict_batch(batch_texts)

                st.success(f"Successfully classified {batch_res.total_records:,} comments!")

                # Batch summary KPIs
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    render_kpi_card("Batch Volume", f"{batch_res.total_records:,}", "Comments processed", badge_text="Complete", badge_type="violet", icon="📦")
                with b2:
                    neg_b = batch_res.sentiment_counts.get("Negative", 0)
                    render_kpi_card("Negative Feedback", f"{neg_b:,}", f"{round(neg_b / max(batch_res.total_records, 1) * 100, 1)}% of batch", badge_text="Negative", badge_type="negative", icon="⚠️")
                with b3:
                    pos_b = batch_res.sentiment_counts.get("Positive", 0)
                    render_kpi_card("Positive Feedback", f"{pos_b:,}", f"{round(pos_b / max(batch_res.total_records, 1) * 100, 1)}% of batch", badge_text="Positive", badge_type="positive", icon="✨")
                with b4:
                    neu_b = batch_res.sentiment_counts.get("Neutral", 0)
                    render_kpi_card("Neutral / Inquiries", f"{neu_b:,}", f"{round(neu_b / max(batch_res.total_records, 1) * 100, 1)}% of batch", badge_text="Neutral", badge_type="neutral", icon="💬")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # Predictions DataFrame Preview
                preview_cols = ["original_text", "predicted_sentiment"]
                if "predicted_probability" in batch_res.dataframe.columns:
                    preview_cols.append("predicted_probability")
                if "clean_text" in batch_res.dataframe.columns:
                    preview_cols.append("clean_text")

                st.dataframe(batch_res.dataframe[preview_cols], use_container_width=True)

                # CSV Download Button
                csv_buf = io.StringIO()
                batch_res.dataframe.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download Batch Predictions CSV",
                    data=csv_buf.getvalue(),
                    file_name="batch_sentiment_predictions.csv",
                    mime="text/csv",
                )


# =========================================================================
# Page 5: Feedback Explorer
# =========================================================================
elif nav_selection == "🔍 Feedback Explorer":
    st.title("Customer Feedback Explorer")
    render_section_header(
        "Granular Comment Search & Filter Telemetry",
        "Search, filter, and inspect customer comments across sentiment classes, keywords, and entity categories.",
        badge="Feedback Browser",
    )

    if st.session_state.active_df.empty:
        st.warning("No dataset loaded.")
    else:
        df = st.session_state.active_df.copy()
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        text_col = "original_text" if "original_text" in df.columns else ("text" if "text" in df.columns else ("clean_text" if "clean_text" in df.columns else df.columns[0]))

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_query = st.text_input("Search keywords (e.g. 'delay', 'crew', 'lost', 'wifi'):")
        with c2:
            sentiment_filter = st.multiselect(
                "Sentiment Filter",
                ["Negative", "Neutral", "Positive"],
                default=["Negative", "Neutral", "Positive"],
            )
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

        st.caption(f"Showing **{len(filtered):,}** matching comments out of {len(df):,} total records.")

        if filtered.empty:
            render_callout("ℹ️ No feedback records match the current keyword and sentiment filters.", callout_type="info")
        else:
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
    render_section_header(
        "Chronological Telemetry & Grievance Surge Detection",
        "Monitor daily sentiment volume trends and detect statistical surges in customer dissatisfaction with supporting evidence.",
        badge="Trend Intelligence",
    )

    if st.session_state.active_df.empty:
        st.warning("No dataset loaded.")
    else:
        df = st.session_state.active_df
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        text_col = "text" if "text" in df.columns else ("clean_text" if "clean_text" in df.columns else ("original_text" if "original_text" in df.columns else df.columns[0]))
        insights = generate_insights(df, text_col=text_col, sentiment_col=sent_col, min_daily_volume=10)

        # Statistical Spike Alerts Section
        st.subheader("🚨 Statistical Negative Grievance Spike Alerts")
        if insights.spike_alerts:
            for alert in insights.spike_alerts:
                render_spike_alert(
                    date=alert["date"],
                    message=alert["message"],
                    severity=alert["severity"],
                    daily_count=alert.get("negative_count"),
                    baseline_mean=alert.get("baseline_rate"),
                )
        else:
            render_callout(
                "✅ <b>No Statistical Negative Spikes:</b> Daily grievance volume remained within normal statistical bounds "
                "(less than 1.5 standard deviations above the historical baseline rate for qualified days).",
                callout_type="success",
            )

        # Methodological Causality Notice
        render_callout(
            f"⚠️ <b>Methodological Causality Notice:</b> {CAUSALITY_DISCLAIMER}",
            callout_type="warning",
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        trend_eda_df = prepare_df_with_sentiment(df, sent_col)
        with c1:
            t_chart = create_temporal_trend_chart(trend_eda_df)
            if t_chart:
                st.plotly_chart(
                    apply_dark_theme(t_chart, title="Daily Sentiment Volume Timeline"),
                    use_container_width=True,
                )
            else:
                render_callout("ℹ️ Date information is not available in the active dataset to plot timeline trends.", callout_type="info")

        with c2:
            p_chart = create_product_sentiment_chart(trend_eda_df)
            if p_chart:
                st.plotly_chart(
                    apply_dark_theme(p_chart, title="Comparative Sentiment by Product / Airline (%)"),
                    use_container_width=True,
                )
            else:
                render_callout("ℹ️ Product/brand entity column is not present in the active dataset to plot comparative distributions.", callout_type="info")


# =========================================================================
# Page 7: Model Performance
# =========================================================================
elif nav_selection == "🎯 Model Performance":
    st.title("Machine Learning Model Evaluation")
    render_section_header(
        "Candidate Benchmark & Test Set Audit",
        "Comparative benchmark across candidate classifiers and unbiased final evaluation on the held-out test split.",
        badge="Model Governance",
    )

    # 1. Multi-Model Candidate Leaderboard
    if COMPARISON_RESULTS_PATH.exists():
        with open(COMPARISON_RESULTS_PATH, "r") as f:
            comp_data = json.load(f)

        st.markdown(f"""
        <div class="surface-card" style="border-left: 4px solid #06D6A0; margin-bottom: 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #06D6A0; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏆 Selected Production Model</div>
            <h3 style="margin: 0 0 6px 0; color: #F8FAFC;">{comp_data['best_model_name']} (Balanced Class Weights)</h3>
            <p style="font-size: 0.88rem; color: #94A3B8; margin: 0; line-height: 1.5;">{comp_data['selection_rationale']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("1. Candidate Model Comparison (5-Fold Cross-Validation)")
        st.caption("Models trained strictly on the 70% training split and compared using 5-fold cross-validation and the 15% validation split.")
        st.table(pd.DataFrame(comp_data["comparison_table"]))

    # 2. Unbiased Held-Out Test Set Audit
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r") as f:
            eval_data = json.load(f)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_section_header(
            "Unbiased Held-Out Test Split Audit",
            "Strictly evaluated once on 2,165 held-out test comments (15% split) that were never exposed during training or model selection.",
            badge="Held-Out Test Set",
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi_card("Test Accuracy", f"{eval_data['test_accuracy'] * 100:.2f}%", "Overall classification accuracy", badge_text="Accuracy", badge_type="positive", icon="🎯")
        with m2:
            render_kpi_card("Test Macro F1", f"{eval_data['test_macro_f1']:.4f}", "Unweighted mean across classes", badge_text="Balanced", badge_type="violet", icon="⚖️")
        with m3:
            render_kpi_card("Test Macro Precision", f"{eval_data['test_macro_precision']:.4f}", "True positives / predicted positives", badge_text="Precision", badge_type="neutral", icon="🔍")
        with m4:
            render_kpi_card("Test Macro Recall", f"{eval_data['test_macro_recall']:.4f}", "True positives / actual positives", badge_text="Recall", badge_type="neutral", icon="📡")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Confusion Matrix (Held-Out Test Set)")
            cm_fig = apply_dark_theme(
                create_confusion_matrix_chart(eval_data["confusion_matrix"], eval_data["class_names"]),
                title="Test Set Confusion Matrix (Counts & Recall %)",
            )
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
                    "Support": f"{int(clf_rep[cls]['support']):,}",
                })
            st.table(pd.DataFrame(rows))

            st.subheader("Error Archetypes Breakdown")
            st.caption(f"Systematic audit of all {eval_data['total_errors']:,} test misclassifications:")
            for arch, cnt in eval_data["error_archetypes_count"].items():
                pct = (cnt / eval_data["total_errors"]) * 100
                st.write(f"- **{arch}:** `{cnt}` comments ({pct:.1f}%)")

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
    render_section_header(
        "Executive Intelligence Dossier & Export Suite",
        "Actionable business summaries, complaint/praise drivers, and download suite for reports and predictions.",
        badge="Executive Reporting",
    )

    if st.session_state.active_df.empty:
        st.warning("No active dataset loaded.")
    else:
        df = st.session_state.active_df
        sent_col = "predicted_sentiment" if "predicted_sentiment" in df.columns else "sentiment"
        text_col = "text" if "text" in df.columns else ("clean_text" if "clean_text" in df.columns else ("original_text" if "original_text" in df.columns else df.columns[0]))
        insights = generate_insights(df, text_col=text_col, sentiment_col=sent_col)

        # Executive Summary Card
        st.markdown(f"""
        <div class="surface-card" style="border-left: 4px solid #8B5CF6; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #8B5CF6; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">📋 Executive Summary</div>
            <p style="font-size: 0.95rem; color: #F8FAFC; margin: 0; line-height: 1.6;">{insights.executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # Strategic KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Total Analyzed", f"{insights.kpis['total_comments']:,}", "Customer feedback records", badge_text="Total", badge_type="violet", icon="💬")
        with k2:
            render_kpi_card("Negative Feedback", f"{insights.kpis['negative_rate']}%", "Total dissatisfaction share", badge_text="Negative", badge_type="negative", icon="⚠️")
        with k3:
            render_kpi_card("Positive Feedback", f"{insights.kpis['positive_rate']}%", "Brand advocacy share", badge_text="Positive", badge_type="positive", icon="✨")
        with k4:
            render_kpi_card("Net Sentiment Score", f"{insights.kpis['net_sentiment_score']}", "Advocacy vs dissatisfaction index", badge_text="Index", badge_type="positive" if insights.kpis['net_sentiment_score'] > 0 else "negative", icon="📈")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Strategic Pillars: Top Complaint vs Top Praise terms
        col_c, col_p = st.columns(2)
        with col_c:
            st.markdown("""
            <div class="surface-card" style="margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: #EF4444;">🚨 Top Complaint Drivers</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 10px;">Primary keywords driving negative customer sentiment:</p>
            </div>
            """, unsafe_allow_html=True)
            for term, count in insights.top_complaint_terms:
                st.markdown(f"- <b>{term}</b>: `{count:,}` mentions", unsafe_allow_html=True)

        with col_p:
            st.markdown("""
            <div class="surface-card" style="margin-bottom: 12px;">
                <h4 style="margin: 0 0 8px 0; color: #06D6A0;">✨ Top Praise Drivers</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 10px;">Primary keywords driving positive customer advocacy:</p>
            </div>
            """, unsafe_allow_html=True)
            for term, count in insights.top_positive_terms:
                st.markdown(f"- <b>{term}</b>: `{count:,}` mentions", unsafe_allow_html=True)

        render_callout(f"⚠️ <b>Methodological Causality Notice:</b> {CAUSALITY_DISCLAIMER}", callout_type="warning")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        render_section_header("Export Deliverables Suite", "Download generated prediction datasets and executive Markdown intelligence reports.", badge="Export Hub")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"""
            <div class="surface-card" style="margin-bottom: 10px;">
                <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">📥 Predictions Dataset (CSV)</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                    Exports full active dataset ({len(df):,} rows) with clean tokens, predicted sentiments, and calibrated probability scores.
                </p>
            </div>
            """, unsafe_allow_html=True)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                type="primary",
            )

        with d2:
            st.markdown("""
            <div class="surface-card" style="margin-bottom: 10px;">
                <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">📑 Executive Report (Markdown)</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                    Exports formatted executive summary document with all KPIs, complaint drivers, praise terms, and methodology notes.
                </p>
            </div>
            """, unsafe_allow_html=True)
            report_md = f"""# Customer Sentiment Analysis — Executive Intelligence Dossier

## 1. Executive Summary
{insights.executive_summary}

## 2. Key Performance Indicators
- **Total Feedback Volume:** {insights.kpis['total_comments']:,}
- **Negative Feedback Rate:** {insights.kpis['negative_rate']}%
- **Positive Feedback Rate:** {insights.kpis['positive_rate']}%
- **Neutral Rate:** {insights.kpis['neutral_rate']}%
- **Net Sentiment Score (NSS):** {insights.kpis['net_sentiment_score']}

## 3. Top Complaint Drivers (Negative)
{", ".join([f"{t[0]} ({t[1]:,})" for t in insights.top_complaint_terms])}

## 4. Top Praise Drivers (Positive)
{", ".join([f"{t[0]} ({t[1]:,})" for t in insights.top_positive_terms])}

## 5. Methodological Causality Notice
{CAUSALITY_DISCLAIMER}
"""
            st.download_button(
                label="📥 Download Executive Report (.md)",
                data=report_md,
                file_name="sentiment_executive_report.md",
                mime="text/markdown",
                type="primary",
            )
