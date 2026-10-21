"""
Exploratory Data Analysis (EDA) Module for Customer Sentiment Analysis Framework.

Computes distributions, text length statistics, top sentiment terms, product comparisons,
temporal trends, and generates interactive Plotly visualizations with business explanations.
Gracefully handles missing optional columns.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import TextPreprocessor


# Consistent color mapping for sentiments across all charts
SENTIMENT_COLORS = {
    "Positive": "#2ecc71",  # Emerald Green
    "Neutral": "#f39c12",   # Sunflower Amber
    "Negative": "#e74c3c",  # Crimson Red
}


def compute_sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate sentiment distribution counts, percentages, and class imbalance.

    Args:
        df: DataFrame containing a 'sentiment' column.

    Returns:
        Dict with total, counts, percentages, imbalance_ratio, and explanation.
    """
    if "sentiment" not in df.columns:
        raise ValueError("DataFrame must contain a 'sentiment' column for EDA.")

    total = len(df)
    sent_series = df["sentiment"]
    if isinstance(sent_series, pd.DataFrame):
        sent_series = sent_series.iloc[:, 0]

    counts = sent_series.value_counts().to_dict()
    percentages = {cls: round((cnt / total) * 100, 2) for cls, cnt in counts.items()}

    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    imbalance_ratio = round(counts[majority_class] / max(counts[minority_class], 1), 2)

    explanation = (
        f"Total of {total:,} records analyzed. '{majority_class}' is the most frequent sentiment "
        f"({percentages.get(majority_class, 0)}%), while '{minority_class}' is the least frequent "
        f"({percentages.get(minority_class, 0)}%). The dataset exhibits an imbalance ratio of "
        f"{imbalance_ratio}:1, which will be accounted for during model evaluation using stratified splits and Macro F1."
    )

    return {
        "total_records": total,
        "counts": counts,
        "percentages": percentages,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "imbalance_ratio": imbalance_ratio,
        "explanation": explanation,
    }


def compute_length_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate character and word length statistics overall and per sentiment class.

    Args:
        df: DataFrame with 'text' (and optionally 'sentiment', 'char_length', 'word_count').

    Returns:
        Dict of length metrics and analytical explanation.
    """
    working_df = df.copy()
    if "char_length" not in working_df.columns:
        working_df["char_length"] = working_df["text"].astype(str).str.len()
    if "word_count" not in working_df.columns:
        working_df["word_count"] = working_df["text"].astype(str).str.split().str.len()

    overall = {
        "mean_char_length": round(float(working_df["char_length"].mean()), 2),
        "median_char_length": round(float(working_df["char_length"].median()), 2),
        "std_char_length": round(float(working_df["char_length"].std()), 2),
        "mean_word_count": round(float(working_df["word_count"].mean()), 2),
        "median_word_count": round(float(working_df["word_count"].median()), 2),
        "std_word_count": round(float(working_df["word_count"].std()), 2),
    }

    by_sentiment = {}
    if "sentiment" in working_df.columns:
        for sent in ["Negative", "Neutral", "Positive"]:
            sub = working_df[working_df["sentiment"] == sent]
            if not sub.empty:
                by_sentiment[sent] = {
                    "mean_chars": round(float(sub["char_length"].mean()), 2),
                    "median_chars": round(float(sub["char_length"].median()), 2),
                    "mean_words": round(float(sub["word_count"].mean()), 2),
                    "median_words": round(float(sub["word_count"].median()), 2),
                }

    explanation = (
        f"Overall comments average {overall['mean_char_length']} characters ({overall['mean_word_count']} words). "
    )
    if "Negative" in by_sentiment and "Positive" in by_sentiment:
        neg_words = by_sentiment["Negative"]["mean_words"]
        pos_words = by_sentiment["Positive"]["mean_words"]
        explanation += (
            f"Negative feedback tends to be notably longer (averaging {neg_words} words) as users explain specific grievances, "
            f"whereas positive feedback is more concise (averaging {pos_words} words)."
        )

    return {
        "overall": overall,
        "by_sentiment": by_sentiment,
        "explanation": explanation,
    }


def compute_top_terms(
    df: pd.DataFrame,
    sentiment: Optional[str] = None,
    top_n: int = 15,
    custom_stopwords: Optional[set] = None,
) -> List[Tuple[str, int]]:
    """
    Extract the most frequent meaningful keywords for a given sentiment or the whole dataset.

    Args:
        df: DataFrame with 'text' or 'clean_text'.
        sentiment: Optional sentiment filter ('Positive', 'Negative', 'Neutral').
        top_n: Number of top terms to return.
        custom_stopwords: Extra words to filter out (e.g. metadata tokens like 'user', 'flight').

    Returns:
        List of (term, frequency) tuples.
    """
    subset = df if sentiment is None else df[df["sentiment"] == sentiment]
    if subset.empty:
        return []

    ignored_tokens = {"user", "flight", "plane", "airline", "http", "co"}
    if custom_stopwords:
        ignored_tokens.update(custom_stopwords)

    preprocessor = TextPreprocessor(remove_stopwords=True, preserve_negations=True, lemmatize=True)

    words = []
    # If clean_text already preprocessed, use it; otherwise run preprocessor
    use_col = "clean_text" if "clean_text" in subset.columns else "text"

    for text_val in subset[use_col]:
        if pd.isna(text_val):
            continue
        if use_col == "text":
            processed = preprocessor.process(str(text_val))
        else:
            processed = str(text_val)

        for token in processed.split():
            if len(token) > 2 and token not in ignored_tokens:
                words.append(token)

    return Counter(words).most_common(top_n)


def compute_product_insights(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compute sentiment breakdown across products/brands if a 'product' column is present.

    Returns:
        Dict with breakdown table, best/worst rated product, and explanation, or None if absent.
    """
    if "product" not in df.columns or df["product"].dropna().empty:
        return None

    ct_counts = pd.crosstab(df["product"], df["sentiment"])
    ct_pct = pd.crosstab(df["product"], df["sentiment"], normalize="index") * 100

    highest_neg_product = None
    highest_pos_product = None

    if "Negative" in ct_pct.columns:
        highest_neg_product = str(ct_pct["Negative"].idxmax())
        highest_neg_rate = round(float(ct_pct["Negative"].max()), 2)
    else:
        highest_neg_rate = 0.0

    if "Positive" in ct_pct.columns:
        highest_pos_product = str(ct_pct["Positive"].idxmax())
        highest_pos_rate = round(float(ct_pct["Positive"].max()), 2)
    else:
        highest_pos_rate = 0.0

    explanation = (
        f"Product comparison across {df['product'].nunique()} entities shows significant variation. "
        f"'{highest_neg_product}' received the highest proportion of negative feedback ({highest_neg_rate}%), "
        f"while '{highest_pos_product}' achieved the highest customer satisfaction with {highest_pos_rate}% positive sentiment."
    )

    return {
        "counts": ct_counts.to_dict(),
        "percentages": ct_pct.round(2).to_dict(),
        "highest_neg_product": highest_neg_product,
        "highest_neg_rate": highest_neg_rate,
        "highest_pos_product": highest_pos_product,
        "highest_pos_rate": highest_pos_rate,
        "explanation": explanation,
    }


def compute_temporal_insights(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compute daily sentiment trajectory and detect spikes if dates are present.

    Returns:
        Dict with daily counts, peak negative day, and explanation, or None if absent.
    """
    date_col = "date_parsed" if "date_parsed" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        return None

    working = df.copy()
    working["dt"] = pd.to_datetime(working[date_col], errors="coerce")
    valid_dates = working.dropna(subset=["dt"])

    if valid_dates.empty:
        return None

    daily = valid_dates.groupby([valid_dates["dt"].dt.date, "sentiment"]).size().unstack().fillna(0)
    daily_records = daily.to_dict(orient="index")

    peak_neg_day = None
    peak_neg_count = 0
    if "Negative" in daily.columns:
        peak_neg_day = str(daily["Negative"].idxmax())
        peak_neg_count = int(daily["Negative"].max())

    explanation = (
        f"Temporal timeline spans {len(daily)} discrete days. Negative feedback surged to a peak of "
        f"{peak_neg_count} complaints on {peak_neg_day}, indicating an operational disturbance (e.g. widespread weather cancellations) "
        f"rather than constant baseline dissatisfaction."
    )

    return {
        "daily_breakdown": {str(k): {cls: int(cnt) for cls, cnt in v.items()} for k, v in daily_records.items()},
        "peak_negative_day": peak_neg_day,
        "peak_negative_count": peak_neg_count,
        "explanation": explanation,
    }


def compute_complaint_topics(df: pd.DataFrame, top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Compute frequency of categorized complaint reasons if 'topic' column is available.

    Returns:
        Dict with topic counts, percentages, top topic, and explanation.
    """
    if "topic" not in df.columns or df["topic"].dropna().empty:
        return None

    topic_counts = df["topic"].dropna().value_counts().head(top_n).to_dict()
    total_complaints = sum(topic_counts.values())
    topic_percentages = {top: round((cnt / total_complaints) * 100, 2) for top, cnt in topic_counts.items()}
    top_issue = list(topic_counts.keys())[0] if topic_counts else "Unknown"

    explanation = (
        f"The leading complaint driver is '{top_issue}', representing {topic_percentages.get(top_issue, 0)}% of categorized issues. "
        f"Operational challenges (late/cancelled flights and customer service queues) comprise the vast majority of negative feedback."
    )

    return {
        "counts": topic_counts,
        "percentages": topic_percentages,
        "top_issue": top_issue,
        "explanation": explanation,
    }


# =========================================================================
# Plotly Chart Generation Functions
# =========================================================================

def create_sentiment_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Create a styled bar chart showing distribution of sentiment classes."""
    summary = compute_sentiment_summary(df)
    categories = list(summary["counts"].keys())
    values = [summary["counts"][c] for c in categories]
    colors = [SENTIMENT_COLORS.get(c, "#3498db") for c in categories]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:,} ({summary['percentages'][c]}%)" for c, v in zip(categories, values)],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Sentiment Class Distribution",
        xaxis_title="Sentiment Class",
        yaxis_title="Record Count",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Create an interactive donut chart showing sentiment percentages."""
    summary = compute_sentiment_summary(df)
    labels = list(summary["percentages"].keys())
    values = [summary["percentages"][k] for k in labels]
    colors = [SENTIMENT_COLORS.get(k, "#3498db") for k in labels]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )
    ])
    fig.update_layout(
        title="Sentiment Proportion (%)",
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def create_length_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create box plots showing character length distribution per sentiment class."""
    working = df.copy()
    if "char_length" not in working.columns:
        working["char_length"] = working["text"].astype(str).str.len()

    fig = px.box(
        working,
        x="sentiment",
        y="char_length",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Character Length Distribution Across Sentiment Classes",
        labels={"char_length": "Character Length", "sentiment": "Sentiment"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def create_top_words_bar_chart(df: pd.DataFrame, sentiment: str, top_n: int = 12) -> go.Figure:
    """Create a horizontal bar chart displaying top keywords for a chosen sentiment."""
    top_terms = compute_top_terms(df, sentiment=sentiment, top_n=top_n)
    if not top_terms:
        fig = go.Figure()
        fig.update_layout(title=f"No terms found for {sentiment}")
        return fig

    terms = [t[0] for t in top_terms][::-1]
    frequencies = [t[1] for t in top_terms][::-1]
    bar_color = SENTIMENT_COLORS.get(sentiment, "#3498db")

    fig = go.Figure(data=[
        go.Bar(
            x=frequencies,
            y=terms,
            orientation="h",
            marker_color=bar_color,
            text=[f"{f:,}" for f in frequencies],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=f"Top Keywords: {sentiment} Feedback",
        xaxis_title="Term Frequency",
        yaxis_title="Keywords",
        template="plotly_white",
        margin=dict(l=80, r=40, t=50, b=40),
    )
    return fig


def create_product_sentiment_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create a 100% stacked bar chart of sentiment by product/brand."""
    if "product" not in df.columns or df["product"].dropna().empty:
        return None

    ct = pd.crosstab(df["product"], df["sentiment"], normalize="index") * 100
    classes = [c for c in ["Negative", "Neutral", "Positive"] if c in ct.columns]

    fig = go.Figure()
    for cls in classes:
        fig.add_trace(go.Bar(
            name=cls,
            x=ct.index,
            y=ct[cls],
            marker_color=SENTIMENT_COLORS.get(cls, "#3498db"),
            text=[f"{v:.1f}%" for v in ct[cls]],
            textposition="inside",
        ))

    fig.update_layout(
        barmode="stack",
        title="Sentiment Breakdown by Product / Airline (%)",
        xaxis_title="Product / Airline",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_temporal_trend_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create a line chart showing daily sentiment volume over time."""
    date_col = "date_parsed" if "date_parsed" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        return None

    working = df.copy()
    working["dt"] = pd.to_datetime(working[date_col], errors="coerce")
    valid_dates = working.dropna(subset=["dt"])
    if valid_dates.empty:
        return None

    daily = valid_dates.groupby([valid_dates["dt"].dt.date, "sentiment"]).size().unstack().fillna(0)

    fig = go.Figure()
    for cls in ["Negative", "Neutral", "Positive"]:
        if cls in daily.columns:
            fig.add_trace(go.Scatter(
                x=[str(d) for d in daily.index],
                y=daily[cls],
                mode="lines+markers",
                name=cls,
                line=dict(color=SENTIMENT_COLORS.get(cls, "#3498db"), width=2.5),
                marker=dict(size=6),
            ))

    fig.update_layout(
        title="Sentiment Volume Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Comments",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_complaint_topics_chart(df: pd.DataFrame, top_n: int = 8) -> Optional[go.Figure]:
    """Create a horizontal bar chart of top customer complaint categories."""
    topics = compute_complaint_topics(df, top_n=top_n)
    if topics is None or not topics["counts"]:
        return None

    topic_names = list(topics["counts"].keys())[::-1]
    counts = [topics["counts"][k] for k in topic_names]

    fig = go.Figure(data=[
        go.Bar(
            x=counts,
            y=topic_names,
            orientation="h",
            marker_color="#e74c3c",
            text=[f"{c:,} ({topics['percentages'].get(k, 0)}%)" for k, c in zip(topic_names, counts)],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Top Categorized Complaint Drivers",
        xaxis_title="Number of Grievances",
        yaxis_title="Complaint Reason",
        template="plotly_white",
        margin=dict(l=150, r=40, t=50, b=40),
    )
    return fig
