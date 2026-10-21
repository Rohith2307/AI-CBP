"""
Analytics and Insight Generation Module for Customer Sentiment Analysis Framework.

Aggregates model predictions into business KPIs, temporal trends, product breakdowns,
top complaint and praise terms, and configurable negative-sentiment spike detection
while strictly avoiding unverified causal assertions.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.preprocessing import TextPreprocessor


CAUSALITY_DISCLAIMER = (
    "Methodological Note: Sentiment trends and spike alerts reflect observational patterns "
    "in customer text data. Statistical correlation does not establish causation; identified "
    "surges indicate when customer complaints intensified, but external operational records "
    "(such as flight logs, weather reports, or IT system status) are required to determine root causes."
)

IGNORED_COMMON_TOKENS = {
    "user", "flight", "plane", "airline", "co", "http", "https", "get", "would",
    "one", "go", "day", "make", "take", "could", "like", "time", "back"
}


@dataclass
class SpikeAlert:
    """Alert record for a statistically significant surge in negative sentiment."""
    date: str
    negative_count: int
    total_count: int
    negative_rate: float
    baseline_rate: float
    threshold_rate: float
    severity: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InsightReport:
    """Comprehensive executive insight report derived from sentiment predictions."""
    kpis: Dict[str, Any]
    top_complaint_terms: List[Tuple[str, int]]
    top_positive_terms: List[Tuple[str, int]]
    product_comparison: Optional[Dict[str, Any]]
    temporal_trends: Optional[Dict[str, Any]]
    spike_alerts: List[Dict[str, Any]]
    executive_summary: str
    causality_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_kpis(
    df: pd.DataFrame,
    sentiment_col: str = "predicted_sentiment",
) -> Dict[str, Any]:
    """
    Compute core sentiment KPIs from prediction results.

    Args:
        df: DataFrame containing predicted sentiments.
        sentiment_col: Column with predicted labels ('Positive', 'Negative', 'Neutral').

    Returns:
        Dict of KPI counts, rates, and Net Sentiment Score (NSS).
    """
    if sentiment_col not in df.columns:
        raise ValueError(f"Column '{sentiment_col}' not found in DataFrame.")

    total = len(df)
    if total == 0:
        return {
            "total_comments": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_count": 0,
            "negative_rate": 0.0,
            "neutral_rate": 0.0,
            "positive_rate": 0.0,
            "net_sentiment_score": 0.0,
        }

    counts = df[sentiment_col].value_counts().to_dict()
    neg_count = counts.get("Negative", 0)
    neu_count = counts.get("Neutral", 0)
    pos_count = counts.get("Positive", 0)

    neg_rate = round((neg_count / total) * 100, 2)
    neu_rate = round((neu_count / total) * 100, 2)
    pos_rate = round((pos_count / total) * 100, 2)

    # Net Sentiment Score = % Positive - % Negative (-100 to +100)
    net_sentiment_score = round(pos_rate - neg_rate, 2)

    return {
        "total_comments": total,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_count": pos_count,
        "negative_rate": neg_rate,
        "neutral_rate": neu_rate,
        "positive_rate": pos_rate,
        "net_sentiment_score": net_sentiment_score,
        "counts": counts,
        "percentages": {"Negative": neg_rate, "Neutral": neu_rate, "Positive": pos_rate},
    }


def extract_sentiment_keywords(
    df: pd.DataFrame,
    target_sentiment: str,
    text_col: str = "clean_text",
    sentiment_col: str = "predicted_sentiment",
    top_n: int = 15,
    custom_stopwords: Optional[set] = None,
) -> List[Tuple[str, int]]:
    """
    Extract high-frequency discriminative keywords for a targeted sentiment category.

    Args:
        df: DataFrame with predictions and text.
        target_sentiment: Class to inspect ('Negative', 'Positive', 'Neutral').
        text_col: Text column name ('clean_text' or 'text').
        sentiment_col: Sentiment column.
        top_n: Number of keywords to return.
        custom_stopwords: Additional tokens to filter.

    Returns:
        List of (keyword, frequency) tuples.
    """
    sub = df[df[sentiment_col] == target_sentiment]
    if sub.empty:
        return []

    stopwords_filter = set(IGNORED_COMMON_TOKENS)
    if custom_stopwords:
        stopwords_filter.update(custom_stopwords)

    preprocessor = TextPreprocessor(remove_stopwords=True, preserve_negations=True)

    tokens: List[str] = []
    use_col = text_col if text_col in sub.columns else ("original_text" if "original_text" in sub.columns else "text")

    for val in sub[use_col]:
        if pd.isna(val):
            continue
        cleaned = preprocessor.process(str(val)) if use_col != "clean_text" else str(val)
        for w in cleaned.split():
            if len(w) > 2 and w not in stopwords_filter:
                tokens.append(w)

    return Counter(tokens).most_common(top_n)


def compute_product_comparison(
    df: pd.DataFrame,
    product_col: str = "product",
    sentiment_col: str = "predicted_sentiment",
) -> Optional[Dict[str, Any]]:
    """
    Compare sentiment distributions across products or brands when available.

    Args:
        df: Input DataFrame.
        product_col: Product/airline column name.
        sentiment_col: Sentiment prediction column.

    Returns:
        Dict with crosstabs, rankings, and summary or None if column missing.
    """
    if product_col not in df.columns or df[product_col].dropna().empty:
        return None

    valid_df = df.dropna(subset=[product_col])
    if valid_df.empty:
        return None

    ct_counts = pd.crosstab(valid_df[product_col], valid_df[sentiment_col])
    ct_pct = pd.crosstab(valid_df[product_col], valid_df[sentiment_col], normalize="index") * 100

    highest_neg_brand = None
    highest_neg_rate = 0.0
    if "Negative" in ct_pct.columns:
        highest_neg_brand = str(ct_pct["Negative"].idxmax())
        highest_neg_rate = round(float(ct_pct["Negative"].max()), 2)

    highest_pos_brand = None
    highest_pos_rate = 0.0
    if "Positive" in ct_pct.columns:
        highest_pos_brand = str(ct_pct["Positive"].idxmax())
        highest_pos_rate = round(float(ct_pct["Positive"].max()), 2)

    return {
        "products_evaluated": int(valid_df[product_col].nunique()),
        "counts": ct_counts.to_dict(),
        "percentages": ct_pct.round(2).to_dict(),
        "highest_negative_product": highest_neg_brand,
        "highest_negative_rate": highest_neg_rate,
        "highest_positive_product": highest_pos_brand,
        "highest_positive_rate": highest_pos_rate,
    }


def compute_daily_sentiment_trends(
    df: pd.DataFrame,
    date_col: str = "date_parsed",
    sentiment_col: str = "predicted_sentiment",
) -> Optional[pd.DataFrame]:
    """
    Aggregate sentiment counts and percentage rates by calendar date.

    Args:
        df: Input DataFrame.
        date_col: Date column name.
        sentiment_col: Prediction column.

    Returns:
        DataFrame indexed by date with total, counts, and percentage rates, or None.
    """
    actual_date_col = date_col if date_col in df.columns else ("date" if "date" in df.columns else None)
    if actual_date_col is None:
        return None

    working = df.copy()
    working["dt_clean"] = pd.to_datetime(working[actual_date_col], errors="coerce")
    valid_dates = working.dropna(subset=["dt_clean"])
    if valid_dates.empty:
        return None

    daily = valid_dates.groupby(valid_dates["dt_clean"].dt.date).agg(
        total=(sentiment_col, "count"),
        negative=(sentiment_col, lambda s: (s == "Negative").sum()),
        neutral=(sentiment_col, lambda s: (s == "Neutral").sum()),
        positive=(sentiment_col, lambda s: (s == "Positive").sum()),
    )

    daily["negative_rate"] = np.round((daily["negative"] / daily["total"]) * 100, 2)
    daily["positive_rate"] = np.round((daily["positive"] / daily["total"]) * 100, 2)
    daily["neutral_rate"] = np.round((daily["neutral"] / daily["total"]) * 100, 2)
    daily["date_str"] = daily.index.astype(str)

    return daily


def detect_negative_spikes(
    daily_trends: Optional[pd.DataFrame],
    min_daily_volume: int = 30,
    min_days: int = 3,
    std_multiplier: float = 1.5,
    absolute_threshold_pct: float = 65.0,
) -> List[SpikeAlert]:
    """
    Detect statistical surges in negative feedback over chronological data.

    Enforces guards against small sample size artifacts:
    - Excludes days with fewer than min_daily_volume comments.
    - Requires at least min_days of historical observations to compute a reliable baseline.

    Args:
        daily_trends: Daily aggregated DataFrame from compute_daily_sentiment_trends.
        min_daily_volume: Minimum comment volume required on a day to evaluate.
        min_days: Minimum days needed for baseline standard deviation calculation.
        std_multiplier: Multiplier over standard deviation to define anomaly bound.
        absolute_threshold_pct: Minimum floor percentage to trigger an alert.

    Returns:
        List of SpikeAlert dataclasses.
    """
    if daily_trends is None or daily_trends.empty:
        return []

    # Guard 1: Filter out low-volume days to prevent alerting on small samples
    # (e.g. 2 negative comments out of 2 = 100% is not a true operational spike)
    qualified = daily_trends[daily_trends["total"] >= min_daily_volume].copy()
    if len(qualified) < min_days:
        return []

    alerts: List[SpikeAlert] = []
    for date_val, row in qualified.iterrows():
        neg_rate = float(row["negative_rate"])

        # Compute baseline from other days to avoid the spike distorting its own baseline
        other_days = qualified.drop(index=date_val)
        day_baseline_mean = float(other_days["negative_rate"].mean())
        day_baseline_std = float(other_days["negative_rate"].std())

        if np.isnan(day_baseline_std) or day_baseline_std < 1.0:
            day_baseline_std = 2.0

        dynamic_threshold = day_baseline_mean + (std_multiplier * day_baseline_std)
        effective_threshold = round(max(dynamic_threshold, absolute_threshold_pct), 2)

        if neg_rate >= effective_threshold:
            diff = round(neg_rate - day_baseline_mean, 2)
            severity = "HIGH" if neg_rate >= (day_baseline_mean + 2.0 * day_baseline_std) else "MEDIUM"
            msg = (
                f"Negative sentiment surged to {neg_rate:.1f}% on {date_val} "
                f"({int(row['negative']):,} negative of {int(row['total']):,} comments), "
                f"exceeding the baseline of {day_baseline_mean:.1f}% (+{diff}%). "
                f"Statistical spike flagged; verify with operational logs to determine root cause."
            )
            alerts.append(SpikeAlert(
                date=str(date_val),
                negative_count=int(row["negative"]),
                total_count=int(row["total"]),
                negative_rate=neg_rate,
                baseline_rate=round(day_baseline_mean, 2),
                threshold_rate=effective_threshold,
                severity=severity,
                message=msg,
            ))

    return alerts


def generate_insights(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "predicted_sentiment",
    product_col: str = "product",
    date_col: str = "date_parsed",
    min_daily_volume: int = 30,
) -> InsightReport:
    """
    Generate complete business intelligence and analytics report from sentiment predictions.

    Args:
        df: Input DataFrame with predicted sentiment column.
        text_col: Text column name.
        sentiment_col: Predicted sentiment column.
        product_col: Brand/product column name.
        date_col: Timestamp column name.
        min_daily_volume: Minimum comment threshold for daily spike alerts.

    Returns:
        InsightReport dataclass.
    """
    # 1. KPIs
    kpis = compute_kpis(df, sentiment_col=sentiment_col)

    # 2. Keywords
    top_complaint = extract_sentiment_keywords(df, "Negative", text_col=text_col, sentiment_col=sentiment_col, top_n=12)
    top_positive = extract_sentiment_keywords(df, "Positive", text_col=text_col, sentiment_col=sentiment_col, top_n=12)

    # 3. Product Comparisons
    prod_comp = compute_product_comparison(df, product_col=product_col, sentiment_col=sentiment_col)

    # 4. Temporal Trends & Spikes
    daily_df = compute_daily_sentiment_trends(df, date_col=date_col, sentiment_col=sentiment_col)
    spike_alerts = detect_negative_spikes(daily_df, min_daily_volume=min_daily_volume)

    # 5. Executive Summary
    summary_parts = [
        f"Analyzed {kpis['total_comments']:,} customer comments. Overall Negative feedback rate stands at "
        f"{kpis['negative_rate']}%, with Positive feedback at {kpis['positive_rate']}% "
        f"(Net Sentiment Score: {kpis['net_sentiment_score']}).",
    ]

    if top_complaint:
        top_neg_terms_str = ", ".join([f"'{t[0]}'" for t in top_complaint[:4]])
        summary_parts.append(f"Primary complaint drivers center on operational keywords: {top_neg_terms_str}.")

    if prod_comp and prod_comp.get("highest_negative_product"):
        summary_parts.append(
            f"Cross-product analysis shows '{prod_comp['highest_negative_product']}' received the highest "
            f"negative rate ({prod_comp['highest_negative_rate']}%), while '{prod_comp['highest_positive_product']}' "
            f"led in positive customer perception ({prod_comp['highest_positive_rate']}%)."
        )

    if spike_alerts:
        alert_dates = ", ".join([a.date for a in spike_alerts])
        summary_parts.append(f"Negative sentiment spike alert triggered on: {alert_dates}.")

    executive_summary = " ".join(summary_parts)

    temporal_dict = None
    if daily_df is not None:
        temporal_dict = {
            "daily_records": daily_df.to_dict(orient="index"),
            "evaluated_days": len(daily_df),
        }

    return InsightReport(
        kpis=kpis,
        top_complaint_terms=top_complaint,
        top_positive_terms=top_positive,
        product_comparison=prod_comp,
        temporal_trends=temporal_dict,
        spike_alerts=[a.to_dict() for a in spike_alerts],
        executive_summary=executive_summary,
        causality_disclaimer=CAUSALITY_DISCLAIMER,
    )
