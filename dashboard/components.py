"""
Reusable UI Components for Customer Sentiment Analysis Framework Dashboard.

Provides modular HTML/CSS generators for luxury KPI metric cards,
sentiment badges, section headers, statistical spike alert cards,
and model benchmark cards.
"""

from typing import Any, Dict, Optional
import streamlit as st
from dashboard.styles import (
    PRIMARY_TEAL,
    ACCENT_VIOLET,
    CORAL_RED,
    AMBER_WARNING,
    MUTED_SLATE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


def get_sentiment_badge_html(sentiment: str) -> str:
    """Return an HTML sentiment badge pill with theme styling."""
    normalized = str(sentiment).strip().capitalize()
    if normalized == "Positive":
        css_class = "sentiment-badge-pos"
        icon = "●"
    elif normalized == "Negative":
        css_class = "sentiment-badge-neg"
        icon = "●"
    else:
        css_class = "sentiment-badge-neu"
        icon = "●"

    return f'<span class="sentiment-badge {css_class}">{icon} {normalized}</span>'


def render_sentiment_badge(sentiment: str) -> None:
    """Render an HTML sentiment badge directly in Streamlit."""
    st.markdown(get_sentiment_badge_html(sentiment), unsafe_allow_html=True)


def get_kpi_card_html(
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: Optional[str] = None,
    badge_type: str = "neutral",
    icon: str = "📊",
) -> str:
    """
    Generate an HTML string for a luxury glassmorphism KPI card.

    Args:
        title: Metric uppercase label (e.g. 'TOTAL COMMENTS').
        value: Primary metric number (e.g. '14,427' or '77.97%').
        subtitle: Explanatory micro-copy.
        badge_text: Optional delta or tag (e.g. '+15.9%' or 'Active').
        badge_type: Tone for badge ('positive', 'negative', 'warning', 'violet', 'neutral').
        icon: Unicode icon or emoji.
    """
    badge_colors = {
        "positive": (f"rgba(6, 214, 160, 0.15)", PRIMARY_TEAL),
        "negative": (f"rgba(239, 68, 68, 0.15)", CORAL_RED),
        "warning": (f"rgba(245, 158, 11, 0.15)", AMBER_WARNING),
        "violet": (f"rgba(139, 92, 246, 0.15)", ACCENT_VIOLET),
        "neutral": (f"rgba(100, 116, 139, 0.18)", TEXT_SECONDARY),
    }
    bg_color, text_color = badge_colors.get(badge_type.lower(), badge_colors["neutral"])

    badge_html = ""
    if badge_text:
        badge_html = (
            f'<span style="background-color: {bg_color}; color: {text_color}; '
            f'padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; '
            f'letter-spacing: 0.02em;">{badge_text}</span>'
        )

    icon_bg_colors = {
        "positive": "rgba(6, 214, 160, 0.15)",
        "negative": "rgba(239, 68, 68, 0.15)",
        "warning": "rgba(245, 158, 11, 0.15)",
        "violet": "rgba(139, 92, 246, 0.15)",
        "neutral": "rgba(100, 116, 139, 0.15)",
    }
    icon_bg = icon_bg_colors.get(badge_type.lower(), icon_bg_colors["neutral"])

    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-header">
                <span class="kpi-label">{title}</span>
                <div class="kpi-icon-pill" style="background-color: {icon_bg};">
                    {icon}
                </div>
            </div>
            <div class="kpi-value">{value}</div>
        </div>
        <div class="kpi-footer">
            {badge_html}
            <span>{subtitle}</span>
        </div>
    </div>
    """


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: Optional[str] = None,
    badge_type: str = "neutral",
    icon: str = "📊",
) -> None:
    """Render a KPI card directly in Streamlit."""
    st.markdown(
        get_kpi_card_html(title, value, subtitle, badge_text, badge_type, icon),
        unsafe_allow_html=True,
    )


def get_section_header_html(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> str:
    """Generate HTML for a styled section header with optional badge."""
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="background-color: rgba(139, 92, 246, 0.18); color: {ACCENT_VIOLET}; '
            f'padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; '
            f'text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid rgba(139, 92, 246, 0.3);">'
            f'{badge}</span>'
        )

    sub_html = f'<p class="section-header-subtitle">{subtitle}</p>' if subtitle else ""

    return f"""
    <div class="section-header-wrap">
        <div>
            <h3 class="section-header-title">{title}</h3>
            {sub_html}
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Render a section header directly in Streamlit."""
    st.markdown(get_section_header_html(title, subtitle, badge), unsafe_allow_html=True)


def get_spike_alert_html(
    date: str,
    message: str,
    severity: str = "CRITICAL",
    daily_count: Optional[int] = None,
    baseline_mean: Optional[float] = None,
) -> str:
    """Generate HTML for a statistical negative spike alert card."""
    stats_line = ""
    if daily_count is not None and baseline_mean is not None:
        stats_line = (
            f'<div style="font-size: 0.8rem; color: {TEXT_SECONDARY}; margin-top: 6px;">'
            f'<b>Observed:</b> {daily_count:,} negative grievances | '
            f'<b>Baseline Average:</b> {baseline_mean:.1f} comments/day'
            f'</div>'
        )

    return f"""
    <div class="spike-card">
        <div class="spike-card-header">
            <span style="font-size: 0.95rem; font-weight: 700; color: {CORAL_RED};">
                🚨 Statistical Spike Alert — {date}
            </span>
            <span class="spike-badge">{severity}</span>
        </div>
        <div style="font-size: 0.88rem; color: {TEXT_PRIMARY}; line-height: 1.45;">
            {message}
        </div>
        {stats_line}
    </div>
    """


def render_spike_alert(
    date: str,
    message: str,
    severity: str = "CRITICAL",
    daily_count: Optional[int] = None,
    baseline_mean: Optional[float] = None,
) -> None:
    """Render a statistical spike alert directly in Streamlit."""
    st.markdown(
        get_spike_alert_html(date, message, severity, daily_count, baseline_mean),
        unsafe_allow_html=True,
    )


def get_callout_html(message: str, callout_type: str = "info") -> str:
    """Generate an info, success, or warning callout card."""
    css_class = f"{callout_type.lower()}-callout"
    return f'<div class="{css_class}">{message}</div>'


def render_callout(message: str, callout_type: str = "info") -> None:
    """Render a callout card directly in Streamlit."""
    st.markdown(get_callout_html(message, callout_type), unsafe_allow_html=True)
