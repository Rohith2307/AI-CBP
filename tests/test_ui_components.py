"""
Unit tests for dashboard UI styling, dark chart theme engine, and components.

Verifies that design system color tokens, Plotly dark theme overrides,
and reusable HTML component generators function properly and safely.
"""

import plotly.graph_objects as go
import pytest

from dashboard.styles import (
    BG_CANVAS,
    BG_CARD,
    BG_SURFACE,
    PRIMARY_TEAL,
    ACCENT_VIOLET,
    CORAL_RED,
    AMBER_WARNING,
    SENTIMENT_COLORS,
    GLOBAL_CSS,
)
from dashboard.chart_theme import apply_dark_theme
from dashboard.components import (
    get_sentiment_badge_html,
    get_kpi_card_html,
    get_section_header_html,
    get_spike_alert_html,
    get_callout_html,
)


def test_style_color_tokens():
    """Verify that design tokens match the SaaS dark palette specifications."""
    assert BG_CANVAS == "#0B1120"
    assert BG_SURFACE == "#10192D"
    assert BG_CARD == "#162238"
    assert PRIMARY_TEAL == "#06D6A0"
    assert ACCENT_VIOLET == "#8B5CF6"
    assert CORAL_RED == "#EF4444"
    assert AMBER_WARNING == "#F59E0B"

    # Sentiment color mapping
    assert SENTIMENT_COLORS["Positive"] == PRIMARY_TEAL
    assert SENTIMENT_COLORS["Neutral"] == ACCENT_VIOLET
    assert SENTIMENT_COLORS["Negative"] == CORAL_RED


def test_global_css_rules():
    """Verify that global CSS contains essential widget and card styling selectors."""
    assert ".kpi-card" in GLOBAL_CSS
    assert ".sidebar-brand" in GLOBAL_CSS
    assert ".sentiment-badge" in GLOBAL_CSS
    assert ".spike-card" in GLOBAL_CSS
    assert "Inter" in GLOBAL_CSS


def test_chart_theme_apply_dark():
    """Verify that apply_dark_theme overrides figure layout with transparent dark theme."""
    # Test safe handling of None
    assert apply_dark_theme(None) is None

    # Test transformation of a basic figure
    fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[10, 20])])
    fig.update_layout(title="Initial Title", template="plotly_white")

    themed_fig = apply_dark_theme(fig, title="Custom Title", height=350)
    assert themed_fig is not None
    assert themed_fig.layout.paper_bgcolor == "rgba(0, 0, 0, 0)"
    assert themed_fig.layout.plot_bgcolor == "rgba(0, 0, 0, 0)"
    assert themed_fig.layout.height == 350
    assert "Custom Title" in themed_fig.layout.title.text
    assert themed_fig.layout.font.color == "#F8FAFC"


def test_sentiment_badge_html_generation():
    """Verify sentiment badge generator outputs correct classes and labels."""
    pos_html = get_sentiment_badge_html("Positive")
    assert "sentiment-badge-pos" in pos_html
    assert "Positive" in pos_html

    neu_html = get_sentiment_badge_html("Neutral")
    assert "sentiment-badge-neu" in neu_html
    assert "Neutral" in neu_html

    neg_html = get_sentiment_badge_html("Negative")
    assert "sentiment-badge-neg" in neg_html
    assert "Negative" in neg_html


def test_kpi_card_html_generation():
    """Verify KPI card HTML incorporates all metrics, badges, and subtitles."""
    html = get_kpi_card_html(
        title="TEST METRIC",
        value="77.97%",
        subtitle="Evaluated on 2,165 test records",
        badge_text="+2.5%",
        badge_type="positive",
        icon="🎯",
    )
    assert "TEST METRIC" in html
    assert "77.97%" in html
    assert "Evaluated on 2,165 test records" in html
    assert "+2.5%" in html
    assert "kpi-card" in html


def test_section_header_html_generation():
    """Verify section header HTML renders title, subtitle, and badge."""
    html = get_section_header_html(
        title="Model Benchmark",
        subtitle="Comparative analysis across algorithms",
        badge="Stage 7",
    )
    assert "Model Benchmark" in html
    assert "Comparative analysis across algorithms" in html
    assert "Stage 7" in html


def test_spike_alert_html_generation():
    """Verify spike alert card HTML formatting and statistical details."""
    html = get_spike_alert_html(
        date="2015-02-23",
        message="Negative comments surged 135% above average.",
        severity="HIGH",
        daily_count=450,
        baseline_mean=192.4,
    )
    assert "2015-02-23" in html
    assert "HIGH" in html
    assert "450" in html
    assert "192.4" in html
    assert "spike-card" in html


def test_callout_html_generation():
    """Verify callout HTML for different notification levels."""
    info_html = get_callout_html("Information message", callout_type="info")
    assert "info-callout" in info_html
    assert "Information message" in info_html

    warn_html = get_callout_html("Warning message", callout_type="warning")
    assert "warning-callout" in warn_html
    assert "Warning message" in warn_html


def test_global_css_includes_product_shell_and_feedback_reader_patterns():
    """Verify the redesign exposes reusable shell and customer-voice styles."""
    assert ".page-hero" in GLOBAL_CSS
    assert ".dataset-status" in GLOBAL_CSS
    assert ".feedback-record" in GLOBAL_CSS
    assert "focus-visible" in GLOBAL_CSS
