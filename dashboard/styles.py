"""
Global Design System, Color Tokens, and CSS Styles for the Sentiment Framework.

Provides modern SaaS styling, luxury dark-theme palette tokens,
and scoped CSS injection for Streamlit widgets and containers.
"""

import streamlit as st

# =========================================================================
# Design System Color Tokens
# =========================================================================

BG_CANVAS = "#0B1120"        # Deep midnight canvas background
BG_SURFACE = "#10192D"       # Elevated container & sidebar surface
BG_CARD = "#162238"          # Glassmorphic card surface
BG_CARD_HOVER = "#1B2A47"    # Card hover elevation
BORDER_SUBTLE = "#23314E"    # Card & widget borders
BORDER_HOVER = "#3B527E"     # Hover border accent
BORDER_FOCUS = "#06D6A0"     # Input focus ring

# Core Accents
PRIMARY_TEAL = "#06D6A0"     # Positive sentiment, primary actions
ACCENT_VIOLET = "#8B5CF6"    # Machine learning, neural, neutral inquiry
CORAL_RED = "#EF4444"        # Negative sentiment, grievances, critical spikes
AMBER_WARNING = "#F59E0B"    # Duplicates, warnings, medium spikes
MUTED_SLATE = "#64748B"      # Subtitles, inactive states, secondary indicators

# Typography Colors
TEXT_PRIMARY = "#F8FAFC"     # Headers, primary values (crisp off-white)
TEXT_SECONDARY = "#94A3B8"   # Field labels, captions, metadata
TEXT_MUTED = "#64748B"       # Subtle micro-copy, timestamps

# Consistent Sentiment Color Mapping
SENTIMENT_COLORS = {
    "Positive": PRIMARY_TEAL,
    "Neutral": ACCENT_VIOLET,
    "Negative": CORAL_RED,
}


# =========================================================================
# Custom Scoped CSS Definition
# =========================================================================

GLOBAL_CSS = f"""
/* Google Fonts Import for Inter & JetBrains Mono */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: {TEXT_PRIMARY};
}}

code, pre, [class*="stCode"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Main Application Container Spacing */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    max-width: 1300px;
}}

/* Custom Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: {BG_SURFACE};
    border-right: 1px solid {BORDER_SUBTLE};
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
}}

/* Sidebar Brand Header */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0 16px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}

.sidebar-brand-icon {{
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, {PRIMARY_TEAL} 0%, {ACCENT_VIOLET} 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 12px rgba(6, 214, 160, 0.25);
}}

.sidebar-brand-text h2 {{
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.02em;
}}

.sidebar-brand-text span {{
    font-size: 0.75rem;
    color: {PRIMARY_TEAL};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Sidebar Navigation Radio Buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background-color: {BG_CARD};
    border-color: {BORDER_SUBTLE};
}}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
    background: linear-gradient(90deg, rgba(6, 214, 160, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%);
    border: 1px solid rgba(6, 214, 160, 0.35);
    color: {PRIMARY_TEAL} !important;
    font-weight: 600;
}}

/* Glassmorphism Metric / KPI Card */
.kpi-card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}}

.kpi-card:hover {{
    border-color: {BORDER_HOVER};
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transform: translateY(-2px);
}}

.kpi-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}}

.kpi-label {{
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEXT_SECONDARY};
}}

.kpi-icon-pill {{
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}}

.kpi-value {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: {TEXT_PRIMARY};
    line-height: 1.15;
    margin-bottom: 6px;
}}

.kpi-footer {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: {TEXT_MUTED};
}}

/* Section Header Component */
.section-header-wrap {{
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}

.section-header-title {{
    font-size: 1.45rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}}

.section-header-subtitle {{
    font-size: 0.88rem;
    color: {TEXT_SECONDARY};
    margin: 0;
}}

/* Sentiment Badge Pills */
.sentiment-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}

.sentiment-badge-pos {{
    background-color: rgba(6, 214, 160, 0.15);
    color: {PRIMARY_TEAL};
    border: 1px solid rgba(6, 214, 160, 0.4);
}}

.sentiment-badge-neu {{
    background-color: rgba(139, 92, 246, 0.15);
    color: {ACCENT_VIOLET};
    border: 1px solid rgba(139, 92, 246, 0.4);
}}

.sentiment-badge-neg {{
    background-color: rgba(239, 68, 68, 0.15);
    color: {CORAL_RED};
    border: 1px solid rgba(239, 68, 68, 0.4);
}}

/* General Content Surface Card */
.surface-card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
}}

/* Statistical Spike Alert Cards */
.spike-card {{
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border: 1px solid rgba(239, 68, 68, 0.45);
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(22, 34, 56, 0.95) 100%);
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);
}}

.spike-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}}

.spike-badge {{
    background-color: {CORAL_RED};
    color: white;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding-bottom: 6px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 8px;
    color: {TEXT_SECONDARY};
    font-weight: 600;
    font-size: 0.9rem;
    padding: 8px 16px;
    border: 1px solid transparent;
}}

.stTabs [aria-selected="true"] {{
    background-color: {BG_CARD} !important;
    border: 1px solid {BORDER_SUBTLE} !important;
    color: {PRIMARY_TEAL} !important;
}}

/* Custom Tooltip Notice / Micro-Card */
.info-callout {{
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-left: 4px solid {ACCENT_VIOLET};
    border-radius: 8px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 0.88rem;
    margin: 12px 0;
}}

/* Custom Success Callout */
.success-callout {{
    background: rgba(6, 214, 160, 0.08);
    border: 1px solid rgba(6, 214, 160, 0.3);
    border-left: 4px solid {PRIMARY_TEAL};
    border-radius: 8px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 0.88rem;
    margin: 12px 0;
}}

/* Custom Warning Callout */
.warning-callout {{
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid {AMBER_WARNING};
    border-radius: 8px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 0.88rem;
    margin: 12px 0;
}}
"""


def inject_custom_styles():
    """Inject scoped CSS into Streamlit document head."""
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)
