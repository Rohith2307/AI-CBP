"""
Dark SaaS Plotly Theme Engine for Customer Sentiment Analysis Framework.

Applies a unified dark luxury theme, sleek typography, subtle gridlines,
transparent backgrounds, and high-contrast tooltips to all Plotly figures.
"""

from typing import Optional
import plotly.graph_objects as go
from dashboard.styles import (
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_SUBTLE,
    BG_SURFACE,
    PRIMARY_TEAL,
    ACCENT_VIOLET,
    CORAL_RED,
    AMBER_WARNING,
    MUTED_SLATE,
)

# Plotly dark theme color palette
DARK_THEME_PALETTE = [
    PRIMARY_TEAL,
    ACCENT_VIOLET,
    CORAL_RED,
    AMBER_WARNING,
    "#65B7D6",  # Cool blue
    MUTED_SLATE,
]


def apply_dark_theme(
    fig: Optional[go.Figure],
    title: Optional[str] = None,
    height: Optional[int] = None,
    show_legend: Optional[bool] = None,
) -> Optional[go.Figure]:
    """
    Transform any Plotly figure into a luxury dark SaaS interface component.

    Args:
        fig: Plotly Figure instance (returns None safely if fig is None).
        title: Optional override for chart title.
        height: Optional height in pixels.
        show_legend: Optional boolean toggle for legend visibility.

    Returns:
        Dark-themed Plotly Figure.
    """
    if fig is None:
        return None

    # Base layout overrides
    layout_updates = dict(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(
            family="Avenir Next, Segoe UI, sans-serif",
            color=TEXT_PRIMARY,
            size=12,
        ),
        hoverlabel=dict(
            bgcolor=BG_SURFACE,
            bordercolor=BORDER_SUBTLE,
            font=dict(
                family="Avenir Next, Segoe UI, sans-serif",
                color=TEXT_PRIMARY,
                size=12,
            ),
        ),
        margin=dict(l=40, r=30, t=50, b=40),
        transition=dict(duration=280, easing="cubic-in-out"),
    )

    if title:
        layout_updates["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=TEXT_PRIMARY),
            x=0.0,
            xanchor="left",
        )
    elif fig.layout.title.text:
        layout_updates["title"] = dict(
            text=f"<b>{fig.layout.title.text}</b>",
            font=dict(size=14, color=TEXT_PRIMARY),
            x=0.0,
            xanchor="left",
        )

    if height:
        layout_updates["height"] = height

    if show_legend is not None:
        layout_updates["showlegend"] = show_legend

    fig.update_layout(**layout_updates)

    # Style X & Y Axes with subtle gridlines and muted text
    fig.update_xaxes(
        gridcolor=BORDER_SUBTLE,
        zerolinecolor=BORDER_SUBTLE,
        linecolor=BORDER_SUBTLE,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_SECONDARY, size=12),
    )
    fig.update_yaxes(
        gridcolor=BORDER_SUBTLE,
        zerolinecolor=BORDER_SUBTLE,
        linecolor=BORDER_SUBTLE,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
        title_font=dict(color=TEXT_SECONDARY, size=12),
    )

    # Ensure legends are styled with muted text
    fig.update_layout(
        legend=dict(
            font=dict(color=TEXT_SECONDARY, size=11),
            bgcolor="rgba(0,0,0,0)",
        )
    )

    return fig
