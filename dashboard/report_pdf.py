"""PDF rendering for the executive insights export."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.insights import InsightReport


INK = colors.HexColor("#172338")
TEAL = colors.HexColor("#168C78")
MUTED = colors.HexColor("#58677C")
PALE = colors.HexColor("#F1F6F7")
RULE = colors.HexColor("#D8E1E8")


def _pdf_text(value: object) -> str:
    """Escape report text and replace glyphs unavailable in the built-in fonts."""
    text = str(value)
    replacements = {
        "—": "-",
        "–": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "%": "%",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return escape(text.encode("latin-1", "replace").decode("latin-1"))


def create_executive_report_pdf(insights: InsightReport) -> bytes:
    """Render current insight results as a compact, printable PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.7 * inch,
        title="SENTIMENT.AI Executive Intelligence Dossier",
        author="SENTIMENT.AI",
        pageCompression=0,
    )

    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, alignment=TA_LEFT, textColor=INK,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "ReportSection", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=TEAL, spaceBefore=13,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=base["BodyText"], fontName="Helvetica",
        fontSize=9, leading=13, textColor=INK, spaceAfter=5,
    )
    small_style = ParagraphStyle(
        "ReportSmall", parent=body_style, fontSize=8, leading=11,
        textColor=MUTED,
    )
    metric_value_style = ParagraphStyle(
        "MetricValue", parent=body_style, fontName="Helvetica-Bold",
        fontSize=15, leading=18, textColor=INK, spaceAfter=2,
    )

    story = [
        Paragraph("SENTIMENT.AI", ParagraphStyle(
            "Brand", parent=small_style, fontName="Helvetica-Bold",
            fontSize=8, textColor=TEAL, spaceAfter=4,
        )),
        Paragraph("Executive Intelligence Dossier", title_style),
        Paragraph("Evidence-led summary of the active feedback dataset", small_style),
        Spacer(1, 10),
        Paragraph("Executive Summary", section_style),
        Paragraph(_pdf_text(insights.executive_summary), body_style),
        Paragraph("Key Performance Indicators", section_style),
    ]

    kpis = insights.kpis
    metric_items = [
        ("Total feedback", f"{int(kpis['total_comments']):,}"),
        ("Negative", f"{kpis['negative_rate']}%"),
        ("Positive", f"{kpis['positive_rate']}%"),
        ("Neutral", f"{kpis['neutral_rate']}%"),
        ("Net sentiment score", str(kpis["net_sentiment_score"])),
    ]
    cells = [
        Paragraph(f"{_pdf_text(value)}<br/><font size='8' color='#58677C'>{_pdf_text(label)}</font>", metric_value_style)
        for label, value in metric_items
    ]
    rows = [cells[index:index + 3] for index in range(0, len(cells), 3)]
    if len(rows[-1]) < 3:
        rows[-1].extend([""] * (3 - len(rows[-1])))
    metrics_table = Table(rows, colWidths=[2.25 * inch] * 3, hAlign="LEFT")
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([metrics_table, Paragraph("Top Complaint Drivers", section_style)])

    def add_terms(terms: list[tuple[str, int]], empty_message: str) -> None:
        if terms:
            for term, count in terms:
                story.append(Paragraph(f"- {_pdf_text(term)} ({int(count):,})", body_style))
        else:
            story.append(Paragraph(empty_message, small_style))

    add_terms(insights.top_complaint_terms, "No complaint terms available for this dataset.")
    story.append(Paragraph("Top Praise Drivers", section_style))
    add_terms(insights.top_positive_terms, "No praise terms available for this dataset.")
    story.extend([
        Paragraph("Methodological Note", section_style),
        Paragraph(_pdf_text(insights.causality_disclaimer), body_style),
    ])

    def draw_page(canvas, document) -> None:
        canvas.saveState()
        width, height = letter
        canvas.setFillColor(INK)
        canvas.rect(0, height - 0.22 * inch, width, 0.22 * inch, fill=1, stroke=0)
        canvas.setStrokeColor(RULE)
        canvas.line(0.7 * inch, 0.48 * inch, width - 0.7 * inch, 0.48 * inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.7 * inch, 0.32 * inch, "SENTIMENT.AI  |  Customer Intelligence")
        canvas.drawRightString(width - 0.7 * inch, 0.32 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()