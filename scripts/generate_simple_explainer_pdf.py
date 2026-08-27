"""
scripts/generate_simple_explainer_pdf.py
========================================================================================
Generates a simple, visual, easy-to-understand executive PDF guide:
"SmellPredict: How the Dual-Engine Architecture & Training Works"
"""

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Image
)
from reportlab.pdfgen import canvas

OUT_PDF = Path("reports/SmellPredict_Architecture_and_Training_Guide.pdf")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("reports/figures")

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY_TITLE   = colors.HexColor("#0f172a")   # Slate 900
DARK_BLUE    = colors.HexColor("#1e3a8a")   # Blue 900
ACCENT_BLUE  = colors.HexColor("#2563eb")   # Blue 600
TEAL         = colors.HexColor("#0f766e")   # Teal 700
SUCCESS      = colors.HexColor("#15803d")   # Green 700
SUCCESS_BG   = colors.HexColor("#f0fdf4")   # Green 50
AMBER        = colors.HexColor("#b45309")   # Amber 700
AMBER_BG     = colors.HexColor("#fffbeb")   # Amber 50
CHARCOAL     = colors.HexColor("#334155")   # Slate 700
MUTED        = colors.HexColor("#64748b")   # Slate 500

TBL_HDR_BG   = colors.HexColor("#e2e8f0")   # Slate 200 (Clean Light Gray)
TBL_HDR_TEXT = colors.HexColor("#0f172a")   # Deep Slate 900 (Bold Dark Text)
LIGHT_ROW    = colors.HexColor("#f8fafc")   # Slate 50
BORDER_COL   = colors.HexColor("#cbd5e1")   # Slate 300
BORDER_DARK  = colors.HexColor("#94a3b8")   # Slate 400
WHITE        = colors.HexColor("#ffffff")   # Pure White

# ── Numbered Canvas ───────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(DARK_BLUE)
            self.drawString(40, 820, "SMELLPREDICT: ARCHITECTURE & TRAINING GUIDE")
            
            self.setFont("Helvetica", 8)
            self.setFillColor(MUTED)
            self.drawRightString(555, 820, "Simple Technical Overview")
            
            self.setStrokeColor(BORDER_DARK)
            self.setLineWidth(0.6)
            self.line(40, 814, 555, 814)

        # Footer (All pages)
        self.setStrokeColor(BORDER_DARK)
        self.setLineWidth(0.6)
        self.line(40, 42, 555, 42)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(CHARCOAL)
        self.drawString(40, 30, "SmellPredict AI Engine · Architecture & Training Explained")
        
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(DARK_BLUE)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 30, page_str)
        
        self.restoreState()


# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

doc_title = ParagraphStyle(
    "DocTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=NAVY_TITLE,
    alignment=TA_LEFT,
    spaceAfter=4,
)

doc_subtitle = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=CHARCOAL,
    alignment=TA_LEFT,
    spaceAfter=8,
)

h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=15,
    textColor=DARK_BLUE,
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True,
)

h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=13,
    textColor=TEAL,
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True,
)

body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.3,
    leading=12.0,
    textColor=CHARCOAL,
    alignment=TA_JUSTIFY,
    spaceAfter=4,
)

bullet = ParagraphStyle(
    "Bullet",
    parent=body,
    leftIndent=12,
    spaceAfter=3,
)

callout_text = ParagraphStyle(
    "CalloutText",
    parent=body,
    fontSize=8.0,
    leading=11.5,
    textColor=CHARCOAL,
)

caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=7.2,
    leading=10.0,
    textColor=CHARCOAL,
    alignment=TA_CENTER,
    spaceBefore=2.5,
    spaceAfter=4,
)

# Table Styles
th_style = ParagraphStyle(
    "TH_Style",
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=10,
    textColor=TBL_HDR_TEXT,
    alignment=TA_CENTER,
)

th_style_left = ParagraphStyle(
    "TH_StyleLeft",
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=10,
    textColor=TBL_HDR_TEXT,
    alignment=TA_LEFT,
)

td_style = ParagraphStyle(
    "TD_Style",
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.5,
    textColor=CHARCOAL,
    alignment=TA_CENTER,
)

td_style_left = ParagraphStyle(
    "TD_StyleLeft",
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.5,
    textColor=CHARCOAL,
    alignment=TA_LEFT,
)

td_style_bold = ParagraphStyle(
    "TD_StyleBold",
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=NAVY_TITLE,
    alignment=TA_CENTER,
)

td_style_green = ParagraphStyle(
    "TD_StyleGreen",
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=SUCCESS,
    alignment=TA_CENTER,
)

def make_box(text: str, title: str = "KEY TAKEAWAY", bg_color=LIGHT_ROW, border_color=ACCENT_BLUE) -> Table:
    content = [
        Paragraph(f"<b>{title}</b>", ParagraphStyle("CTitle", parent=callout_text, fontName="Helvetica-Bold", textColor=border_color, spaceAfter=2)),
        Paragraph(text, callout_text)
    ]
    t = Table([[content]], colWidths=[515])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BOX', (0, 0), (-1, -1), 0.8, border_color),
        ('PADDING', (0, 0), (-1, -1), 5.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=55,
        bottomMargin=55,
    )
    story = []

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1: WHAT SMELLPREDICT DOES & WHY 2 ENGINES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("SmellPredict: How the System & Training Works", doc_title))
    story.append(Paragraph(
        "A Simple, Visual Guide to Dual-Engine ML Defect Prediction, Training Data, and Real-Time Inference.",
        doc_subtitle
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT_BLUE, spaceAfter=6, spaceBefore=0))

    story.append(Paragraph("1. The Core Question: Why Do We Have 2 Engines?", h1_style))
    story.append(Paragraph(
        "Software defect prediction has to run in two completely different environments. One single model cannot do both without failing:",
        body
    ))

    two_engine_box = (
        "• <b>Environment 1: The Live Browser/IDE Editor (Engine A)</b><br/>"
        "When you open a file or type code in the editor, the browser only has the raw text of that single file. "
        "It does <b>not</b> have 5 years of Git history, contributor logs, or co-change maps. <b>Engine A is built specifically for this</b>: "
        "it needs zero Git history and predicts defect risk using 34 pure code structure and smell metrics in real-time (&lt;80ms).<br/><br/>"
        "• <b>Environment 2: CI/CD & Automated Pull Request Review (Engine B)</b><br/>"
        "When a pull request is opened or a build runs in GitHub Actions, the entire Git repository is available. "
        "<b>Engine B runs here</b>: it combines code structure with 39 Git churn and developer ownership features for maximum enterprise accuracy."
    )
    story.append(make_box(two_engine_box, title="WHY 2 ENGINES INSTEAD OF 1?", bg_color=LIGHT_ROW, border_color=ACCENT_BLUE))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Side-by-Side Comparison: Engine A vs. Engine B", h2_style))
    
    comp_headers = [
        Paragraph("Dimension", th_style_left),
        Paragraph("⚡ Engine A (Static AST)", th_style),
        Paragraph("🏢 Engine B (Full Enterprise)", th_style)
    ]
    comp_rows = [
        [
            Paragraph("<b>Primary Purpose</b>", td_style_left),
            Paragraph("<b>Live IDE Editor / Cold-Start Files</b>", td_style_bold),
            Paragraph("<b>CI/CD Pipelines / PR Review Bot</b>", td_style_bold),
        ],
        [
            Paragraph("<b>Git History Needed?</b>", td_style_left),
            Paragraph("<b>ZERO (Works on raw text)</b>", td_style_green),
            Paragraph("Mandatory (Uses 5-year churn)", td_style),
        ],
        [
            Paragraph("<b>Features Used</b>", td_style_left),
            Paragraph("<b>34 AST & Smell Features</b>", td_style),
            Paragraph("<b>73 Multidimensional Features</b>", td_style),
        ],
        [
            Paragraph("<b>Inference Speed</b>", td_style_left),
            Paragraph("<b>&lt; 80 ms (Instant typing feedback)</b>", td_style_green),
            Paragraph("~250 ms (Parses Git commit log)", td_style),
        ],
        [
            Paragraph("<b>Accuracy (ROC-AUC)</b>", td_style_left),
            Paragraph("<b>0.7113 (96.1% retention of Engine B)</b>", td_style_green),
            Paragraph("<b>0.7400 (Maximum full accuracy)</b>", td_style_bold),
        ],
        [
            Paragraph("<b>Empirical Lift</b>", td_style_left),
            Paragraph("<b>+0.1990 above random baseline</b>", td_style_green),
            Paragraph("<b>+0.2314 above random baseline</b>", td_style_green),
        ],
    ]

    t_comp = Table([comp_headers] + comp_rows, colWidths=[135, 190, 190])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TBL_HDR_BG),
        ('LINEBELOW', (0, 0), (-1, 0), 1.2, ACCENT_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_ROW]),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 6))

    # Page Break for Training Pipeline
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2: HOW THE DATA WAS MINED & HOW MODELS WERE TRAINED
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. How the Model Was Trained: Data & Ground Truth", h1_style))
    story.append(Paragraph(
        "To train the AI models, we mined <b>23,170 real code snapshots</b> across <b>48 major open-source Python repositories</b> "
        "(Django, Flask, Celery, Pytest, Requests, FastAPI, Black, Aiohttp, Scikit-Learn, Pydantic, etc.).",
        body
    ))

    training_box = (
        "<b>Step 1: Historical Data Extraction</b><br/>"
        "We extracted every commit in the history of all 48 repositories. For each file at that point in time, we parsed its exact AST "
        "and calculated lines of code, cyclomatic complexity, nesting depth, and code smells.<br/><br/>"
        "<b>Step 2: Ground Truth Labeling (SZZ Bug Linking)</b><br/>"
        "• <b>Defective (1):</b> If a later commit in that repository modified this file and referenced a bug fix "
        "(e.g., <code>fix #102</code>, <code>bug</code>, <code>patch</code>, <code>CVE</code>), the snapshot is labeled <b>Defective</b>.<br/>"
        "• <b>Clean (0):</b> If the file remained stable without needing corrective bug fixes, it is labeled <b>Clean</b>.<br/><br/>"
        "<b>Step 3: Training with Platt Sigmoid Scaling</b><br/>"
        "We trained gradient-boosted decision trees (LightGBM) with monotonicity constraints (higher complexity always increases risk) "
        "and calibrated the outputs using <b>Platt Sigmoid scaling</b> to ensure probabilities are smooth, realistic, and bounded."
    )
    story.append(make_box(training_box, title="3-STEP TRAINING & GROUND TRUTH PIPELINE", bg_color=SUCCESS_BG, border_color=SUCCESS))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Visual Proof: Model Calibration (No Extreme 0% or 100% Bugs)", h1_style))
    story.append(Paragraph(
        "A critical achievement of the upgraded system is <b>Platt Sigmoid Calibration</b>. The legacy model suffered from "
        "tail collapse (falsely predicting hard 0% or 100%). The upgraded model produces smooth, realistic probabilities:",
        body
    ))

    cal_img = FIG_DIR / "proof1_calibration_comparison.png"
    if cal_img.exists():
        story.append(Image(str(cal_img), width=480, height=215))
        story.append(Paragraph("<b>Figure 1:</b> Probability calibration proof. The upgraded model (green line) stays smoothly bounded between 16% and 78%, eliminating false 0% or 100% pinning.", caption_style))

    # Page Break for Real-Time Inference
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3: REAL-TIME INFERENCE & PROTON.PY WALKTHROUGH
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. How Live Evaluation Works: The 'Proton.py' Example", h1_style))
    story.append(Paragraph(
        "When you open a file like <b>`Proton.py`</b> in the live editor, here is the exact 5-step process that occurs in under 80 milliseconds:",
        body
    ))

    proton_box = (
        "<b>1. Editor Keystroke Sent:</b> The Monaco web editor sends the Python text to the API.<br/>"
        "<b>2. Language Filter:</b> The API verifies it is a Python file (<code>.py</code>). Polyglot and media files return <code>risk: null</code>.<br/>"
        "<b>3. Engine A Selected:</b> Because no Git history is attached to the live browser buffer, the router automatically selects <b>Engine A</b>.<br/>"
        "<b>4. Empirical CDF Lookup:</b> Engine A parses <code>Proton.py</code> and finds:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Lines of Code:</b> 253 (75th percentile of Python codebases)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Max Cyclomatic Complexity:</b> 35.0 (97th percentile)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Max Nesting Depth:</b> 19 levels (99th percentile)<br/>"
        "<b>5. Calibrated Risk Output:</b> Based on this high complexity and deep nesting, Engine A computes <b>65% Defect Probability</b>, displaying <b>MEDIUM RISK · 65%</b> in the top bar."
    )
    story.append(make_box(proton_box, title="REAL-TIME INFERENCE PIPELINE (UNDER 80ms)", bg_color=AMBER_BG, border_color=AMBER))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5. Visual Proof: Engine A vs Engine B Retention", h1_style))
    story.append(Paragraph(
        "Can Engine A really be trusted without Git history? Yes! Cross-validation proofs show Engine A retains <b>95% to 98%</b> "
        "of the predictive capacity of Engine B:",
        body
    ))

    ret_img = FIG_DIR / "proof4_dual_engine_retention.png"
    if ret_img.exists():
        story.append(Image(str(ret_img), width=480, height=195))
        story.append(Paragraph("<b>Figure 2:</b> Engine A vs Engine B retention proof. Engine A preserves 96.1% of ROC-AUC and 97.9% of top-20% inspection precision with zero Git dependencies.", caption_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Summary Summary Checklist", h2_style))
    summary_box = (
        "✔ <b>Engine A</b> = Fast live editor model (34 AST metrics, 0 Git needed, &lt;80ms latency).<br/>"
        "✔ <b>Engine B</b> = Full CI/CD gating model (73 features including commit churn and author ownership).<br/>"
        "✔ <b>Training Data</b> = 23,170 snapshots labeled using historical bug fixes from 48 repositories.<br/>"
        "✔ <b>Strict Isolation</b> = Python-only ML defect scoring; images and polyglot files never get false risks."
    )
    story.append(make_box(summary_box, title="EXECUTIVE SUMMARY CHECKLIST", bg_color=SUCCESS_BG, border_color=SUCCESS))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated simple explainer PDF: {OUT_PDF.resolve()}")

if __name__ == "__main__":
    build_pdf()
