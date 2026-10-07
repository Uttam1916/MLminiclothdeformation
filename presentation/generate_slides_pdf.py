"""
Generates the Presentation Slide Deck PDF for the Mini-Project Review & Demo.
Creates a professional landscape presentation with embedded figures and tables.
"""

import os
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class SlideCanvas(canvas.Canvas):
    """Draws slide footers and slide numbers."""

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
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, total_pages):
        self.saveState()
        # Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#777777"))
        self.drawString(40, 22, "UE24CS352A ML Mini-Project | 3D Garment Deformation from Human Pose")
        self.drawRightString(792 - 40, 22, f"Slide {self._pageNumber} of {total_pages}")
        self.setStrokeColor(colors.HexColor("#e0e0e0"))
        self.setLineWidth(0.6)
        self.line(40, 32, 792 - 40, 32)
        self.restoreState()


def build_presentation_pdf(output_pdf: str = "presentation/presentation.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(letter),  # 792 x 612 pt
        leftMargin=40,
        rightMargin=40,
        topMargin=35,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_slide_title = ParagraphStyle(
        "TitleSlideTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1a365d"),
        alignment=1,
    )
    title_slide_sub = ParagraphStyle(
        "TitleSlideSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4a5568"),
        alignment=1,
    )
    slide_title = ParagraphStyle(
        "SlideTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "SlideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "SlideBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2d3748"),
        leftIndent=15,
        spaceAfter=4,
    )
    caption_style = ParagraphStyle(
        "SlideCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#718096"),
        spaceBefore=3,
    )

    slides = []

    # Slide 1: Title Slide
    slides.append(Spacer(1, 80))
    slides.append(Paragraph("Data-Driven 3D Garment Deformation from Human Pose", title_slide_title))
    slides.append(Spacer(1, 15))
    slides.append(Paragraph("UE24CS352A - Machine Learning Mini-Project Final Review & Demonstration", title_slide_sub))
    slides.append(Spacer(1, 10))
    slides.append(HRFlowable(width="60%", thickness=1.5, color=colors.HexColor("#2b5c8f"), spaceBefore=5, spaceAfter=15))
    slides.append(Paragraph("<b>Reference Study:</b> Xue & Wu (Stanford CS 229) &nbsp;|&nbsp; <b>Alternative Dataset:</b> TailorNet (CVPR 2020)<br/><b>Team:</b> 2-Member Student Team", title_slide_sub))
    slides.append(PageBreak())

    # Slide 2: Problem Statement & Motivation
    slides.append(Paragraph("1. Problem Statement & Motivation", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=12))
    slides.append(Paragraph("<b>The Real-Time Cloth Challenge:</b>", body_style))
    slides.append(Paragraph("• Interactive 3D applications (gaming, VR, character animation, and virtual try-on) demand realistic clothing behavior.", bullet_style))
    slides.append(Paragraph("• <b>The Bottleneck:</b> Traditional physics-based cloth simulation (PDE solvers like PhysBAM using mass-spring/finite element models) requires solving complex time-stepping and contact equations, taking <b>~180 seconds per frame</b>.", bullet_style))
    slides.append(Paragraph("• <b>The Solution:</b> A data-driven machine learning model that maps human body pose parameters directly to 3D garment mesh displacements.", bullet_style))
    slides.append(Paragraph("• <b>Our Goal:</b> Reproduce the core methodology of the reference paper using the public TailorNet dataset, and introduce architectural improvements that outperform the paper's baseline in accuracy and stability.", bullet_style))
    slides.append(PageBreak())

    # Slide 3: Original Paper vs Dataset Choice
    slides.append(Paragraph("2. Original Paper & Dataset Adaptation", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=12))
    slides.append(Paragraph("<b>Original Paper (Xue & Wu, 2021):</b>", body_style))
    slides.append(Paragraph("• Focused on predicting loose coat deformation from 14 skeletal joints (56 quaternion dimensions).", bullet_style))
    slides.append(Paragraph("• Used proprietary synthetic PhysBAM coat simulation data (9,813 poses). <i>Dataset is not publicly available.</i>", bullet_style))
    slides.append(Spacer(1, 8))
    slides.append(Paragraph("<b>Public Alternative: TailorNet Dataset (CVPR 2020, MPI):</b>", body_style))
    slides.append(Paragraph("• TailorNet provides physically simulated 3D garment displacements across SMPL body poses.", bullet_style))
    slides.append(Paragraph("• We utilized the curated <code>t-shirt_female</code> dataset (110 frames, 7,702 vertices, 15,180 triangular faces).", bullet_style))
    slides.append(Paragraph("• <b>Pose Input:</b> Converted 24 SMPL joint rotations into 96-dimensional unit quaternions ($w,x,y,z$) to prevent gimbal lock.", bullet_style))
    slides.append(Paragraph("• <b>Deformation Output:</b> 3D vertex displacements from canonical t-shirt template ($7,702 \times 3 = 23,106$ dims).", bullet_style))
    slides.append(PageBreak())

    # Slide 4: PCA Dimensionality Reduction
    slides.append(Paragraph("3. PCA Dimensionality Reduction & Analysis", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=8))
    pca_fig = "figures/pca_analysis.png"
    if os.path.exists(pca_fig):
        slides.append(Image(pca_fig, width=680, height=200))
        slides.append(Paragraph("<b>Figure 1:</b> (a) Cumulative explained variance vs principal components. (b) Test reconstruction MSE vs <i>k</i>.", caption_style))
    slides.append(Paragraph("• Fitted strictly on the 70 training samples: $k=32$ principal components captures <b>99.49% of total variance</b> with test reconstruction MSE of $1.15 \\times 10^{-5}$.", body_style))
    slides.append(PageBreak())

    # Slide 5: Model Architectures & Proposed Improvements
    slides.append(Paragraph("4. Model Architecture & Proposed Improvements", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=10))
    slides.append(Paragraph("<b>1. Linear Baseline (Paper ID 0):</b> Linear layer (96 → 32) + Fixed PC Layer.", body_style))
    slides.append(Paragraph("<b>2. Paper MLP Baseline (Paper ID 3):</b> <code>Linear(96,128) → ReLU → Linear(128,256) → ReLU → Linear(256,32) → PC Layer</code>.", body_style))
    slides.append(Spacer(1, 6))
    slides.append(Paragraph("<b>3. Proposed Improvements (ResMLP + Augmentation):</b>", body_style))
    slides.append(Paragraph("• <b>Layer Normalization:</b> Stabilizes hidden activations across poses and prevents internal covariate shift.", bullet_style))
    slides.append(Paragraph("• <b>GELU Activations:</b> Continuous non-linear gating that prevents dead neurons on quaternion coordinates.", bullet_style))
    slides.append(Paragraph("• <b>Residual Skip Connections:</b> Direct identity paths ($h_{l+1} = h_l + \\mathcal{F}(h_l)$) for unobstructed gradient flow.", bullet_style))
    slides.append(Paragraph("• <b>Dropout ($p=0.1$):</b> Prevents feature co-adaptation on high-dimensional deformation manifolds.", bullet_style))
    slides.append(Paragraph("• <b>Pose Jitter Augmentation:</b> Gaussian perturbation ($\sigma = 0.015$) with spherical re-normalization during training.", bullet_style))
    slides.append(PageBreak())

    # Slide 6: Quantitative Benchmarks
    slides.append(Paragraph("5. Quantitative Experimental Results", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=10))
    table_data = [
        ["Model Architecture", "Trainable Params", "Test MSE Loss", "Mean Vertex Err", "Max Vertex Err", "Latency"],
        ["Linear Baseline (Paper ID 0)", "3,104", "1.543e-4", "17.57 mm", "72.56 mm", "0.078 ms"],
        ["Paper MLP Baseline (Paper ID 3)", "53,664", "1.242e-4", "15.95 mm", "83.90 mm", "0.096 ms"],
        ["Improved ResMLP (Ours)", "299,296", "0.851e-4", "11.99 mm", "72.15 mm", "0.299 ms"],
        ["Improved ResMLP + Aug (Ours)", "299,296", "0.802e-4", "11.64 mm", "69.12 mm", "0.236 ms"],
    ]
    t = Table(table_data, colWidths=[200, 95, 95, 95, 95, 80])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5c8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    slides.append(t)
    slides.append(Spacer(1, 12))
    metrics_fig = "figures/metrics_summary.png"
    if os.path.exists(metrics_fig):
        slides.append(Image(metrics_fig, width=680, height=155))
    slides.append(PageBreak())

    # Slide 7: Convergence Curves & Physical Error Heatmaps
    slides.append(Paragraph("6. Training Convergence & Vertex Error Heatmaps", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=8))
    heatmap_fig = "figures/vertex_error_heatmap.png"
    if os.path.exists(heatmap_fig):
        slides.append(Image(heatmap_fig, width=620, height=210))
        slides.append(Paragraph("<b>Figure 2:</b> Normalized squared per-vertex error heatmaps: front (a) and back (b) views.", caption_style))
    slides.append(Paragraph("• <b>Physical Insight:</b> Upper chest/shoulders have low error (< 4 mm); lower loose hem exhibits higher error (up to 69 mm). Exactly mirrors Xue & Wu's coattail findings.", body_style))
    slides.append(PageBreak())

    # Slide 8: Qualitative 3D Cloth Comparison
    slides.append(Paragraph("7. Qualitative 3D Deformation Comparison", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=8))
    mesh_fig = "figures/mesh_comparison_best.png"
    if os.path.exists(mesh_fig):
        slides.append(Image(mesh_fig, width=660, height=210))
        slides.append(Paragraph("<b>Figure 3:</b> 3D Cloth Deformation on unseen test pose: Ground Truth vs Predicted vs Error Overlay.", caption_style))
    slides.append(Paragraph("• Predicted mesh faithfully reconstructs the full garment volume and surface drape with 11.64 mm average error.", body_style))
    slides.append(PageBreak())

    # Slide 9: Live Demo & Execution Instructions
    slides.append(Paragraph("8. Live Demonstration & Verification", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=10))
    slides.append(Paragraph("<b>Interactive Demo Execution:</b>", body_style))
    slides.append(Paragraph("<code>python scripts/demo.py --sample-idx 0 --model-type improved</code>", body_style))
    slides.append(Paragraph("• <b>Real-Time Inference:</b> Evaluates pose quaternions in <b>~0.24 ms</b> (> 4,000 FPS).", bullet_style))
    slides.append(Paragraph("• <b>3D Asset Export:</b> Outputs Wavefront <code>.obj</code> files for inspection in Blender/MeshLab:", bullet_style))
    slides.append(Paragraph("&nbsp;&nbsp;→ <code>outputs/demo_sample_0_pred.obj</code> (Predicted 3D mesh)", bullet_style))
    slides.append(Paragraph("&nbsp;&nbsp;→ <code>outputs/demo_sample_0_gt.obj</code> (Ground truth simulated mesh)", bullet_style))
    slides.append(Paragraph("• <b>Full Pipeline Reproduction:</b> <code>python scripts/run_pipeline.py</code> reproduces all benchmarks and figures from scratch.", bullet_style))
    slides.append(PageBreak())

    # Slide 10: Conclusion & Discussion
    slides.append(Paragraph("9. Conclusion & Limitations", slide_title))
    slides.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2b5c8f"), spaceBefore=2, spaceAfter=10))
    slides.append(Paragraph("<b>Conclusions:</b>", body_style))
    slides.append(Paragraph("• Validated the core hypothesis of Xue & Wu: pose-to-deformation regression with PCA bottleneck provides massive speedups (~100,000×) with high physical fidelity.", bullet_style))
    slides.append(Paragraph("• Our proposed ResMLP with LayerNorm, GELU, and pose jittering achieves <b>35.4% lower MSE</b> and <b>27.0% lower vertex error</b> than the paper baseline.", bullet_style))
    slides.append(Spacer(1, 6))
    slides.append(Paragraph("<b>Limitations & Future Directions:</b>", body_style))
    slides.append(Paragraph("• <b>Quasi-static Assumption:</b> Models single-pose equilibrium; does not capture dynamic velocity or motion inertia across video sequences.", bullet_style))
    slides.append(Paragraph("• <b>Future Work:</b> Incorporate recurrent sequence modeling (GRUs/Transformers) and physics-informed Laplacian surface loss to enforce mesh smoothness.", bullet_style))

    doc.build(slides, canvasmaker=SlideCanvas)
    print(f"Generated clean presentation slide deck at {output_pdf}")


if __name__ == "__main__":
    build_presentation_pdf()
