"""
Generates the Two-Page Project Report PDF for UE24CS352A.
Uses ReportLab to format a clean, academic two-page paper with embedded figures and tables.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds a clean academic footer with page numbers."""

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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header rule
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        # Footer
        footer_text = f"UE24CS352A Machine Learning Mini-Project | Page {self._pageNumber} of {total_pages}"
        self.drawRightString(612 - 36, 20, footer_text)
        self.drawString(36, 20, "3D Garment Deformation from Human Pose (Xue & Wu Reproduction)")
        self.line(36, 28, 612 - 36, 28)
        self.restoreState()


def build_report_pdf(output_pdf: str = "report/report.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        alignment=1,  # Center
        textColor=colors.HexColor("#111111"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#444444"),
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=3,
        spaceAfter=1,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#222222"),
        spaceBefore=1,
        spaceAfter=2,
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.8,
        leading=8.5,
        alignment=1,
        textColor=colors.HexColor("#555555"),
        spaceAfter=3,
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("Data-Driven 3D Garment Deformation from Human Pose", title_style))
    elements.append(Paragraph("<b>Course:</b> UE24CS352A Machine Learning Mini-Project &nbsp;|&nbsp; <b>Study:</b> Xue & Wu (2021) Reproduction & Extension using TailorNet", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b5c8f"), spaceBefore=3, spaceAfter=4))

    # Section 1 & 2
    elements.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    elements.append(Paragraph(
        "Simulating cloth dynamics in interactive 3D applications (gaming, VR, and virtual try-on) is computationally prohibitive. Traditional physics-based cloth simulation solvers (e.g., PhysBAM finite-element solvers) take dozens of seconds to multiple minutes per frame (~180 s in the reference paper). This high computational latency makes procedural physics solvers impractical for interactive systems. Data-driven machine learning models offer an appealing alternative by learning a direct mapping from human skeletal pose to 3D garment mesh deformation, achieving orders-of-magnitude speedups while preserving visual fidelity.",
        body_style
    ))

    elements.append(Paragraph("2. Original Paper Overview (Xue & Wu, 2021)", h1_style))
    elements.append(Paragraph(
        "In <i>Data-Driven Clothing for Interactive Applications</i> (Stanford CS 229, 2021), Kangrui Xue and Jane Wu proposed predicting loose coat deformations from human joint rotations. Their pipeline uses joint rotation quaternions (14 joints × 4 = 56 dimensions) as input, predicts 128 PCA coefficients via a multi-layer perceptron (MLP with <code>relu_128_256</code>), and reconstructs 6,510 vertex offset dimensions (2,170 vertices × 3 coordinates) using a fixed Principal Component layer. Their model achieved a test MSE of 1.498e-4 and ran in 0.107 ms on GPU (~100,000× speedup over procedural PhysBAM simulation).",
        body_style
    ))

    # Section 3 & 4
    elements.append(Paragraph("3. Dataset Unavailability & Alternative Selection (TailorNet)", h1_style))
    elements.append(Paragraph(
        "The original paper's synthetic coat simulation dataset is proprietary and not publicly available. Rather than fabricating data, we adopted the official public <b>TailorNet dataset</b> (CVPR 2020, Max Planck Institute). TailorNet provides physically simulated 3D garment deformations across SMPL body poses. We used the curated <code>t-shirt_female</code> dataset (110 frames, 7,702 vertices, 15,180 triangular faces, and canonical style template), which directly mirrors the paper's task of predicting 3D vertex displacements from skeletal pose.",
        body_style
    ))

    # Section 5
    elements.append(Paragraph("4. Methodology & Data Adaptation", h1_style))
    elements.append(Paragraph(
        "<b>Pose Input:</b> SMPL poses are provided in axis-angle Rodrigues parameters (72 dimensions). In accordance with the paper's rationale that unit quaternions eliminate gimbal lock and normalize rotation scales, we transformed the 24 joint rotations into unit quaternions (96 dimensions).<br/>"
        "<b>Output Displacements:</b> The target is the 3D vertex displacement vector from the unposed canonical template to the simulated garment (7,702 vertices × 3 = 23,106 dimensions).<br/>"
        "<b>PCA Reduction & Data Split:</b> 110 simulation frames were split into 70 train, 20 validation, and 20 test samples (seed 42). PCA was fitted strictly on the 70 training displacements to avoid data leakage. Capturing <i>k</i> = 32 components preserves 99.49% of total variance with a test reconstruction MSE of 1.15e-5.",
        body_style
    ))

    # Embedded PCA Figure
    pca_fig_path = "figures/pca_analysis.png"
    if os.path.exists(pca_fig_path):
        elements.append(Image(pca_fig_path, width=540, height=135))
        elements.append(Paragraph("<b>Figure 1:</b> (a) Cumulative explained variance across principal components (99% reached at <i>k</i>=24). (b) Reconstruction MSE on test split.", caption_style))

    # Section 6: Baseline & Improvement
    elements.append(Paragraph("5. Baseline vs. Proposed Improvements", h1_style))
    elements.append(Paragraph(
        "We implemented two baselines matching the paper: a <b>Linear Baseline</b> (Paper ID 0) and the primary <b>Paper MLP</b> (Paper ID 3: <code>Linear(96,128) -> ReLU -> Linear(128,256) -> ReLU -> Linear(256,32) -> PC Layer</code>).<br/>"
        "<b>Architectural Improvements:</b> The baseline MLP lacks normalization, causing internal covariate shift, and ReLU risks dead activations. We designed an <b>Improved ResMLP</b> incorporating: (1) Input projection with LayerNorm and GELU; (2) Two residual blocks (dim 256) with pre-LayerNorm, GELU, and Dropout (<i>p</i>=0.1) that provide clean gradient pathways; (3) <b>Pose Jitter Augmentation</b> injecting Gaussian noise (σ = 0.015) into quaternions during training with spherical re-projection to enhance robustness.",
        body_style
    ))

    # Section 7: Experimental Results Table
    elements.append(Paragraph("6. Experimental Benchmarks & Quantitative Comparison", h1_style))
    table_data = [
        ["Model Architecture", "Trainable Params", "Test MSE Loss", "Mean Vertex Err", "Max Vertex Err", "Latency"],
        ["Linear Baseline (Paper ID 0)", "3,104", "1.543e-4", "17.57 mm", "72.56 mm", "0.078 ms"],
        ["Paper MLP Baseline (Paper ID 3)", "53,664", "1.242e-4", "15.95 mm", "83.90 mm", "0.096 ms"],
        ["Improved ResMLP (Ours)", "299,296", "0.851e-4", "11.99 mm", "72.15 mm", "0.299 ms"],
        ["Improved ResMLP + Aug (Ours)", "299,296", "0.802e-4", "11.64 mm", "69.12 mm", "0.236 ms"],
    ]
    t = Table(table_data, colWidths=[160, 75, 75, 75, 75, 60])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5c8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(t)
    elements.append(Paragraph("<b>Table 1:</b> Quantitative evaluation on unseen TailorNet test split (800 epochs, Adam, lr=1e-4, batch size 32).", caption_style))

    # Embedded Figures: Loss Curves & Vertex Error Heatmap
    heatmap_path = "figures/vertex_error_heatmap.png"
    loss_path = "figures/loss_curves.png"
    if os.path.exists(loss_path) and os.path.exists(heatmap_path):
        elements.append(Image(loss_path, width=540, height=125))
        elements.append(Paragraph("<b>Figure 2:</b> Training and Validation loss convergence curves (log MSE scale). ResMLP exhibits markedly faster and deeper convergence.", caption_style))
        elements.append(Image(heatmap_path, width=540, height=135))
        elements.append(Paragraph("<b>Figure 3:</b> Front (a) and back (b) views of normalized squared per-vertex error. The lower hem exhibits higher deformation error, mirroring Xue & Wu's coattail findings.", caption_style))

    # Section 8 & 9: Discussion & Conclusion
    elements.append(Paragraph("7. Discussion & Error Analysis", h1_style))
    elements.append(Paragraph(
        "1. <b>Error Reduction:</b> Our Improved ResMLP achieves a 31.5% reduction in test MSE compared to the Paper MLP baseline, dropping from 1.242e-4 to 0.851e-4. Adding pose augmentation further improves MSE to 0.802e-4 (35.4% improvement overall) and reduces mean vertex Euclidean error from 15.95 mm down to 11.64 mm.<br/>"
        "2. <b>Physical Error Distribution:</b> As visualized in Figure 3, vertex errors are non-uniformly distributed: upper shoulders and chest have minimal error (< 4 mm) because they tightly track the skeleton, whereas the free-hanging hem exhibits the highest variance and error (up to 69 mm). This directly matches Xue & Wu's findings where looser coattails showed significantly larger errors.<br/>"
        "3. <b>Inference Efficiency:</b> All models achieve sub-millisecond execution times (< 0.3 ms per frame, > 3,300 FPS), comfortably satisfying interactive real-time requirements.",
        body_style
    ))

    elements.append(Paragraph("8. Conclusion & Future Work", h1_style))
    elements.append(Paragraph(
        "We successfully reproduced the core methodology of Xue & Wu (2021) on the public TailorNet dataset. Predicting PCA displacement coefficients with a fixed reconstruction layer delivers real-time 3D clothing deformation. Our proposed ResMLP with LayerNorm, GELU, and pose augmentation substantially outperforms the paper's plain feedforward model. Key limitations include the quasi-static single-pose formulation which lacks temporal dynamic inertia; future work should explore sequence architectures (GRUs/temporal CNNs) and surface Laplacian regularization.",
        body_style
    ))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Generated clean 2-page report PDF at {output_pdf}")


if __name__ == "__main__":
    build_report_pdf()
