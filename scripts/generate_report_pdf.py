"""
Generates the Two-Page Project Report PDF for UE24CS352A.
Academic, black-and-white publication layout in Times-Roman.
Contains thorough methodology, formulas, tables, figures, and benchmark metrics.
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
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class SimplePageNumberedCanvas(canvas.Canvas):
    """Adds a clean, centered page number at the bottom of each page."""

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
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, total_pages):
        self.saveState()
        self.setFont("Times-Roman", 8.5)
        self.setFillColor(colors.black)
        # Clean page number centered at the bottom, no extra header/footer text
        page_str = f"{self._pageNumber}"
        self.drawCentredString(612 / 2.0, 20, page_str)
        self.restoreState()


def build_report_pdf(output_pdf: str = "report.pdf"):
    # Letter dimensions: 612 x 792 pt. Margins: 32 pt left/right, 26 pt top/bottom.
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=26,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=12.5,
        leading=14.5,
        alignment=1,
        textColor=colors.black,
        spaceAfter=1,
    )
    meta_style = ParagraphStyle(
        "PaperMeta",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.0,
        leading=10.0,
        alignment=1,
        textColor=colors.black,
        spaceAfter=2,
    )
    sec_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=8.6,
        leading=10.4,
        textColor=colors.black,
        spaceBefore=2.5,
        spaceAfter=1.0,
    )
    body_style = ParagraphStyle(
        "BodyTextBlack",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.black,
        spaceBefore=0.5,
        spaceAfter=1.5,
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=6.4,
        leading=7.8,
        alignment=1,
        textColor=colors.black,
        spaceBefore=1.0,
        spaceAfter=2.0,
    )

    cell_hdr = ParagraphStyle(
        "CellHdr",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=6.5,
        leading=7.5,
        alignment=1,
        textColor=colors.black,
    )
    cell_body = ParagraphStyle(
        "CellBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=6.4,
        leading=7.4,
        alignment=1,
        textColor=colors.black,
    )
    cell_body_bold = ParagraphStyle(
        "CellBodyBold",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=6.4,
        leading=7.4,
        alignment=1,
        textColor=colors.black,
    )

    elements = []

    # ==================== PAGE 1 ====================
    # Title & Metadata
    elements.append(Paragraph("Data-Driven 3D Garment Deformation from Human Pose", title_style))
    elements.append(Paragraph(
        "Course: UE24CS352A Machine Learning Mini-Project &nbsp;|&nbsp; "
        "Reproduction & Extension of Xue & Wu (Stanford CS 229, 2021) on Multi-Sequence TailorNet",
        meta_style
    ))
    elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceBefore=1, spaceAfter=2.5))

    # 1. Problem Statement & Motivation
    elements.append(Paragraph("1. Problem Statement & Motivation", sec_style))
    elements.append(Paragraph(
        "Simulating realistic cloth dynamics in interactive applications (gaming, virtual reality, and digital try-on) is computationally prohibitive. "
        "Standard finite-element physical simulation solvers (such as PhysBAM) require solving large non-linear partial differential equations per time step, "
        "consuming dozens of seconds to several minutes per frame (~180 s in the reference paper). This high computational cost prevents physics solvers from "
        "operating at interactive frame rates (30&ndash;60 FPS). "
        "Data-driven machine learning offers an effective solution: by training a regression model to map skeletal joint rotations directly to 3D surface "
        "mesh offsets, inference runs in sub-millisecond latency (< 0.55 ms, > 1,800 FPS) while preserving visual cloth wrinkle formations.",
        body_style
    ))

    # 2. Reference Paper Overview
    elements.append(Paragraph("2. Reference Paper Overview (Xue & Wu, 2021)", sec_style))
    elements.append(Paragraph(
        "In <i>Data-Driven Clothing for Interactive Applications</i> (Stanford CS 229), Kangrui Xue and Jane Wu proposed predicting loose coat deformations "
        "from human body joint rotations. Their pipeline extracts joint rotation quaternions (14 joints &times; 4 = 56 dimensions), predicts 128 Principal "
        "Component Analysis (PCA) coefficients through a feedforward Multi-Layer Perceptron (MLP with <code>relu_128_256</code>), and reconstructs 6,510 "
        "vertex displacement dimensions (2,170 vertices &times; 3 coordinates) using a fixed linear Principal Component (PC) reconstruction layer. "
        "Their model reached a test MSE of 1.498 &times; 10<sup>-4</sup> and executed in 0.107 ms on GPU (~100,000&times; faster than PhysBAM).",
        body_style
    ))

    # 3. Dataset Selection & Scaling
    elements.append(Paragraph("3. Dataset Selection & Multi-Sequence Scaling (TailorNet Benchmark)", sec_style))
    elements.append(Paragraph(
        "The coat simulation dataset used in the reference paper is proprietary and unavailable. To ensure rigorous academic reproducibility, we adopted "
        "the official public <b>TailorNet benchmark</b> (CVPR 2020, Max Planck Institute). Unlike preliminary prototype setups restricted to a single 110-frame "
        "clip, our pipeline streams and extracts all <b>18 motion sequences</b> (sequences <code>000</code> to <code>017</code>, totaling <b>2,122 simulation frames</b>) "
        "for garment style/shape <code>000_023</code>. The garment mesh consists of <b>7,702 vertices</b> and <b>15,180 triangular faces</b> (23,106 displacement "
        "coordinates). The motion sequences encompass diverse human motions including walking, running, jumping, bending, and stretching, providing dynamic "
        "cloth fold deformations across varied joint angle ranges.",
        body_style
    ))

    # 4. Mathematical Methodology & PCA Ablation
    elements.append(Paragraph("4. Mathematical Formulation & Systematic PCA Dimensionality (<i>k</i>) Ablation", sec_style))
    elements.append(Paragraph(
        "<b>Input Pose Representation:</b> SMPL poses are provided in 72-dimensional Rodrigues axis-angle coordinates. In accordance with the paper's rationale, "
        "we convert the 24 joint rotations into unit quaternions <i><b>q</b></i> &isin; <b>H</b> (96 dimensions). Unit quaternions eliminate gimbal lock singularities, "
        "guarantee continuous rotation representations, and standardize input feature magnitude to ||<i><b>q</b></i>||<sub>2</sub> = 1.<br/>"
        "<b>Output Displacements & Fixed PC Layer:</b> Let <i>V</i><sub>0</sub> &isin; <b>R</b><sup>7702 &times; 3</sup> denote the unposed canonical template mesh, and <i>V<sub>t</sub></i> "
        "the deformed cloth mesh. The displacement target is <i>y<sub>t</sub></i> = vec(<i>V<sub>t</sub></i> &minus; <i>V</i><sub>0</sub>) &isin; <b>R</b><sup>23106</sup>. We fit PCA strictly on training displacements: "
        "<i>y&#770;</i> = <i>c&#770;</i> <i>U</i> + &mu;, where <i>U</i> &isin; <b>R</b><sup><i>k</i> &times; 23106</sup> contains the top <i>k</i> principal component eigenvectors, "
        "&mu; &isin; <b>R</b><sup>23106</sup> is the mean training offset, and <i>c&#770;</i> &isin; <b>R</b><sup><i>k</i></sup> are network-predicted coefficients.<br/>"
        "<b>Data Partition:</b> The 2,122 frames were partitioned into 1,697 training (80%), 212 validation (10%), and 213 test frames (10%) with fixed seed 42. "
        "PCA was fitted strictly on the 1,697 training frames to prevent data leakage.<br/>"
        "<b>Why Early Prototypes Used <i>k</i> = 32:</b> In small 110-frame prototype runs (70 training frames), the sample covariance matrix had empirical rank "
        "&le; min(<i>N</i>, <i>D</i>) = 70. Attempting <i>k</i> = 128 was rank-deficient and caused severe numerical overfitting. With 1,697 training frames, we performed a systematic "
        "ablation across <i>k</i> &isin; {32, 64, 128, 256} using the paper's MLP architecture (Table 1):",
        body_style
    ))

    # Table 1: PCA Ablation Table (Booktabs academic style: clean black rules, white background)
    pca_raw_data = [
        ["k", "Expl. Var (%)", "Direct PCA MSE", "Direct Err (mm)", "Model Test MSE", "Mean Err (mm)", "Max Err (mm)", "Trainable Params"],
        ["32", "95.85%", "1.758 &times; 10<sup>-5</sup>", "6.00 mm", "8.366 &times; 10<sup>-5</sup>", "13.50 mm", "103.85 mm", "53,664"],
        ["64", "97.70%", "1.044 &times; 10<sup>-5</sup>", "4.54 mm", "8.120 &times; 10<sup>-5</sup>", "13.31 mm", "103.55 mm", "61,888"],
        ["128", "98.84%", "5.877 &times; 10<sup>-6</sup>", "3.39 mm", "<b>8.112 &times; 10<sup>-5</sup></b>", "<b>13.24 mm</b>", "108.38 mm", "78,336"],
        ["256", "99.52%", "2.934 &times; 10<sup>-6</sup>", "2.40 mm", "8.258 &times; 10<sup>-5</sup>", "13.42 mm", "107.90 mm", "111,232"],
    ]

    t1_cells = []
    for r_idx, row in enumerate(pca_raw_data):
        row_cells = []
        for c_idx, val in enumerate(row):
            if r_idx == 0:
                row_cells.append(Paragraph(val, cell_hdr))
            elif r_idx == 3:
                row_cells.append(Paragraph(val, cell_body_bold))
            else:
                row_cells.append(Paragraph(val, cell_body))
        t1_cells.append(row_cells)

    t1 = Table(t1_cells, colWidths=[24, 72, 76, 68, 76, 68, 68, 76])
    t1.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    elements.append(t1)
    elements.append(Paragraph(
        "<b>Table 1:</b> PCA dimensionality ablation study. <i>k</i> = 128 achieves lowest Model Test MSE (8.112 &times; 10<sup>-5</sup>) and lowest mean geometric error (13.24 mm), "
        "confirming that <i>k</i> = 128 optimally balances high-frequency wrinkle retention with network generalization.",
        caption_style
    ))

    # Embedded PCA Figure
    pca_fig = "figures/pca_analysis.png"
    if os.path.exists(pca_fig):
        elements.append(Image(pca_fig, width=540, height=100))
        elements.append(Paragraph(
            "<b>Figure 1:</b> (a) Cumulative explained variance across principal components (98.84% at <i>k</i>=128). "
            "(b) Linear reconstruction MSE and mean vertex error curves illustrating diminishing returns beyond <i>k</i>=128.",
            caption_style
        ))

    # ==================== PAGE BREAK ====================
    elements.append(PageBreak())

    # ==================== PAGE 2 ====================
    # 5. Baseline vs Proposed Architecture
    elements.append(Paragraph("5. Baseline vs. Proposed Architectural Improvements", sec_style))
    elements.append(Paragraph(
        "We implemented two reference paper baselines: a <b>Linear Baseline</b> (Paper ID 0, 12,416 params) and the primary <b>Paper MLP</b> "
        "(Paper ID 3: <code>Linear(96,128) &rarr; ReLU &rarr; Linear(128,256) &rarr; ReLU &rarr; Linear(256,128) &rarr; PC Layer</code>, 78,336 params).<br/>"
        "<b>Limitations of Paper MLP:</b> The baseline MLP lacks normalization, causing internal covariate shift during training; utilizes standard ReLU, "
        "which risks dead neuron activations; and exhibits degradation when model depth is scaled.<br/>"
        "<b>Proposed Improved ResMLP (Ours):</b> We designed a deep residual architecture (323,968 params) with the following key components:<br/>"
        "&bull; <i>Input Projection:</i> Linear(96, 256) followed by LayerNorm and GELU non-linearities.<br/>"
        "&bull; <i>Pre-LN Residual Blocks:</i> Two residual blocks with identity skip connections: <i>x</i><sub><i>l</i>+1</sub> = <i>x<sub>l</sub></i> + <i>W</i><sub>2</sub>(Dropout(GELU(<i>W</i><sub>1</sub>(LN(<i>x<sub>l</sub></i>))))), "
        "where hidden dimension is 256 and dropout <i>p</i> = 0.1. Pre-LayerNorm ensures clean gradient propagation back to initial layers.<br/>"
        "&bull; <i>Output Head:</i> Linear(256, 128) predicting PCA coefficients feeding into the fixed PC reconstruction buffer.<br/>"
        "&bull; <i>Pose Jitter Augmentation:</i> During training, isotropic Gaussian noise is injected into input quaternions: <i>q'</i> = (<i>q</i> + &epsilon;) / ||<i>q</i> + &epsilon;||<sub>2</sub>, "
        "where &epsilon; ~ <i>N</i>(0, &sigma;<sup>2</sup><i>I</i>) with &sigma; = 0.015, followed by unit spherical normalization to regularize pose interpolation.<br/>"
        "&bull; <i>Training Optimization:</i> Trained with Adam optimizer (initial learning rate 10<sup>-4</sup>, weight decay 10<sup>-5</sup>, batch size 64) with "
        "ReduceLROnPlateau scheduler (factor 0.5, patience 5) and early stopping on validation loss (patience 15).",
        body_style
    ))

    # 6. Experimental Benchmarks Table (Booktabs academic style)
    elements.append(Paragraph("6. Experimental Benchmarks & Quantitative Comparison (<i>k</i> = 128)", sec_style))
    eval_raw_data = [
        ["Model Architecture", "Trainable Params", "Test MSE Loss", "Mean Vertex Err", "Max Vertex Err", "Inference Latency"],
        ["Linear Baseline (Paper ID 0)", "12,416", "1.188 &times; 10<sup>-4</sup>", "15.81 mm", "101.13 mm", "0.261 ms"],
        ["Paper MLP Baseline (Paper ID 3)", "78,336", "7.822 &times; 10<sup>-5</sup>", "13.02 mm", "98.55 mm", "0.339 ms"],
        ["Improved ResMLP (Ours)", "323,968", "<b>4.500 &times; 10<sup>-5</sup></b>", "<b>9.66 mm</b>", "81.05 mm", "0.482 ms"],
        ["Improved ResMLP + Aug (Ours)", "323,968", "4.543 &times; 10<sup>-5</sup>", "9.83 mm", "<b>75.43 mm</b>", "0.549 ms"],
    ]

    t2_cells = []
    for r_idx, row in enumerate(eval_raw_data):
        row_cells = []
        for c_idx, val in enumerate(row):
            if r_idx == 0:
                row_cells.append(Paragraph(val, cell_hdr))
            elif r_idx >= 3:
                row_cells.append(Paragraph(val, cell_body_bold))
            else:
                row_cells.append(Paragraph(val, cell_body))
        t2_cells.append(row_cells)

    t2 = Table(t2_cells, colWidths=[150, 75, 75, 75, 75, 78])
    t2.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    elements.append(t2)
    elements.append(Paragraph(
        "<b>Table 2:</b> Final benchmark on unseen 213-frame test split (Adam, lr=1e-4, batch size 64, early stopping patience 15). "
        "Our Improved ResMLP reduces test MSE by 42.5% and brings mean vertex error to sub-centimeter accuracy (9.66 mm).",
        caption_style
    ))

    # Embedded Figures: Loss Curves & Heatmap
    loss_fig = "figures/loss_curves.png"
    if os.path.exists(loss_fig):
        elements.append(Image(loss_fig, width=540, height=84))
        elements.append(Paragraph(
            "<b>Figure 2:</b> Training and Validation loss convergence across 100 epochs (log MSE scale). "
            "ResMLP demonstrates faster learning dynamics and reaches lower asymptotic validation error without overfitting.",
            caption_style
        ))

    heatmap_fig = "figures/vertex_error_heatmap.png"
    if os.path.exists(heatmap_fig):
        elements.append(Image(heatmap_fig, width=540, height=90))
        elements.append(Paragraph(
            "<b>Figure 3:</b> Spatial distribution of vertex deformation error (mm) across garment mesh. "
            "Tightly constrained shoulder and collar regions exhibit minimal error (< 4 mm), whereas unconstrained bottom hem folds exhibit larger variance.",
            caption_style
        ))

    # 7. Discussion & Error Analysis
    elements.append(Paragraph("7. Discussion & Detailed Error Analysis", sec_style))
    elements.append(Paragraph(
        "<b>1. Significant Accuracy Gains:</b> Our Improved ResMLP achieves a <b>42.5% reduction in Test MSE</b> (4.500 &times; 10<sup>-5</sup> vs 7.822 &times; 10<sup>-5</sup>) "
        "and lowers mean 3D Euclidean vertex error from 13.02 mm to <b>9.66 mm</b> (a 25.8% reduction), achieving sub-centimeter deformation precision.<br/>"
        "<b>2. Outlier Suppression via Augmentation:</b> Pose Jitter Augmentation regularizes the manifold, reducing maximum peak vertex error "
        "from 98.55 mm to <b>75.43 mm (23.5% reduction in peak error)</b>, preventing unphysical mesh stretching on unseen extreme poses.<br/>"
        "<b>3. Physical Error Distribution:</b> As evidenced in Figure 3, errors correlate directly with cloth freedom-of-movement: upper chest and "
        "shoulder fabric adheres rigidly to the skeleton, while the bottom hem hangs freely under gravity, directly replicating Xue & Wu's coattail findings.<br/>"
        "<b>4. Real-Time Framerates:</b> Inference latency is 0.482 ms on GPU and ~16.8 ms on CPU (> 1,800 FPS batched), comfortably exceeding 60 FPS VR/game requirements.",
        body_style
    ))

    # 8. Conclusion & Future Work
    elements.append(Paragraph("8. Conclusion & Future Work", sec_style))
    elements.append(Paragraph(
        "We successfully reproduced Xue & Wu (2021) on the public 18-sequence TailorNet benchmark (2,122 frames). "
        "Systematic ablation established <i>k</i> = 128 as the optimal PCA subspace. Our proposed ResMLP with LayerNorm, GELU, residual skips, and pose "
        "augmentation dramatically outperforms the reference MLP baseline, delivering sub-centimeter 3D deformation at real-time speeds. "
        "Future extensions could incorporate temporal recurrence (GRU/Transformer) to model dynamic inertia and edge-spring physics loss terms.",
        body_style
    ))

    doc.build(elements, canvasmaker=SimplePageNumberedCanvas)
    print(f"Successfully generated clean 2-page black-and-white report PDF at: {output_pdf}")


if __name__ == "__main__":
    build_report_pdf()
