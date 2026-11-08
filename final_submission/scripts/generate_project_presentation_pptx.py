"""
Generate FloodSense_AI_Project_Presentation.pptx
Professional 20-slide widescreen (16:9) presentation for B.E. Final Year Project Review.
Integrates actual metrics, architecture diagrams, and real embedded screenshots.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def add_styled_run(paragraph, text, bold=False, size_pt=11, color_rgb=None):
    run = paragraph.add_run()
    run.text = text
    run.font.bold = bold
    run.font.size = Pt(size_pt)
    if color_rgb:
        run.font.color.rgb = color_rgb
    return run

def add_header(slide, title_text, category_text="FLOODSENSE-AI FINAL YEAR PROJECT"):
    # Header container
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.0))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0)
    tf.margin_left = Inches(0)
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = RGBColor(2, 132, 199) # Sky blue
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(15, 23, 42) # Slate 900

def add_card(slide, left, top, width, height, bg_rgb=RGBColor(248, 250, 252), border_rgb=RGBColor(226, 232, 240)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_rgb
    shape.line.color.rgb = border_rgb
    shape.line.width = Pt(1)
    return shape

def add_stat_box(slide, left, top, width, height, stat_value, stat_label, accent_rgb=RGBColor(2, 132, 199)):
    shape = add_card(slide, left, top, width, height, bg_rgb=RGBColor(255, 255, 255), border_rgb=RGBColor(203, 213, 225))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0.12)
    tf.margin_left = Inches(0.15)
    
    p_val = tf.paragraphs[0]
    p_val.text = stat_value
    p_val.font.size = Pt(20)
    p_val.font.bold = True
    p_val.font.color.rgb = accent_rgb
    
    p_lbl = tf.add_paragraph()
    p_lbl.text = stat_label
    p_lbl.font.size = Pt(9.5)
    p_lbl.font.color.rgb = RGBColor(71, 85, 105)

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # ------------------ SLIDE 1: TITLE SLIDE ------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(15, 23, 42) # Deep Slate Navy
    bg1.line.fill.background()
    
    tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "FINAL YEAR ACADEMIC PROJECT PRESENTATION • B.E. COMPUTER SCIENCE & ENGINEERING"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(56, 189, 248) # Cyan accent
    
    p1 = tf.add_paragraph()
    p1.text = "FloodSense-AI"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    
    p2 = tf.add_paragraph()
    p2.text = "Multi-Modal Flood Detection, Flood Dynamics, Infrastructure Exposure Mapping and Emergency Rescue Routing"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(28)
    p3.text = "A Zero-Cost, Production-Grade Earth Observation & Emergency Response Decision Support System\n" \
              "Validated on Sen1Floods11 Benchmark & Chennai Adyar Basin Testbed | 61/61 Automated Tests Passing"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(148, 163, 184)

    # ------------------ SLIDE 2: PROBLEM STATEMENT ------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Disaster Problem Statement & Operational Context")
    
    # Card 1: Cloud Cover & Optical Failure
    c1 = add_card(slide2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0))
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.margin_top = Inches(0.3); tf1.margin_left = Inches(0.25); tf1.margin_right = Inches(0.25)
    p = tf1.paragraphs[0]; p.text = "1. Optical Cloud Blindness"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGBColor(185, 28, 28)
    p_b = tf1.add_paragraph(); p_b.space_before = Pt(10)
    p_b.text = "• Extreme precipitation events feature dense, continuous convective cloud cover.\n" \
               "• Optical satellites (Sentinel-2, Landsat) cannot penetrate cloud decks during peak flooding.\n" \
               "• Emergency managers face 48-72 hour blind spots waiting for clear skies."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Card 2: Topological Disconnect
    c2 = add_card(slide2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0))
    tf2 = c2.text_frame; tf2.word_wrap = True
    tf2.margin_top = Inches(0.3); tf2.margin_left = Inches(0.25); tf2.margin_right = Inches(0.25)
    p = tf2.paragraphs[0]; p.text = "2. Tactical Disconnect"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGBColor(194, 65, 12)
    p_b = tf2.add_paragraph(); p_b.space_before = Pt(10)
    p_b.text = "• Academic research models output isolated raster images without spatial referencing.\n" \
               "• Responders cannot tell which specific residential parcels or roads are compromised.\n" \
               "• Standard navigation apps (Google Maps) lack real-time raster flood hazard layers."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Card 3: Commercial & Cloud Barrier
    c3 = add_card(slide2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0))
    tf3 = c3.text_frame; tf3.word_wrap = True
    tf3.margin_top = Inches(0.3); tf3.margin_left = Inches(0.25); tf3.margin_right = Inches(0.25)
    p = tf3.paragraphs[0]; p.text = "3. Commercial Barriers"; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGBColor(79, 70, 229)
    p_b = tf3.add_paragraph(); p_b.space_before = Pt(10)
    p_b.text = "• Commercial GIS and hazard platforms require expensive recurring subscriptions.\n" \
               "• Municipal civil defense teams often lack multi-thousand dollar cloud budgets.\n" \
               "• Solution must run 100% locally on standard workstation GPU hardware with zero cloud fees."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)

    # ------------------ SLIDE 3: OBJECTIVES ------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Project Objectives & Scope of Work")
    
    c_obj = add_card(slide3, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_obj = c_obj.text_frame; tf_obj.word_wrap = True
    tf_obj.margin_top = Inches(0.3); tf_obj.margin_left = Inches(0.4)
    
    objs = [
        ("Multi-Modal Data Ingestion:", " Harmonize Sentinel-1 C-band SAR (VV, VH) with Sentinel-2 MSI optical reflectance onto a common 10m grid."),
        ("Deep Learning Segmentation:", " Implement a Dual-Stream Spatial Cross-Attention UNet categorizing pixels into Dry Land (0), Permanent Water (1), and Floodwater (2)."),
        ("Rigorously Benchmarked:", " Train and evaluate on the official Sen1Floods11 benchmark dataset, strictly separating verified benchmark models from procedural demonstration baselines."),
        ("Temporal Dynamics:", " Isolate dynamic flood expansion from pre-existing water bodies via co-registered SAR delta-backscatter analysis."),
        ("Infrastructure Exposure:", " Automatically query OpenStreetMap building polygons and calculate 3-tier exposure severity (High, Moderate, Low)."),
        ("Directed Road Modeling:", " Build a NetworkX DiGraph strictly adhering to OSM oneway attributes and classify road segments by flood hazard."),
        ("Candidate Rescue Routing:", " Execute a flood-penalized A* algorithm delivering safe evacuation routes and verified kinematic ETAs."),
        ("Zero-Cost Local Architecture:", " Execute the entire pipeline locally on consumer GPU hardware with FastAPI and React dashboard.")
    ]
    for idx, (head, body) in enumerate(objs):
        p = tf_obj.paragraphs[0] if idx == 0 else tf_obj.add_paragraph()
        if idx > 0: p.space_before = Pt(6)
        add_styled_run(p, f"• {head}", bold=True, size_pt=12, color_rgb=RGBColor(15, 23, 42))
        add_styled_run(p, body, bold=False, size_pt=12, color_rgb=RGBColor(71, 85, 105))

    # ------------------ SLIDE 4: REMOTE SENSING PHYSICS ------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Remote Sensing Modalities: SAR C-Band vs. Optical MSI")
    
    # Left Card: Sentinel-1 SAR
    c_sar = add_card(slide4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tfs = c_sar.text_frame; tfs.word_wrap = True
    tfs.margin_top = Inches(0.3); tfs.margin_left = Inches(0.3)
    p = tfs.paragraphs[0]; p.text = "Sentinel-1 Synthetic Aperture Radar (SAR)"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(2, 132, 199)
    p_b = tfs.add_paragraph(); p_b.space_before = Pt(10)
    p_b.text = "• Active Microwave Sensor: C-band (5.405 GHz, λ ≈ 5.6 cm).\n" \
               "• Specular Reflection: Smooth open water acts as a mirror, reflecting radar pulses away from sensor -> Appears dark (low backscatter, -20 to -16 dB).\n" \
               "• Rough Land: Diffuse volumetric backscatter returns signal -> Appears bright (-12 to -6 dB).\n" \
               "• Penetration: Zero cloud occlusion, 24/7 all-weather operational capability.\n" \
               "• Polarizations: Dual-polarization VV (vertical transmit/receive) and VH (cross-polarization)."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Right Card: Sentinel-2 Optical
    c_opt = add_card(slide4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tfo = c_opt.text_frame; tfo.word_wrap = True
    tfo.margin_top = Inches(0.3); tfo.margin_left = Inches(0.3)
    p = tfo.paragraphs[0]; p.text = "Sentinel-2 Multi-Spectral Instrument (MSI)"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(16, 185, 129)
    p_b = tfo.add_paragraph(); p_b.space_before = Pt(10)
    p_b.text = "• Passive Optical Sensor: 13 spectral bands (Visible, NIR, SWIR).\n" \
               "• Water Absorption: Strong absorption in Near-Infrared (NIR) and Shortwave-Infrared (SWIR), high reflectance in Green.\n" \
               "• Multi-Spectral Indices:\n" \
               "    NDWI = (Green - NIR) / (Green + NIR)\n" \
               "    MNDWI = (Green - SWIR) / (Green + SWIR)\n" \
               "• Limitations: Severely blocked by cloud decks and shadows during active precipitation."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)

    # ------------------ SLIDE 5: SYSTEM ARCHITECTURE ------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "End-to-End System Pipeline Architecture")
    
    steps = [
        ("1. Data Ingestion", "Sentinel-1 GRD SAR\nSentinel-2 L2A MSI\nLocal Preset Cache", RGBColor(59, 130, 246)),
        ("2. Preprocessing", "dB Radiometric Cal\nNDWI / MNDWI\n10m Co-registration", RGBColor(14, 165, 233)),
        ("3. AI Segmentation", "Dual-Stream Cross-Attn\nUNet (3-Class Map)\nPyTorch CUDA RT", RGBColor(16, 185, 129)),
        ("4. GIS Overlay", "Overpass OSM Query\nBuilding Exposure\nRoad Intersections", RGBColor(245, 158, 11)),
        ("5. Rescue Routing", "OSM Directed DiGraph\nFlood Hazard Costs\nA* Kinematic Routing", RGBColor(239, 68, 68))
    ]
    for idx, (title, desc, color) in enumerate(steps):
        left = Inches(0.8 + idx * 2.4)
        c = add_card(slide5, left, Inches(1.8), Inches(2.2), Inches(3.5), border_rgb=color)
        tf = c.text_frame; tf.word_wrap = True; tf.margin_top = Inches(0.2); tf.margin_left = Inches(0.15)
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = color
        p_d = tf.add_paragraph(); p_d.space_before = Pt(8); p_d.text = desc; p_d.font.size = Pt(10.5); p_d.font.color.rgb = RGBColor(51, 65, 85)
        
    c_bot = add_card(slide5, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.2), bg_rgb=RGBColor(241, 245, 249))
    tf_b = c_bot.text_frame; tf_b.word_wrap = True; tf_b.margin_top = Inches(0.15); tf_b.margin_left = Inches(0.25)
    p = tf_b.paragraphs[0]; p.text = "FastAPI Asynchronous Backend  <--- Server-Sent Events (SSE) --->  React 18 / Leaflet Web Dashboard"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    p2 = tf_b.add_paragraph(); p2.text = "Complete local offline execution on Windows 11 with NVIDIA RTX 4070 Laptop GPU (8GB VRAM) & Zero External API Subscriptions."; p2.font.size = Pt(10.5); p2.font.color.rgb = RGBColor(71, 85, 105)

    # ------------------ SLIDE 6: DATA PREPROCESSING ------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Radiometric Calibration & Multi-Modal Harmonization")
    
    c_p1 = add_card(slide6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_p1 = c_p1.text_frame; tf_p1.word_wrap = True; tf_p1.margin_top = Inches(0.3); tf_p1.margin_left = Inches(0.3)
    p = tf_p1.paragraphs[0]; p.text = "SAR Radiometric Calibration"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(2, 132, 199)
    p_b = tf_p1.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Sensor Digital Numbers (DN) converted to backscatter (σ° in dB):\n" \
               "    σ°_dB = 10 · log10(DN) - 83.0 dB\n\n" \
               "• Operational dynamic range clipping to [-30.0 dB, 0.0 dB] to suppress extreme speckle spikes.\n\n" \
               "• Min-Max normalization to [0.0, 1.0] for neural network numerical stability:\n" \
               "    VV_norm = (σ°_dB,VV + 30.0) / 30.0\n" \
               "    VH_norm = (σ°_dB,VH + 30.0) / 30.0"
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    c_p2 = add_card(slide6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_p2 = c_p2.text_frame; tf_p2.word_wrap = True; tf_p2.margin_top = Inches(0.3); tf_p2.margin_left = Inches(0.3)
    p = tf_p2.paragraphs[0]; p.text = "Optical & Spatial Harmonization"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(16, 185, 129)
    p_b = tf_p2.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Sentinel-2 L2A BOA surface reflectance normalized by dividing by 10,000.\n\n" \
               "• Multi-Spectral Water Index Computation:\n" \
               "    NDWI = (Band 3 - Band 8) / (Band 3 + Band 8)\n" \
               "    MNDWI = (Band 3 - Band 11) / (Band 3 + Band 11)\n\n" \
               "• Affine Spatial Re-projection: Co-registration to WGS 84 (EPSG:4326) and 10-meter Ground Sample Distance (GSD)."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)

    # ------------------ SLIDE 7: DEEP LEARNING ARCHITECTURE ------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Dual-Stream Spatial Cross-Attention UNet")
    
    c_net = add_card(slide7, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_net = c_net.text_frame; tf_net.word_wrap = True; tf_net.margin_top = Inches(0.3); tf_net.margin_left = Inches(0.4)
    p = tf_net.paragraphs[0]; p.text = "Cross-Modal Spatial Attention Mechanism"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    
    points = [
        ("Modality-Specific Encoders:", "Dedicated convolutional streams extract hierarchical features (channels: 32 -> 64 -> 128 -> 256) without corrupting distinct sensor physical properties."),
        ("Spatial Cross-Attention Gating:", "Computes spatial importance coefficients across modalities: A_SAR = σ(W_SAR * F_SAR + W_OPT * F_OPT + b). Guides noisy SAR speckle using optical spectral boundaries."),
        ("Cross-Fused Feature Bottleneck:", "F_fused = (A_SAR ⊙ F_SAR) ⊕ ((1 - A_SAR) ⊙ F_OPT), dynamically adapting weighting based on local cloud contamination."),
        ("Multi-Scale Decoder:", "Progressive upsampling with skip connections restores crisp 10-meter road and shoreline contours."),
        ("3-Class Semantic Output:", "Softmax output projecting into Class 0 (Dry Land), Class 1 (Permanent Water), and Class 2 (Mapped Floodwater).")
    ]
    for head, body in points:
        p_pt = tf_net.add_paragraph(); p_pt.space_before = Pt(8)
        add_styled_run(p_pt, f"• {head} ", bold=True, size_pt=11.5, color_rgb=RGBColor(2, 132, 199))
        add_styled_run(p_pt, body, bold=False, size_pt=11.5, color_rgb=RGBColor(51, 65, 85))

    # ------------------ SLIDE 8: LOSS & TRAINING ------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Loss Function Formulation & Optimization Strategy")
    
    c_l1 = add_card(slide8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_l1 = c_l1.text_frame; tf_l1.word_wrap = True; tf_l1.margin_top = Inches(0.3); tf_l1.margin_left = Inches(0.3)
    p = tf_l1.paragraphs[0]; p.text = "Hybrid Loss Formulation"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(185, 28, 28)
    p_b = tf_l1.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Severe Class Imbalance: Floodwater represents < 5% of total scene pixels in typical satellite scenes.\n\n" \
               "• Compound Objective Function:\n" \
               "    L_total = L_CE + λ · L_Dice  (with λ = 1.0)\n\n" \
               "• Cross-Entropy Loss (L_CE): Optimizes pixel-wise multi-class probabilistic classification.\n\n" \
               "• Soft Dice Loss (L_Dice): Directly maximizes regional Intersection-over-Union (IoU) overlap and penalizes boundary misalignments."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    c_l2 = add_card(slide8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_l2 = c_l2.text_frame; tf_l2.word_wrap = True; tf_l2.margin_top = Inches(0.3); tf_l2.margin_left = Inches(0.3)
    p = tf_l2.paragraphs[0]; p.text = "Optimization & Hardware Execution"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(2, 132, 199)
    p_b = tf_l2.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Optimizer: AdamW (Initial Learning Rate = 1e-4, Weight Decay = 1e-5).\n\n" \
               "• Learning Rate Scheduling: ReduceLROnPlateau (factor 0.5, patience 3 epochs).\n\n" \
               "• Early Stopping: Triggered after 7 epochs without validation mIoU improvement (Best epoch: 28).\n\n" \
               "• Hardware Acceleration: NVIDIA RTX 4070 Laptop GPU (8GB VRAM) via PyTorch CUDA 12.1."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)

    # ------------------ SLIDE 9: MODEL A VS MODEL B ------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Scientific Model Provenance: Model A vs. Model B")
    
    c_a = add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_a = c_a.text_frame; tf_a.word_wrap = True; tf_a.margin_top = Inches(0.3); tf_a.margin_left = Inches(0.3)
    p = tf_a.paragraphs[0]; p.text = "Model A: Procedural Baseline Checkpoint"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGBColor(100, 116, 139)
    p_b = tf_a.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Checkpoint: dual_stream_attention_unet.pth (155 MB)\n" \
               "• Architecture: Full Dual-Stream Attention UNet\n" \
               "• Parameter Count: 40,610,691 (40.6M params)\n" \
               "• Training Dataset: 24 synthetic procedural tiles (5 epochs)\n" \
               "• Role: Pipeline execution & dashboard interface calibration baseline\n" \
               "• Scientific Integrity Status: NOT an independently evaluated real satellite model."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    c_b = add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    tf_b = c_b.text_frame; tf_b.word_wrap = True; tf_b.margin_top = Inches(0.3); tf_b.margin_left = Inches(0.3)
    p = tf_b.paragraphs[0]; p.text = "Model B: Sen1Floods11 Benchmark Model"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGBColor(16, 185, 129)
    p_b = tf_b.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Checkpoint: best_checkpoint.pth (31.5 MB)\n" \
               "• Architecture: Lightweight Cross-Attention UNet\n" \
               "• Parameter Count: 8,241,731 (8.24M params)\n" \
               "• Training Dataset: Sen1Floods11 official hand-labeled data\n" \
               "• Validation mIoU: 0.7217 (Best Epoch: 28)\n" \
               "• Test mIoU: 0.7198 | Overall Test Accuracy: 95.50%\n" \
               "• Scientific Integrity Status: Formally peer-dataset evaluated benchmark."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)

    # ------------------ SLIDE 10: SEN1FLOODS11 BENCHMARK RESULTS ------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Sen1Floods11 Quantitative Evaluation & Training Curves")
    
    # Left stats
    add_stat_box(slide10, Inches(0.8), Inches(1.8), Inches(2.6), Inches(1.1), "95.50%", "Overall Test Accuracy", RGBColor(16, 185, 129))
    add_stat_box(slide10, Inches(3.6), Inches(1.8), Inches(2.6), Inches(1.1), "0.7198", "Test Mean IoU (mIoU)", RGBColor(2, 132, 199))
    add_stat_box(slide10, Inches(0.8), Inches(3.1), Inches(2.6), Inches(1.1), "0.526", "Class 2 (Flood) IoU", RGBColor(245, 158, 11))
    add_stat_box(slide10, Inches(3.6), Inches(3.1), Inches(2.6), Inches(1.1), "Epoch 28", "Best Validation Epoch", RGBColor(147, 51, 234))
    
    c_desc = add_card(slide10, Inches(0.8), Inches(4.4), Inches(5.4), Inches(2.4))
    tf_d = c_desc.text_frame; tf_d.word_wrap = True; tf_d.margin_top = Inches(0.15); tf_d.margin_left = Inches(0.2)
    p = tf_d.paragraphs[0]; p.text = "Per-Class IoU Breakdown:"; p.font.size = Pt(12); p.font.bold = True
    p2 = tf_d.add_paragraph(); p2.space_before = Pt(4)
    p2.text = "• Class 0 (Dry Land): 0.949 IoU | 0.978 Precision | 0.969 Recall\n" \
              "• Class 1 (Permanent Water): 0.684 IoU | 0.782 Precision\n" \
              "• Class 2 (Mapped Floodwater): 0.526 IoU | 0.674 Precision\n" \
              "Convergence achieved in 28 epochs with zero GPU memory spills."
    p2.font.size = Pt(10.5); p2.font.color.rgb = RGBColor(71, 85, 105)
    
    # Right Image: Training curves
    curves_img = "backend/outputs/sen1floods11_training/training_curves.png"
    if os.path.exists(curves_img):
        slide10.shapes.add_picture(curves_img, Inches(6.5), Inches(1.8), width=Inches(6.0))

    # ------------------ SLIDE 11: STATISTICAL RECONCILIATION ------------------
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Phase 4E.1 Bolivia Holdout Statistical Reconciliation")
    
    c_rec = add_card(slide11, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_r = c_rec.text_frame; tf_r.word_wrap = True; tf_r.margin_top = Inches(0.3); tf_r.margin_left = Inches(0.4)
    p = tf_r.paragraphs[0]; p.text = "Resolution of Historical Discrepancies & Audit Findings"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    
    points_rec = [
        ("Reference Water Pixel Total:", "Direct recalculation from ground truth label rasters verified exactly 386,588 Class 2 flood pixels across all 12 evaluation chips. (Identified and corrected an earlier narrative omission of 9,622 pixels from chip Bolivia_291602)."),
        ("Three-Chip Concentration Percentage:", "The top 3 chips contain 305,186 pixels (Bolivia_129334: 152,073; Bolivia_314919: 83,131; Bolivia_432776: 69,982). Exactly 78.94% of the verified Class 2 total. Corrected previous unverified '>80%' narrative claims."),
        ("Evaluation Pixel Domains:", "Distinguished the 5-chip optical clean subset (1,310,720 valid pixels) from the 12-chip all-weather SAR domain (3,145,728 valid pixels)."),
        ("Scientific Integrity:", "All documentation, JSON manifests, and automated regression tests (test_phase4e_integrity.py) updated to reflect mathematically verified counts.")
    ]
    for head, body in points_rec:
        p_pt = tf_r.add_paragraph(); p_pt.space_before = Pt(8)
        add_styled_run(p_pt, f"• {head} ", bold=True, size_pt=11.5, color_rgb=RGBColor(2, 132, 199))
        add_styled_run(p_pt, body, bold=False, size_pt=11.5, color_rgb=RGBColor(51, 65, 85))

    # ------------------ SLIDE 12: TEMPORAL DYNAMICS ------------------
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Temporal Flood Dynamics & Change Detection Engine")
    
    c_td = add_card(slide12, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf_td = c_td.text_frame; tf_td.word_wrap = True; tf_td.margin_top = Inches(0.3); tf_td.margin_left = Inches(0.3)
    p = tf_td.paragraphs[0]; p.text = "Co-Registered Differential SAR Dynamics"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(2, 132, 199)
    p_b = tf_td.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Formulation: Compares pre-flood baseline (t_pre) with post-flood acquisition (t_post):\n" \
               "    Δσ°_VV = σ°_VV(t_post) - σ°_VV(t_pre)\n\n" \
               "• Newly Inundated Land: Negative backscatter delta (ΔVV < -5.0 dB) indicating specular open water formation.\n\n" \
               "• Permanent Water Mask: High negative backscatter in both scenes; excluded from emergency evacuation alerts."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    c_ts = add_card(slide12, Inches(7.2), Inches(1.8), Inches(5.3), Inches(5.0), bg_rgb=RGBColor(254, 243, 199), border_rgb=RGBColor(245, 158, 11))
    tf_ts = c_ts.text_frame; tf_ts.word_wrap = True; tf_ts.margin_top = Inches(0.3); tf_ts.margin_left = Inches(0.3)
    p = tf_ts.paragraphs[0]; p.text = "Demonstration Preset Safeguards"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(180, 83, 9)
    p_b = tf_ts.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Chennai Preset Testbed: Configured with identical pre- and post-flood SAR tiles as a self-contained execution package.\n\n" \
               "• Software Behavior: Evaluates to ΔVV = 0.0 dB and 0 change pixels.\n\n" \
               "• Transparent System Notification: Emits a documented telemetry notice rather than fabricating synthetic flood evolution.\n\n" \
               "• Provenance Tracked: 100% transparent reporting across all pipeline execution logs."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(120, 53, 15)

    # ------------------ SLIDE 13: GIS BUILDING EXPOSURE ------------------
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "OpenStreetMap GIS Infrastructure Exposure Mapping")
    
    # Left text
    c_bld = add_card(slide13, Inches(0.8), Inches(1.8), Inches(5.4), Inches(5.0))
    tf_b = c_bld.text_frame; tf_b.word_wrap = True; tf_b.margin_top = Inches(0.3); tf_b.margin_left = Inches(0.3)
    p = tf_b.paragraphs[0]; p.text = "Building Exposure Methodology"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    p_b = tf_b.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Vector Acquisition: Live Overpass API queries retrieve OpenStreetMap building polygons.\n\n" \
               "• Spatial Raster Intersection: Building polygons are rasterized onto the 10m flood mask.\n\n" \
               "• 3-Tier Exposure Classification:\n" \
               "    - High Exposure (> 50% flood overlap): 290 buildings\n" \
               "    - Moderate Exposure (10-50% overlap): 4 buildings\n" \
               "    - Low Exposure (< 10% overlap): 706 buildings\n\n" \
               "• Scope Distinction: Measures surface water exposure, not physical structural damage."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Right Image
    bldg_img = "backend/outputs/smoke_test_screenshots/building_exposure_map.png"
    if os.path.exists(bldg_img):
        slide13.shapes.add_picture(bldg_img, Inches(6.5), Inches(1.8), width=Inches(6.0))

    # ------------------ SLIDE 14: ROAD NETWORK HAZARD ------------------
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "Road Network Modeling & One-Way Directed DiGraph")
    
    # Left text
    c_rd = add_card(slide14, Inches(0.8), Inches(1.8), Inches(5.4), Inches(5.0))
    tf_r = c_rd.text_frame; tf_r.word_wrap = True; tf_r.margin_top = Inches(0.3); tf_r.margin_left = Inches(0.3)
    p = tf_r.paragraphs[0]; p.text = "Directed DiGraph Road Modeling"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    p_b = tf_r.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Directed Graph Upgrades: NetworkX DiGraph strictly honors OSM oneway attributes (oneway=yes, -1, no).\n\n" \
               "• Segment Buffering: Physical roadway widths buffered based on highway hierarchy (motorway, trunk, primary, residential).\n\n" \
               "• Segment Hazard Classification (1,000 segments):\n" \
               "    - Road Hazard (> 30% floodwater): 504 segments\n" \
               "    - Caution (10-30% floodwater): 27 segments\n" \
               "    - Clear (< 10% floodwater): 469 segments"
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Right Image
    road_img = "backend/outputs/smoke_test_screenshots/road_hazard_map.png"
    if os.path.exists(road_img):
        slide14.shapes.add_picture(road_img, Inches(6.5), Inches(1.8), width=Inches(6.0))

    # ------------------ SLIDE 15: RESCUE ROUTING ------------------
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "Candidate Emergency Rescue Routing Engine")
    
    # Left text
    c_rt = add_card(slide15, Inches(0.8), Inches(1.8), Inches(5.4), Inches(5.0))
    tf_rt = c_rt.text_frame; tf_rt.word_wrap = True; tf_rt.margin_top = Inches(0.3); tf_rt.margin_left = Inches(0.3)
    p = tf_rt.paragraphs[0]; p.text = "Constrained A* Routing Mechanics"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    p_b = tf_rt.add_paragraph(); p_b.space_before = Pt(8)
    p_b.text = "• Dynamic Evaluation Function: f(n) = g(n) + h(n)\n\n" \
               "• Flood Penalty Edge Weighting:\n" \
               "    cost(u, v) = length · [1.0 + W_flood · P_flood(u, v)]\n" \
               "    (W_flood = 50.0 steers routes away from floodwaters)\n\n" \
               "• Kinematic ETA Calculation (25 km/h convoy speed):\n" \
               "    ETA = (3,469.2 m / 1000) / 25 km/h · 60 = 8.33 minutes\n" \
               "    (Reconciled from uncorrected 9.4 min display)\n\n" \
               "• Candidate Routing Notice: Verifies surface road bypass; does not verify bridge structural integrity."
    p_b.font.size = Pt(11); p_b.font.color.rgb = RGBColor(51, 65, 85)
    
    # Right Image
    route_img = "backend/outputs/smoke_test_screenshots/candidate_rescue_route.png"
    if os.path.exists(route_img):
        slide15.shapes.add_picture(route_img, Inches(6.5), Inches(1.8), width=Inches(6.0))

    # ------------------ SLIDE 16: APPLICATION SMOKE TEST ------------------
    slide16 = prs.slides.add_slide(blank_layout)
    add_header(slide16, "Live Smoke Test & Dashboard Demonstration")
    
    # Left Image
    dash_img = "backend/outputs/smoke_test_screenshots/dashboard_main.png"
    if os.path.exists(dash_img):
        slide16.shapes.add_picture(dash_img, Inches(0.8), Inches(1.8), width=Inches(6.0))
        
    # Right Image
    pipe_img = "backend/outputs/smoke_test_screenshots/pipeline_execution_progress.png"
    if os.path.exists(pipe_img):
        slide16.shapes.add_picture(pipe_img, Inches(7.0), Inches(1.8), width=Inches(5.5))

    # ------------------ SLIDE 17: AREA RECONCILIATION ------------------
    slide17 = prs.slides.add_slide(blank_layout)
    add_header(slide17, "Forensic Flood Area Calculation Reconciliation")
    
    c_ar = add_card(slide17, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_ar = c_ar.text_frame; tf_ar.word_wrap = True; tf_ar.margin_top = Inches(0.3); tf_ar.margin_left = Inches(0.4)
    p = tf_ar.paragraphs[0]; p.text = "Mathematical Resolution of Area Discrepancy"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    
    p_sub = tf_ar.add_paragraph(); p_sub.space_before = Pt(4)
    p_sub.text = "Audit of Chennai Demonstration Raster (1304 x 1015, 210,497 floodwater pixels):"
    p_sub.font.size = Pt(12); p_sub.font.bold = True; p_sub.font.color.rgb = RGBColor(2, 132, 199)
    
    comparisons = [
        ("Row-Integrated WGS84 Geodesic:", "21.049 km² (2,104.93 ha) — Exact physical geodesic calculation accounting for latitudinal convergence."),
        ("Centroid WGS84 Geodesic:", "21.050 km² (2,104.97 ha) — Centroid latitude approximation (13.0039° N). Differs by only 0.04 ha."),
        ("UTM Zone 44N Metric Projection:", "21.025 km² (2,102.46 ha) — Conformal metric projection area."),
        ("Legacy Planar Code Formula:", "21.584 km² (2,158.39 ha) — Earlier reported 21.58 km² was a legacy planar approximation omitting cosine latitude convergence (a 2.5% planar distortion)."),
        ("Final Scientific Action:", "Standardized all reports, README documentation, and viva defense to 21.05 km² geodesic flood extent.")
    ]
    for head, body in comparisons:
        p_pt = tf_ar.add_paragraph(); p_pt.space_before = Pt(8)
        add_styled_run(p_pt, f"• {head} ", bold=True, size_pt=11, color_rgb=RGBColor(15, 23, 42))
        add_styled_run(p_pt, body, bold=False, size_pt=11, color_rgb=RGBColor(71, 85, 105))

    # ------------------ SLIDE 18: TEST SUITES & QA ------------------
    slide18 = prs.slides.add_slide(blank_layout)
    add_header(slide18, "Automated Testing & Software Quality Assurance")
    
    add_stat_box(slide18, Inches(0.8), Inches(1.8), Inches(2.6), Inches(1.1), "61 / 61", "Automated Pytest Tests", RGBColor(16, 185, 129))
    add_stat_box(slide18, Inches(3.6), Inches(1.8), Inches(2.6), Inches(1.1), "100%", "Test Pass Rate", RGBColor(2, 132, 199))
    add_stat_box(slide18, Inches(6.4), Inches(1.8), Inches(2.6), Inches(1.1), "7.59 s", "Total Execution Time", RGBColor(245, 158, 11))
    add_stat_box(slide18, Inches(9.2), Inches(1.8), Inches(3.3), Inches(1.1), "0 Errors", "Frontend Production Build", RGBColor(147, 51, 234))
    
    c_t = add_card(slide18, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6))
    tf_t = c_t.text_frame; tf_t.word_wrap = True; tf_t.margin_top = Inches(0.2); tf_t.margin_left = Inches(0.3)
    p = tf_t.paragraphs[0]; p.text = "Exhaustive Test Suite Breakdown:"; p.font.size = Pt(13); p.font.bold = True
    
    t_suites = [
        ("test_phase6_pipeline.py (19 tests):", "End-to-end pipeline execution, SSE live streaming, execution manifest generation, artifact integrity."),
        ("test_phase5_gis_integrity.py (13 tests):", "OSM directed DiGraph oneway enforcement, A* routing convergence, ETA kinematic formula verification."),
        ("test_phase4e_integrity.py (14 tests):", "Sen1Floods11 Bolivia Class 2 pixel reconciliation (386,588), IoU computation, numerical reproducibility."),
        ("test_flood_change_detection_integrity.py (15 tests):", "Delta-VV differencing, baseline alignment, zero-change safeguard warnings.")
    ]
    for head, body in t_suites:
        p_pt = tf_t.add_paragraph(); p_pt.space_before = Pt(6)
        add_styled_run(p_pt, f"• {head} ", bold=True, size_pt=11, color_rgb=RGBColor(2, 132, 199))
        add_styled_run(p_pt, body, bold=False, size_pt=11, color_rgb=RGBColor(51, 65, 85))

    # ------------------ SLIDE 19: LIMITATIONS & ETHICS ------------------
    slide19 = prs.slides.add_slide(blank_layout)
    add_header(slide19, "Technical Limitations & Ethical Safeguards")
    
    c_lim = add_card(slide19, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf_l = c_lim.text_frame; tf_l.word_wrap = True; tf_l.margin_top = Inches(0.3); tf_l.margin_left = Inches(0.4)
    p = tf_l.paragraphs[0]; p.text = "Explicit Scientific Disclosures & Operational Boundaries"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGBColor(15, 23, 42)
    
    limits = [
        ("Two-Dimensional Surface Inundation:", "Satellite radar observes 2D surface water extent. It cannot directly measure floodwater depth (e.g. 10 cm vs 2 meters) or hydrological current velocity."),
        ("Structural Building Damage:", "Intersection with floodwater indicates exposure, not structural collapse or building uninhabitable status."),
        ("Bridge & Road Structural Integrity:", "A* candidate routes avoid flood-covered segments. Responders are notified that routing does not verify structural load-bearing capacity of bridges or fallen tree obstructions."),
        ("Radar Layover & Urban Canyons:", "High-rise urban structures create radar shadow and double-bounce effects that can obscure narrow streets."),
        ("Zero-Cost Commitment:", "No proprietary cloud services, ensuring reproducibility and sovereignty for emergency responders.")
    ]
    for head, body in limits:
        p_pt = tf_l.add_paragraph(); p_pt.space_before = Pt(8)
        add_styled_run(p_pt, f"• {head} ", bold=True, size_pt=11, color_rgb=RGBColor(185, 28, 28))
        add_styled_run(p_pt, body, bold=False, size_pt=11, color_rgb=RGBColor(51, 65, 85))

    # ------------------ SLIDE 20: CONCLUSION ------------------
    slide20 = prs.slides.add_slide(blank_layout)
    bg20 = slide20.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg20.fill.solid()
    bg20.fill.fore_color.rgb = RGBColor(15, 23, 42)
    bg20.line.fill.background()
    
    tb20 = slide20.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.5))
    tf20 = tb20.text_frame; tf20.word_wrap = True
    
    p0 = tf20.paragraphs[0]
    p0.text = "PROJECT SUMMARY & CONCLUSION"
    p0.font.size = Pt(13); p0.font.bold = True; p0.font.color.rgb = RGBColor(56, 189, 248)
    
    p1 = tf20.add_paragraph()
    p1.text = "FloodSense-AI: Ready for Academic Submission"
    p1.font.size = Pt(36); p1.font.bold = True; p1.font.color.rgb = RGBColor(255, 255, 255)
    
    p2 = tf20.add_paragraph(); p2.space_before = Pt(16)
    p2.text = "• Successfully bridged multi-modal remote sensing with tactical ground emergency response.\n" \
              "• Sen1Floods11 benchmark validation: 95.50% test accuracy, 0.7198 test mIoU.\n" \
              "• End-to-end integration: 21.05 km² flood mapped, 290 high-exposure buildings, 504 road hazards, 8.33 min A* route.\n" \
              "• Zero-cost, 100% offline local GPU execution with 61/61 automated tests passing.\n" \
              "• Full academic deliverables package: Final Report DOCX, PPTX Presentation, Demo Guide, and 56 Viva Q&As."
    p2.font.size = Pt(13); p2.font.color.rgb = RGBColor(226, 232, 240)
    
    p3 = tf20.add_paragraph(); p3.space_before = Pt(24)
    p3.text = "Thank You. Ready for Technical Questions and Live Demonstration."
    p3.font.size = Pt(15); p3.font.bold = True; p3.font.color.rgb = RGBColor(56, 189, 248)
    
    out_dir = "final_submission"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FloodSense_AI_Project_Presentation.pptx")
    prs.save(out_path)
    print(f"Presentation saved successfully to {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == "__main__":
    build_presentation()
